"""Trigger word tests: the child protocol, the phrase table, and what the
daemon does with each kind.

The rules that decide whether always-on listening is usable rather than
merely present all live here: the trigger word comes back off the transcript,
"type" never presses Enter, "send it" only ever presses Enter on something
Bol pasted, Bol is deaf while it speaks, a pause is a pause until the key
comes back, and a recording nobody asked for gives the microphone up.

Nothing here imports sherpa-onnx or sentencepiece. CI installs base
dependencies only on some machines, and the whole point of putting the
keyword model in a child process is that the daemon's own tests never have to
load it: the spotter, the BPE encoder, and the child process are all fakes.
"""

import asyncio
import io
import logging
import tarfile
import types

import numpy as np
import pytest

import bol.cli as cli
import bol.daemon as daemon_mod
import bol.wake as wake
from bol.bridge.focused import SubmitBlocked
from bol.config import Config, validate_config
from bol.daemon import Daemon
from bol.wake import WakeListener, model as wake_model
from bol.wake.listener import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    RESET_GAP_S,
    WakeError,
    keywords_text,
    parse_phrase_arg,
    read_frame,
    run,
)


from test_daemon_loop import (
    FakeBridge,
    FakeHud,
    FakeRecorder,
    FakeSpeaker,
    FakeTranscriber,
)


@pytest.fixture(autouse=True)
def no_osascript(monkeypatch):
    """Nothing in this file may shell out to read the frontmost app.

    Every recording a trigger word starts watches for the user going
    somewhere else, and that watcher runs an osascript. Left real, these
    tests would spawn a subprocess per recording and depend on which window
    happens to be in front of the machine running them.
    """

    async def frontmost():
        return "com.apple.Terminal"

    monkeypatch.setattr(daemon_mod, "frontmost_bundle_id", frontmost)


class Clock:
    """Monotonic seconds the test moves by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _frames(count: int) -> bytes:
    return b"\x00" * (FRAME_BYTES * count)


# ------------------------------------------------------------ the child's loop


class FakeStream:
    def __init__(self) -> None:
        self.blocks = []

    def accept_waveform(self, rate, samples):
        self.blocks.append((rate, np.asarray(samples)))


class FakeSpotter:
    """The KeywordSpotter API the child actually uses, and nothing else."""

    def __init__(self, hits=()) -> None:
        # Frame numbers (1-based) at which get_result reports a keyword.
        self.hits = set(hits)
        self.frames = 0
        self.resets = 0
        self.stream = None

    def create_stream(self):
        self.stream = FakeStream()
        return self.stream

    def is_ready(self, stream) -> bool:
        # One decode step per frame, then the stream wants more audio.
        return len(stream.blocks) > self.frames

    def decode_stream(self, stream) -> None:
        self.frames += 1

    def get_result(self, stream) -> str:
        return "HEY BOL" if self.frames in self.hits else ""

    def reset_stream(self, stream) -> None:
        self.resets += 1


def test_the_child_writes_ready_then_one_line_per_wake():
    spotter = FakeSpotter(hits={2})
    sink = io.StringIO()

    code = run(io.BytesIO(_frames(4)), sink, spotter, 0.12, clock=lambda: 0.0)

    assert code == 0
    assert sink.getvalue() == "ready\nwake 0.12 hey bol\n"


def test_the_child_feeds_every_frame_as_512_samples():
    spotter = FakeSpotter()
    run(io.BytesIO(_frames(3)), io.StringIO(), spotter, 0.12, clock=lambda: 0.0)

    assert len(spotter.stream.blocks) == 3
    rate, samples = spotter.stream.blocks[0]
    assert rate == 16000
    assert samples.shape == (FRAME_SAMPLES,)
    assert samples.dtype == np.float32


def test_the_child_resets_its_decoder_after_a_muted_gap():
    # The decoder is recurrent. Bol speaks, the daemon stops forwarding, and
    # the audio on either side of that silence is not one utterance: decoding
    # it as one is exactly how Bol wakes itself on its own voice.
    clock = Clock(0.0)
    ticks = [0.0, 0.032, 0.032 + RESET_GAP_S + 0.1, 0.032]

    def tick():
        if ticks:
            clock.advance(ticks.pop(0))
        return clock.now

    spotter = FakeSpotter()
    run(io.BytesIO(_frames(3)), io.StringIO(), spotter, 0.12, clock=tick)

    assert spotter.resets == 1


def test_a_steady_stream_never_resets_the_decoder():
    clock = Clock(0.0)

    def tick():
        clock.advance(0.032)
        return clock.now

    spotter = FakeSpotter()
    run(io.BytesIO(_frames(5)), io.StringIO(), spotter, 0.12, clock=tick)

    assert spotter.resets == 0


def test_a_frame_split_across_two_pipe_reads_is_reassembled():
    # A pipe read returns whatever has arrived. A short read taken as a whole
    # frame would shift every later frame by a few samples, forever.
    class Dribble:
        def __init__(self, data, chunk):
            self.data = data
            self.chunk = chunk

        def read(self, want):
            take = min(want, self.chunk, len(self.data))
            out, self.data = self.data[:take], self.data[take:]
            return out

    frame = read_frame(Dribble(_frames(1), chunk=7))

    assert frame is not None
    assert len(frame) == FRAME_BYTES


def test_a_half_frame_at_the_end_of_the_stream_is_not_decoded():
    spotter = FakeSpotter()
    source = io.BytesIO(_frames(2) + b"\x00" * 16)

    assert run(source, io.StringIO(), spotter, 0.12, clock=lambda: 0.0) == 0
    assert len(spotter.stream.blocks) == 2


# ----------------------------------------------------------- the keyword file


class FakeEncoder:
    PIECES = {
        "HEY BOL": ["▁HE", "Y", "▁BO", "L"],
        "HEY BOWL": ["▁HE", "Y", "▁BO", "W", "L"],
        "HEY BALL": ["▁HE", "Y", "▁BA", "LL"],
        "HEY XYZZY": ["▁HE", "Y", "▁XYZZY"],
    }

    def load(self, path):
        self.loaded = path

    def encode(self, text, out_type=str):
        return self.PIECES.get(text, [])


@pytest.fixture
def bpe(monkeypatch, tmp_path):
    """A stand-in for sentencepiece plus the model's token table."""
    module = types.ModuleType("sentencepiece")
    module.SentencePieceProcessor = FakeEncoder
    monkeypatch.setitem(__import__("sys").modules, "sentencepiece", module)
    tokens = tmp_path / "tokens.txt"
    tokens.write_text(
        "\n".join(
            f"{piece} {index}"
            for index, piece in enumerate(
                ["<blk>", "▁HE", "Y", "▁BO", "L", "W", "▁BA", "LL"]
            )
        )
    )
    return tmp_path / "bpe.model", tokens


def test_the_keyword_file_carries_a_score_and_a_threshold_per_line(bpe):
    bpe_model, tokens = bpe

    text = keywords_text(["hey bol", "hey bowl"], 0.12, bpe_model, tokens)

    assert text == (
        "▁HE Y ▁BO L :2 #0.12\n"
        "▁HE Y ▁BO W L :2 #0.12\n"
    )


def test_duplicate_spellings_collapse_to_one_keyword(bpe):
    bpe_model, tokens = bpe

    text = keywords_text(["hey bol", "Hey Bol"], 0.2, bpe_model, tokens)

    assert text.count("\n") == 1


def test_a_phrase_the_model_has_no_tokens_for_is_refused(bpe):
    # Silently keeping an unencodable keyword would arm a listener that can
    # never fire, which looks exactly like a broken microphone.
    bpe_model, tokens = bpe

    with pytest.raises(WakeError) as exc:
        keywords_text(["hey xyzzy"], 0.12, bpe_model, tokens)

    assert "hey xyzzy" in str(exc.value)


# ----------------------------------------------------------------- spellings


def test_the_default_phrase_carries_the_spellings_a_decoder_produces():
    assert wake.spellings("hey bol") == ("hey bol", "hey bowl", "hey ball")


def test_an_unknown_phrase_is_used_exactly_as_written():
    assert wake.spellings("Computer Please") == ("computer please",)


def test_the_spellings_read_in_the_order_a_person_would_say_them():
    # This list is shown by `bol doctor`, so "hey bol" comes first.
    assert wake.all_spellings(["hey bol"]) == ["hey bol", "hey bowl", "hey ball"]


def test_a_longer_spelling_is_not_half_eaten_by_a_shorter_one():
    # "hey bol" matched first would leave a stray "l" on the front.
    assert wake.strip_wake_phrase("hey bowl run it", ["hey bol"]) == "run it"


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("hey bol, what files are in this folder", "what files are in this folder"),
        ("Hey Bol what files are in this folder", "what files are in this folder"),
        ("hey bowl. run the tests", "run the tests"),
        ("HEY BALL - run the tests", "run the tests"),
        ("Hey, Bol, run the tests", "run the tests"),
        ("  hey bol   run the tests", "run the tests"),
        ("hey bol", ""),
        # Not at the front, so it is dictation, not a wake.
        ("remind me to say hey bol", "remind me to say hey bol"),
        # A longer word that merely starts with the phrase.
        ("hey bollocks to that", "hey bollocks to that"),
        ("add a login test", "add a login test"),
        ("", ""),
    ],
)
def test_the_wake_phrase_comes_back_off_the_front(spoken, expected):
    assert wake.strip_wake_phrase(spoken, ["hey bol"]) == expected


def test_only_the_first_wake_phrase_is_stripped():
    text = "hey bol remind me to say hey bol"
    assert wake.strip_wake_phrase(text, ["hey bol"]) == "remind me to say hey bol"


def test_a_custom_phrase_is_stripped_too():
    assert wake.strip_wake_phrase("computer, ship it", ["computer"]) == "ship it"


@pytest.mark.parametrize(
    "line,expected",
    [
        ("wake 0.6 hey bol\n", (0.6, "hey bol")),
        ("wake 0.3 SEND IT\n", (0.3, "send it")),
        # No phrase on it is still a detection, and means the plain wake the
        # format used to be able to say and nothing else.
        ("wake 0.12\n", (0.12, "")),
        ("wake\n", None),
        ("ready\n", None),
        ("", None),
        ("wake later\n", None),
        ("2026-09-01 onnxruntime says hello\n", None),
        (None, None),
    ],
)
def test_only_a_real_wake_line_is_read_as_one(line, expected):
    assert wake.parse_wake_line(line) == expected


# ------------------------------------------------------------- the child pipe


class FakeStdin:
    def __init__(self, on_close=None) -> None:
        self.written = bytearray()
        self.closed = False
        self._on_close = on_close

    def write(self, data) -> None:
        if self.closed:
            raise BrokenPipeError("the wake child is gone")
        self.written += data

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True
        if self._on_close is not None:
            self._on_close()


class FakeReader:
    """Enough of asyncio.StreamReader for a test to script a child's stdout."""

    def __init__(self, lines=()) -> None:
        self._lines = list(lines)
        self._eof = False
        self._more = asyncio.Event()
        if self._lines:
            self._more.set()

    def push(self, line: bytes) -> None:
        self._lines.append(line)
        self._more.set()

    async def readline(self) -> bytes:
        while not self._lines:
            if self._eof:
                return b""
            self._more.clear()
            await self._more.wait()
        return self._lines.pop(0)


class FakeProc:
    def __init__(self, lines=(b"ready\n",)) -> None:
        self._exit = asyncio.Event()
        self.stdin = FakeStdin(on_close=self._exit.set)
        self.stdout = FakeReader(lines)
        self.stderr = FakeReader()
        self.killed = False

    async def wait(self) -> int:
        await self._exit.wait()
        return 0

    def kill(self) -> None:
        self.killed = True
        self._exit.set()


async def _listener(clock=None, lines=(b"ready\n",), on_wake=None, proc=None):
    cfg = Config().wake
    cfg.enabled = True
    proc = proc or FakeProc(lines)
    listener = WakeListener(
        cfg,
        on_wake or (lambda score, phrase: None),
        spawn=_spawner(proc),
        clock=clock or Clock(),
    )
    return listener, proc


def _spawner(proc):
    async def spawn():
        return proc

    return spawn


async def test_the_daemon_waits_for_the_child_to_say_ready():
    listener, proc = await _listener()
    assert await listener.start() is True
    assert listener.running is True
    await listener.stop()
    assert proc.stdin.closed


async def test_a_child_that_dies_before_ready_is_not_armed(caplog):
    proc = FakeProc()
    proc.stdout = FakeReader()
    proc.stdout._eof = True
    proc.stdout._more.set()
    listener, _ = await _listener(proc=proc)

    assert await listener.start() is False
    assert listener.running is False


async def test_a_wake_line_reaches_the_callback_with_the_phrase():
    # The phrase is not decoration: it is the only thing that says whether
    # this was "type", "send it" or "hey bol".
    heard = asyncio.Event()
    hits = []

    def on_wake(score, phrase):
        hits.append((score, phrase))
        heard.set()

    listener, proc = await _listener(on_wake=on_wake)
    assert await listener.start() is True
    proc.stdout.push(b"wake 0.6 type\n")

    await asyncio.wait_for(heard.wait(), timeout=1.0)

    assert hits == [(0.6, "type")]
    await listener.stop()


async def test_noise_on_the_child_pipe_is_ignored():
    hits = []
    listener, proc = await _listener(on_wake=lambda score, phrase: hits.append(score))
    assert await listener.start() is True
    proc.stdout.push(b"onnxruntime: using CPU\n")
    proc.stdout.push(b"wake\n")
    proc.stdout.push(b"wake 0.5 hey bowl\n")

    for _ in range(20):
        await asyncio.sleep(0)
        if hits:
            break

    assert hits == [0.5]
    await listener.stop()


async def test_blocks_reach_the_child_as_fixed_size_frames():
    listener, proc = await _listener()
    assert await listener.start() is True

    listener.feed(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    listener.feed(np.zeros(17, dtype=np.float32))  # a stream stopped mid-block
    for _ in range(20):
        await asyncio.sleep(0)
        if len(proc.stdin.written) >= 2 * FRAME_BYTES:
            break

    assert len(proc.stdin.written) == 2 * FRAME_BYTES
    await listener.stop()


async def test_a_muted_listener_forwards_nothing():
    listener, proc = await _listener()
    assert await listener.start() is True

    listener.mute()
    for _ in range(10):
        listener.feed(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    for _ in range(10):
        await asyncio.sleep(0)

    assert proc.stdin.written == b""
    await listener.stop()


async def test_a_wake_that_lands_while_muted_is_dropped():
    # The line was already in the pipe when Bol started speaking.
    scores = []
    listener, proc = await _listener(on_wake=scores.append)
    assert await listener.start() is True

    listener.mute()
    proc.stdout.push(b"wake 0.6 hey bol\n")
    for _ in range(20):
        await asyncio.sleep(0)

    assert scores == []
    await listener.stop()


async def test_the_child_falling_behind_costs_frames_not_the_microphone():
    listener, proc = await _listener()
    assert await listener.start() is True

    # The pump never runs, so the queue fills and then has to stop growing.
    for _ in range(wake.QUEUE_FRAMES * 4):
        listener.feed(np.zeros(FRAME_SAMPLES, dtype=np.float32))

    assert listener._frames.qsize() <= wake.QUEUE_FRAMES
    await listener.stop()


def test_mute_lasts_until_half_a_second_after_bol_stops_talking():
    clock = Clock()
    listener = WakeListener(Config().wake, lambda score: None, clock=clock)

    listener.mute()
    clock.advance(600.0)
    assert listener.muted is True  # however long the sentence runs

    listener.unmute()
    assert listener.muted is True
    clock.advance(wake.UNMUTE_DELAY_S - 0.01)
    assert listener.muted is True
    clock.advance(0.02)
    assert listener.muted is False


# ---------------------------------------------------------------- the daemon


class FakeWake:
    """The listener as the daemon sees it: mute, unmute, feed, stop."""

    def __init__(self, clock=None) -> None:
        self._clock = clock or Clock()
        self._muted = False
        self.fed = []
        self.stopped = False

    def mute(self) -> None:
        self._muted = True

    def unmute(self, delay=wake.UNMUTE_DELAY_S) -> None:
        self._muted = False

    @property
    def muted(self) -> bool:
        return self._muted

    def feed(self, block) -> None:
        self.fed.append(block)

    async def stop(self) -> None:
        self.stopped = True


class TickingRecorder(FakeRecorder):
    """A recorder that costs time, so an awake window can actually run out."""

    def __init__(self, count, clock, tick=8.0) -> None:
        super().__init__(count)
        self._clock = clock
        self._tick = tick
        self.muted = []

    async def record(self, session, until_silence):
        self._clock.advance(self._tick)
        return await super().record(session, until_silence)


class TriggerRecorder(TickingRecorder):
    """Keeps the sessions it was handed, so a test can read the timings the
    daemon put on them before the recording ran."""

    def __init__(self, count, clock, tick=8.0) -> None:
        super().__init__(count, clock, tick)
        self.sessions = []

    def begin(self):
        session = super().begin()
        self.sessions.append(session)
        return session


class BlockingRecorder(FakeRecorder):
    """Records forever, until somebody stops the session. For the two ways a
    recording ends that nobody in the room asked for."""

    def __init__(self) -> None:
        super().__init__(0)
        self.started = asyncio.Event()

    async def record(self, session, until_silence):
        self.calls.append(until_silence)
        self.started.set()
        for _ in range(5000):
            if session.stopped:
                return None
            await asyncio.sleep(0)
        return None


def _wake_daemon(
    utterances, texts, clock, awake_s=60.0, armed=True, commands=None
):
    cfg = Config()
    cfg.ui.sounds = False
    # No cleanup here: these tests are about trigger words, and the "always"
    # default would put a 195 MB model in front of every assertion.
    cfg.cleanup.mode = "off"
    cfg.cleanup.model = ""
    # Two-way, because several of these tests are about what Bol says and
    # when it is deaf while saying it.
    cfg.talk_back = True
    cfg.wake.enabled = True
    cfg.wake.awake_s = awake_s
    if commands:
        # Before the Daemon is built: the phrase table is read once, there.
        cfg.commands = commands
    d = Daemon(cfg, text_mode=False, clock=clock)
    d.recorder = TriggerRecorder(utterances, clock)
    d.transcriber = FakeTranscriber(texts)
    d.bridge = FakeBridge()
    d.speaker = FakeSpeaker()
    d.hud = FakeHud()
    d.wake = FakeWake(clock) if armed else None
    return d


async def test_a_wake_starts_a_recording_exactly_like_a_tap():
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=0.0)

    d._wake_detected(0.6)

    # The pill appears on the detection, before anything the wake starts.
    assert d.hud.calls[0] == ("listening", "Listening", "")
    await asyncio.sleep(0.05)
    # Ended by the silence gate, not by a key, and pasted rather than sent.
    assert d.recorder.calls == [True]
    assert d.bridge.injected == [("add a login test ", False)]


async def test_the_wake_phrase_never_reaches_claude():
    clock = Clock()
    d = _wake_daemon(1, ["hey bol, add a login test"], clock, awake_s=0.0)

    d._wake_detected(0.6)
    await asyncio.sleep(0.05)

    assert d.bridge.injected == [("add a login test ", False)]


async def test_a_wake_during_a_recording_is_ignored():
    clock = Clock()
    d = _wake_daemon(1, ["unused"], clock)

    async with d._listen_lock:
        d._wake_detected(0.6)

    assert d.hud.calls == []
    assert d.recorder.calls == []


async def test_a_wake_while_asleep_is_ignored():
    clock = Clock()
    d = _wake_daemon(1, ["unused"], clock)
    d._asleep = True

    d._wake_detected(0.6)
    await asyncio.sleep(0.05)

    assert d.recorder.calls == []


async def test_the_awake_window_keeps_the_mic_open_through_the_pauses():
    # 20 seconds awake, 8 seconds per silent listen window: the mic reopens
    # while the window is open and stops the moment it is not.
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=20.0)

    d._wake_detected(0.6)
    await asyncio.sleep(0.05)

    assert d.recorder.calls == [True, True, True, True]


async def test_words_heard_extend_the_awake_window():
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=20.0)

    d._wake_detected(0.6)
    await asyncio.sleep(0.05)

    # The wake was at t=1000 and the utterance ended at t=1008, so the window
    # runs from the last thing said, not from the wake that opened it.
    assert d._awake_until == pytest.approx(1028.0)


async def test_without_wake_mode_a_silent_window_ends_the_turn():
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=20.0, armed=False)

    await d._listen_session(d.recorder.begin(), until_silence=True)

    assert d.recorder.calls == [True]


async def test_a_tap_opens_the_awake_window_too():
    clock = Clock()
    d = _wake_daemon(0, [], clock, awake_s=20.0)

    assert d._awake() is False
    d._hotkey_pressed()
    assert d._awake() is True
    await asyncio.sleep(0.05)


async def test_a_tap_opens_nothing_when_wake_is_off():
    clock = Clock()
    d = _wake_daemon(0, [], clock, awake_s=20.0, armed=False)

    d._hotkey_pressed()

    assert d._awake() is False
    await asyncio.sleep(0.05)


async def test_going_to_sleep_closes_the_awake_window():
    clock = Clock()
    d = _wake_daemon(1, ["stop listening"], clock, awake_s=60.0)

    d._wake_detected(0.6)
    await asyncio.sleep(0.05)

    assert d._asleep is True
    assert d._awake() is False
    assert d.recorder.calls == [True]


async def test_the_pill_goes_away_the_moment_the_words_land():
    # The user feedback this exists for: "after pasting the content the pill
    # should close itself". No blue hint, no awake dot, nothing left over,
    # and that holds even with the awake window open behind it.
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=60.0)

    d._wake_detected(0.6)
    await asyncio.sleep(0.05)

    states = d.hud.states
    assert states[:3] == ["listening", "finalizing", "idle"]
    assert "sending" not in states
    assert "awake" not in states
    assert states[-1] == "idle"


async def test_bol_is_deaf_while_it_speaks_and_for_the_tail_after():
    clock = Clock()
    d = _wake_daemon(0, [], clock)
    listener = WakeListener(d.cfg.wake, lambda score: None, clock=clock)
    d.wake = listener
    heard = []

    class Speaker:
        async def speak(self, text):
            heard.append(listener.muted)

        async def stop(self):
            pass

    d.speaker = Speaker()

    await d._speak("Claude finished the job.")

    assert heard == [True]
    assert listener.muted is True
    clock.advance(wake.UNMUTE_DELAY_S + 0.01)
    assert listener.muted is False


async def test_bol_is_deaf_while_it_records():
    # Same microphone. A wake fired by the user's own dictation would land
    # the moment the recording ends, on top of whatever comes next.
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=0.0)
    muted = []

    class Watching(TickingRecorder):
        async def record(self, session, until_silence):
            muted.append(d.wake.muted)
            return await super().record(session, until_silence)

    d.recorder = Watching(1, clock)

    d._wake_detected(0.6)
    await asyncio.sleep(0.05)

    assert muted == [True]
    assert d.wake.muted is False  # the ear comes back when the recording ends


async def test_wake_is_not_started_in_text_mode():
    cfg = Config()
    cfg.wake.enabled = True
    d = Daemon(cfg, text_mode=True)

    await d._start_wake()

    assert d.wake is None


# ----------------------------------------------------------------- the config


def test_the_trigger_words_are_on_by_default():
    cfg = Config().wake
    assert cfg.enabled is True
    assert cfg.phrases == ["hey bol"]
    assert cfg.type_phrases == ["type"]
    assert cfg.send_phrases == ["send it", "send", "enter"]
    assert cfg.cancel_phrases == ["scratch that", "close"]
    assert cfg.sleep_phrases == ["stop listening"]
    assert cfg.threshold == 0.12
    assert cfg.type_threshold == 0.0  # 0 = use threshold
    assert cfg.pause_ms == 2000        # two seconds, then the paste is instant
    assert cfg.speak_window_ms == 5000  # how long the pill waits for speech
    # Off by default: with the window open, room noise kept reopening the
    # microphone and the pill kept coming back.
    assert cfg.awake_s == 0.0


def test_two_configs_do_not_share_a_phrase_list():
    first, second = Config(), Config()
    first.wake.phrases.append("computer")
    assert second.wake.phrases == ["hey bol"]


def test_the_default_file_says_what_the_dataclass_says():
    import tomllib

    from bol.config import DEFAULT_CONFIG_TOML

    data = tomllib.loads(DEFAULT_CONFIG_TOML)["wake"]
    cfg = Config().wake
    for key, value in data.items():
        assert getattr(cfg, key) == value, key


def test_the_default_file_is_honest_about_the_open_microphone():
    from bol.config import DEFAULT_CONFIG_TOML

    section = DEFAULT_CONFIG_TOML.split("[wake]", 1)[1]
    for promise in (
        "Wake mode keeps the microphone open",
        "Nothing is recorded or sent anywhere.",
        "false wake costs a Listening pill",
        "Bol itself needs no internet",
        # On by default now, so the way to close it has to be right there.
        "enabled = false to close the microphone",
    ):
        assert promise in section


def test_the_default_file_explains_the_awake_window_it_ships_off():
    # It ships at 0 because room noise kept reopening the mic. The file has
    # to say both what that means and how to get the minute back.
    from bol.config import DEFAULT_CONFIG_TOML

    section = DEFAULT_CONFIG_TOML.split("[wake]", 1)[1]
    assert "awake_s = 0" in section
    assert "only trigger words and the key ever start" in section
    assert "room noise cannot wake the pill" in section
    assert "60 = a free minute" in section


def test_the_type_threshold_is_documented_as_the_wrong_lever():
    # Measured against `say`: raising it stops the real "type add a login
    # test" firing before it stops the "type" inside "prototype".
    from bol.config import DEFAULT_CONFIG_TOML

    section = DEFAULT_CONFIG_TOML.split("[wake]", 1)[1]
    assert "change type_phrases instead" in section


def test_load_config_reads_the_wake_section(tmp_path):
    from bol.config import load_config

    path = tmp_path / "config.toml"
    path.write_text(
        "[wake]\nenabled = true\nphrases = [\"computer\"]\n"
        "type_phrases = [\"dictate\"]\nsend_phrases = [\"off you go\"]\n"
        "cancel_phrases = [\"forget it\"]\nsleep_phrases = [\"that's enough\"]\n"
        "threshold = 0.3\ntype_threshold = 0.4\npause_ms = 1500\nawake_s = 15\n"
        "speak_window_ms = 9000\n"
    )

    cfg = load_config(path)

    assert cfg.wake.enabled is True
    assert cfg.wake.phrases == ["computer"]
    assert cfg.wake.type_phrases == ["dictate"]
    assert cfg.wake.send_phrases == ["off you go"]
    assert cfg.wake.cancel_phrases == ["forget it"]
    assert cfg.wake.sleep_phrases == ["that's enough"]
    assert cfg.wake.threshold == 0.3
    assert cfg.wake.type_threshold == 0.4
    assert cfg.wake.pause_ms == 1500
    assert cfg.wake.awake_s == 15
    assert cfg.wake.speak_window_ms == 9000


def test_a_configured_trigger_word_reaches_the_daemon(tmp_path):
    # The whole path: a config file, a phrase table, and a daemon that knows
    # "dictate" means dictation.
    from bol.config import load_config

    path = tmp_path / "config.toml"
    path.write_text('[wake]\ntype_phrases = ["dictate"]\n')

    cfg = load_config(path)

    kinds = wake.keyword_map(cfg.wake, cfg.commands)
    assert kinds["dictate"] == wake.TYPE
    assert "type" not in kinds
    assert Daemon(cfg, text_mode=True)._wake_kinds["dictate"] == wake.TYPE


def test_wake_is_switched_off_in_one_line(tmp_path):
    from bol.config import load_config

    path = tmp_path / "config.toml"
    path.write_text("[wake]\nenabled = false\n")

    cfg = load_config(path)

    assert cfg.wake.enabled is False
    validate_config(cfg)  # and nothing else in the section is second-guessed


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("phrases", [], "at least one phrase"),
        ("phrases", ["  "], "at least one phrase"),
        ("threshold", 0.0, "above 0"),
        ("threshold", 1.5, "at most 1"),
        ("threshold", "loud", "must be a number"),
        ("awake_s", -1, "cannot be negative"),
        ("type_threshold", 1.5, "between 0 and 1"),
        ("type_threshold", -0.1, "between 0 and 1"),
        ("type_threshold", "loud", "must be a number"),
        ("pause_ms", 0, "must be above 0"),
        ("pause_ms", -100, "must be above 0"),
        ("pause_ms", "three", "must be a number"),
        ("speak_window_ms", 0, "must be above 0"),
        ("speak_window_ms", "five", "must be a number"),
    ],
)
def test_a_wake_section_that_cannot_work_is_refused(field, value, message):
    cfg = Config()
    cfg.wake.enabled = True
    setattr(cfg.wake, field, value)

    with pytest.raises(ValueError) as exc:
        validate_config(cfg)

    assert message in str(exc.value)


def test_a_disabled_wake_section_is_never_second_guessed():
    # A threshold someone left themselves a note about is not a reason to
    # refuse to start a daemon that will never read it.
    cfg = Config()
    cfg.wake.enabled = False
    cfg.wake.threshold = 99
    cfg.wake.phrases = []

    validate_config(cfg)


# ----------------------------------------------------------------- the doctor


def _install_model(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in wake_model.WANTED.values():
        (root / name).write_bytes(b"x" * 1000)


def test_doctor_stays_quiet_when_wake_is_off():
    cfg = Config()
    cfg.wake.enabled = False

    rows = cli.probe_wake(cfg)

    assert len(rows) == 1
    status, label, _hint = rows[0]
    assert status == cli.INFO
    assert "off" in label
    assert "hotkey still works" in label


def test_doctor_names_the_missing_extra(monkeypatch, tmp_path):
    monkeypatch.setattr(wake_model, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(wake, "wake_available", lambda: False)
    cfg = Config()
    cfg.wake.enabled = True

    rows = cli.probe_wake(cfg)

    bad = [row for row in rows if row[0] == cli.BAD]
    assert len(bad) == 2  # the package and the model it never downloaded
    # Wake is base now, so the fix is a plain reinstall, not a ",wake" extra.
    assert "bol[stt,llm]" in bad[0][2]


def test_doctor_reports_the_model_and_what_it_listens_for(monkeypatch, tmp_path):
    monkeypatch.setattr(wake_model, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(wake, "wake_available", lambda: True)
    _install_model(tmp_path / "models" / "kws")
    cfg = Config()
    cfg.wake.enabled = True

    rows = cli.probe_wake(cfg)

    assert [row[0] for row in rows] == [cli.OK, cli.OK, cli.INFO, cli.INFO]
    assert "on disk" in rows[1][1]
    # What to say, not every spelling the decoder might produce: "hey bowl"
    # is in the keyword file because the model emits it, not because anyone
    # should be told to say it.
    assert "listening for: hey bol, type, send it" in rows[2][1]
    assert "threshold 0.12" in rows[2][1]
    assert "2s pause pastes a dictation" in rows[2][1]
    assert rows[3][1] == cli.MIC_NOTE
    assert "microphone indicator stays on" in rows[3][1]


def test_setup_downloads_nothing_when_wake_is_off(capsys):
    cfg = Config()
    cfg.wake.enabled = False

    assert cli._setup_wake(cfg) is True

    out = capsys.readouterr().out
    assert "off" in out
    assert "hotkey works as before" in out


def test_setup_says_what_the_keyword_model_costs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(wake_model, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(wake, "wake_available", lambda: True)
    fetched = []

    def fake_download(root=None, **kwargs):
        fetched.append(root)
        _install_model(root)
        return root

    monkeypatch.setattr(wake, "download_model", fake_download)
    cfg = Config()
    cfg.wake.enabled = True

    assert cli._setup_wake(cfg) is True

    out = capsys.readouterr().out
    assert wake_model.MODEL_NAME in out
    assert "17.6 MB to download" in out
    assert fetched == [tmp_path / "models" / "kws"]
    # Setup says the same two things the doctor does.
    assert "listening for: hey bol, type, send it" in out
    assert cli.MIC_NOTE in out


# ------------------------------------------------------------------ the model


def test_only_the_five_files_bol_loads_come_out_of_the_tarball(tmp_path):
    # The release tarball also carries float32 copies, test wavs, and a
    # README: 19 MB where Bol needs 5.
    archive = tmp_path / "kws.tar.bz2"
    payload = tmp_path / "payload"
    payload.mkdir()
    names = list(wake_model.WANTED.values()) + [
        "encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        "README.md",
        "test_wavs/0.wav",
    ]
    with tarfile.open(archive, "w:bz2") as tar:
        for name in names:
            path = payload / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"y" * 10)
            tar.add(path, arcname=f"{wake_model.MODEL_NAME}/{name}")

    root = tmp_path / "kws"
    wake_model.download_model(root, fetch=lambda url, dest: dest.write_bytes(archive.read_bytes()))

    assert sorted(p.name for p in root.iterdir()) == sorted(wake_model.WANTED.values())
    assert wake_model.model_present(root) is True


def test_a_tarball_missing_a_file_is_an_error_not_a_broken_listener(tmp_path):
    archive = tmp_path / "kws.tar.bz2"
    payload = tmp_path / "tokens.txt"
    payload.write_bytes(b"y")
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(payload, arcname=f"{wake_model.MODEL_NAME}/tokens.txt")

    with pytest.raises(RuntimeError) as exc:
        wake_model.download_model(
            tmp_path / "kws",
            fetch=lambda url, dest: dest.write_bytes(archive.read_bytes()),
        )

    assert "bpe.model" in str(exc.value)


def test_the_model_url_points_at_the_english_keyword_release():
    assert wake_model.MODEL_URL.startswith(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    )
    assert wake_model.MODEL_URL.endswith(".tar.bz2")
    assert "gigaspeech" in wake_model.MODEL_NAME


# ------------------------------------------------------------------- startup


class DeadMicRecorder(FakeRecorder):
    async def monitor(self, listener):
        raise OSError("PortAudioError: device unavailable")


async def test_a_microphone_that_cannot_be_shared_still_leaves_a_hotkey(caplog):
    # Wake mode is a second way in. Losing it must never cost the first one.
    cfg = Config()
    cfg.wake.enabled = True
    d = Daemon(cfg, text_mode=False)
    d.transcriber = FakeTranscriber([])
    d.recorder = DeadMicRecorder(0)
    stopped = []

    class Listener:
        async def start(self):
            return True

        async def stop(self):
            stopped.append(True)

        def feed(self, block):
            pass

    import bol.daemon as daemon_mod

    original = daemon_mod.WakeListener
    daemon_mod.WakeListener = lambda *a, **k: Listener()
    try:
        await d._start_wake()
    finally:
        daemon_mod.WakeListener = original

    assert d.wake is None
    assert stopped == [True]


async def test_a_listener_that_will_not_start_is_simply_not_used():
    cfg = Config()
    cfg.wake.enabled = True
    d = Daemon(cfg, text_mode=False)
    d.transcriber = FakeTranscriber([])
    d.recorder = FakeRecorder(0)

    class Listener:
        async def start(self):
            return False

        async def stop(self):
            pass

    import bol.daemon as daemon_mod

    original = daemon_mod.WakeListener
    daemon_mod.WakeListener = lambda *a, **k: Listener()
    try:
        await d._start_wake()
    finally:
        daemon_mod.WakeListener = original

    assert d.wake is None
    assert d._awake() is False


# --------------------------------------------------------- the trigger table


@pytest.mark.parametrize(
    "phrase,kind",
    [
        ("hey bol", wake.WAKE),
        ("hey bowl", wake.WAKE),
        ("hey ball", wake.WAKE),
        ("type", wake.TYPE),
        ("send it", wake.SEND),
        ("send", wake.SEND),
        ("enter", wake.SEND),
        ("scratch that", wake.CANCEL),
        ("close", wake.CANCEL),
        ("stop listening", wake.SLEEP),
    ],
)
def test_every_trigger_word_maps_to_one_kind(phrase, kind):
    cfg = Config()

    assert wake.keyword_map(cfg.wake, cfg.commands)[phrase] == kind


def test_the_doctor_line_names_one_phrase_per_kind():
    cfg = Config()

    # Not "hey bowl": that is in the keyword file because the decoder emits
    # it, not because anyone should be told to say it.
    assert wake.lead_phrases(cfg.wake, cfg.commands) == [
        "hey bol", "type", "send it", "scratch that", "stop listening",
    ]


def test_a_remapped_send_command_becomes_the_trigger_word():
    # One name for sending. A grammar that says "ship it" and a keyword model
    # still listening for "send it" is two answers to the same question.
    cfg = Config()
    cfg.commands = {"send": ["ship it", "fire away"]}

    kinds = wake.keyword_map(cfg.wake, cfg.commands)

    assert kinds["ship it"] == wake.SEND
    assert kinds["fire away"] == wake.SEND
    assert "send it" not in kinds


def test_an_edited_wake_list_is_never_overruled_by_commands():
    cfg = Config()
    cfg.commands = {"send": ["ship it"]}
    cfg.wake.send_phrases = ["fire it off"]

    kinds = wake.keyword_map(cfg.wake, cfg.commands)

    assert kinds["fire it off"] == wake.SEND
    assert "ship it" not in kinds


def test_a_spelling_two_kinds_claim_belongs_to_the_first():
    # The child reports a spelling and nothing else, so a spelling that meant
    # two things would be a coin toss every time it fired.
    cfg = Config()
    cfg.wake.cancel_phrases = ["type"]

    assert wake.keyword_map(cfg.wake, cfg.commands)["type"] == wake.TYPE


def test_blank_and_repeated_phrases_never_reach_the_keyword_file():
    cfg = Config()
    cfg.wake.type_phrases = ["  Type ", "type", ""]

    args = wake.keyword_args(cfg.wake, cfg.commands)

    assert args.count("type") == 1


def test_only_type_may_carry_its_own_threshold():
    cfg = Config()
    assert "type" in wake.keyword_args(cfg.wake, cfg.commands)

    cfg.wake.type_threshold = 0.3
    args = wake.keyword_args(cfg.wake, cfg.commands)

    assert "type=0.3" in args
    assert "type" not in args
    assert "send it" in args  # everything else stays on the shared threshold


@pytest.mark.parametrize(
    "value,expected",
    [
        ("type", ("type", None)),
        ("type=0.3", ("type", 0.3)),
        ("send it", ("send it", None)),
        ("SEND IT", ("send it", None)),
        # Not a number, so it was never a threshold suffix.
        ("send=it", ("send=it", None)),
    ],
)
def test_a_phrase_argument_may_carry_its_own_threshold(value, expected):
    assert parse_phrase_arg(value) == expected


def test_a_keyword_with_its_own_threshold_does_not_move_the_others(bpe):
    bpe_model, tokens = bpe

    text = keywords_text([("hey bol", 0.4), "hey bowl"], 0.12, bpe_model, tokens)

    assert text == (
        "▁HE Y ▁BO L :2 #0.4\n"
        "▁HE Y ▁BO W L :2 #0.12\n"
    )


# ----------------------------------------------------------- the type flow


def test_a_type_dictation_strips_only_the_leading_trigger_word():
    phrases = ["hey bol", "type"]
    assert wake.strip_wake_phrase("type the prototype loader", phrases) == (
        "the prototype loader"
    )
    assert wake.strip_wake_phrase("type what type of file", phrases) == (
        "what type of file"
    )
    # The word is only ever taken off the front, and only once.
    assert wake.strip_wake_phrase("the prototype loader", phrases) == (
        "the prototype loader"
    )


async def test_a_type_dictation_leaves_the_trigger_word_behind():
    clock = Clock()
    d = _wake_daemon(1, ["type add a login test"], clock, awake_s=0.0)

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    # The trailing space is the tell: this went through as plain dictation
    # with the trigger word already gone, not through the grammar's own
    # "type ..." prefix, which pastes the payload with nothing added.
    assert d.bridge.injected == [("add a login test ", False)]


async def test_a_type_dictation_ends_on_the_wake_pause_not_the_audio_one():
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=0.0)
    d.cfg.audio.silence_ms = 900
    d.cfg.wake.pause_ms = 3000

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    session = d.recorder.sessions[0]
    assert session.silence_ms == 3000
    # And it gives the microphone back after speak_window_ms if nobody speaks:
    # the two ends of the recording are two settings now, because "how long a
    # pause lasts" and "how long to wait for a first word" are two questions.
    assert session.window_ms == d.cfg.wake.speak_window_ms == 5000


async def test_a_wake_keeps_the_audio_pause_and_takes_the_short_window():
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=0.0)

    d._wake_detected(0.6, "hey bol")
    await asyncio.sleep(0.05)

    session = d.recorder.sessions[0]
    assert session.silence_ms is None  # a conversation pauses like a conversation
    assert session.window_ms == 5000   # but still gives up in seconds


async def test_a_hotkey_recording_keeps_every_configured_timing():
    # The hand on the key is already the statement that something is coming.
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=0.0)

    d._hotkey_pressed()
    await asyncio.sleep(0.05)

    session = d.recorder.sessions[0]
    assert session.silence_ms is None
    assert session.window_ms is None


async def test_a_type_dictation_never_presses_enter():
    # "type" means put these characters there, and nothing else does either:
    # eight words finished on a pause are still only pasted.
    clock = Clock()
    d = _wake_daemon(
        1, ["add a login test to the auth module"], clock, awake_s=0.0
    )
    d.recorder.end_reason = "silence"

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    assert d.bridge.injected == [("add a login test to the auth module ", False)]
    assert d.bridge.keys == []
    assert d._pending_paste is True


async def test_a_type_dictation_takes_its_own_pill_down():
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=0.0)

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    assert d.hud.states[-1] == "idle"
    assert "sending" not in d.hud.states
    assert d._pending_paste is True  # still pending; the pill just says nothing


async def test_a_type_dictation_waits_speak_window_ms_for_speech():
    # The trigger word decides both ends of the recording: speak_window_ms to
    # start talking, pause_ms of silence to end it.
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=0.0)
    d.cfg.wake.speak_window_ms = 4000
    d.cfg.wake.pause_ms = 2000

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    session = d.recorder.sessions[0]
    assert session.window_ms == 4000
    assert session.silence_ms == 2000


async def test_a_trigger_that_hears_nothing_hides_the_pill_again():
    # Five seconds of nothing said is a trigger word heard across the room.
    # It costs one dark capsule and then the screen is clear again.
    clock = Clock()
    d = _wake_daemon(0, [], clock, awake_s=0.0)

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    assert d.recorder.sessions[0].window_ms == d.cfg.wake.speak_window_ms
    assert d.hud.states == ["listening", "idle"]
    assert d.bridge.injected == []


async def test_an_explicit_send_it_still_sends_a_type_dictation():
    # The trigger word decides the default, never the user's own last words.
    clock = Clock()
    d = _wake_daemon(1, ["add a login test send it"], clock, awake_s=0.0)

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    assert d.bridge.injected == [("add a login test", True)]
    assert d._pending_paste is False


async def test_a_type_dictation_stays_awake_for_the_next_sentence():
    clock = Clock()
    d = _wake_daemon(
        2, ["add a login test", "and a logout test"], clock, awake_s=20.0
    )

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    # The second sentence needed no trigger word, and was typed like the first.
    assert d.bridge.injected == [
        ("add a login test ", False),
        ("and a logout test ", False),
    ]
    assert all(s.silence_ms == 2000 for s in d.recorder.sessions)


async def test_awake_s_zero_means_only_trigger_words_start_anything():
    clock = Clock()
    d = _wake_daemon(2, ["add a login test", "never heard"], clock, awake_s=0.0)

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    assert d.recorder.calls == [True]
    assert d.bridge.injected == [("add a login test ", False)]
    assert d._awake() is False


async def test_a_phrase_the_table_never_heard_of_is_a_plain_wake():
    # A child from another version, or a keyword file nobody updated: the
    # safest reading of an unknown trigger word is the one that only opens
    # the microphone.
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=0.0)

    d._wake_detected(0.6, "something else entirely")
    await asyncio.sleep(0.05)

    assert d.recorder.calls == [True]
    assert d.recorder.sessions[0].silence_ms is None


# ------------------------------------------------------- send, cancel, pause


async def _typed(clock, text="add a login test", **kwargs):
    """A daemon that has just pasted something and not sent it."""
    d = _wake_daemon(1, [text], clock, awake_s=0.0, **kwargs)
    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)
    assert d._pending_paste is True
    return d


async def test_a_send_trigger_presses_enter_on_a_pending_paste():
    clock = Clock()
    d = await _typed(clock)

    d._wake_detected(0.6, "send it")
    await asyncio.sleep(0.05)

    assert d.bridge.keys == [("Enter",)]
    assert d._pending_paste is False
    assert d.recorder.calls == [True]  # it started no recording of its own
    # And nothing on screen: the text left the box where the user is looking,
    # and the chime is the whole receipt.
    assert d.hud.calls[-1] == ("idle", "", "")


async def test_a_send_trigger_with_nothing_pasted_is_ignored():
    # "Send it" on the television must never press Enter on something the
    # user typed by hand.
    clock = Clock()
    d = _wake_daemon(0, [], clock)

    d._wake_detected(0.6, "send it")
    await asyncio.sleep(0.05)

    assert d.bridge.keys == []
    assert d.bridge.injected == []
    assert d.recorder.calls == []
    assert d.hud.calls == []


async def test_a_send_trigger_into_the_wrong_window_keeps_the_paste_pending():
    class BlockedBridge(FakeBridge):
        async def inject_keys(self, *keys, explicit=False):
            if "Enter" in keys:
                raise SubmitBlocked("Notes isn't Claude", "Notes isn't Claude")
            await super().inject_keys(*keys, explicit=explicit)

    clock = Clock()
    d = await _typed(clock)
    d.bridge = BlockedBridge()

    d._wake_detected(0.6, "send it")
    await asyncio.sleep(0.05)

    # The Enter never landed, so the paste is still pending and saying it
    # again with Claude in front will finish the job.
    assert d._pending_paste is True
    assert d.bridge.keys == []
    assert "didn't press Enter" in d.speaker.spoken[-1]


async def test_a_cancel_trigger_wipes_a_pending_paste():
    clock = Clock()
    d = await _typed(clock)

    d._wake_detected(0.6, "scratch that")
    await asyncio.sleep(0.05)

    assert d.bridge.keys == [("C-u",)]
    assert d._pending_paste is False
    assert d.recorder.calls == [True]


async def test_a_cancel_trigger_with_nothing_pasted_is_ignored():
    clock = Clock()
    d = _wake_daemon(0, [], clock)

    d._wake_detected(0.6, "scratch that")
    await asyncio.sleep(0.05)

    assert d.bridge.keys == []


async def test_a_send_after_a_hotkey_paste_works_too():
    # The flag is about what is in the box, not about how it got there.
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=0.0)

    d._hotkey_pressed()
    await asyncio.sleep(0.05)
    assert d._pending_paste is True

    d._wake_detected(0.6, "send it")
    await asyncio.sleep(0.05)

    assert d.bridge.keys == [("Enter",)]


async def test_a_sleep_trigger_pauses_the_ear_as_well_as_the_loop():
    clock = Clock()
    d = _wake_daemon(0, [], clock)

    d._wake_detected(0.6, "stop listening")

    assert d._asleep is True
    assert d.wake.muted is True  # no trigger words, and no core burnt on them
    assert d._awake() is False
    assert ("sending", daemon_mod.SLEEP_HINT, "") in d.hud.calls
    assert d.hud.holds[-1] == daemon_mod.HINT_S


async def test_a_paused_bol_ignores_every_trigger_word():
    clock = Clock()
    d = _wake_daemon(1, ["never heard"], clock)
    d._wake_detected(0.6, "stop listening")
    d._pending_paste = True

    for phrase in ("type", "hey bol", "send it", "scratch that"):
        d._wake_detected(0.6, phrase)
    await asyncio.sleep(0.05)

    assert d.recorder.calls == []
    assert d.bridge.keys == []
    assert d.bridge.injected == []


async def test_the_spoken_sleep_command_pauses_the_ear_too(capsys):
    # The regression this exists for: _capture_and_handle unmutes the wake
    # listener in a finally, and that must not undo a pause set inside it.
    clock = Clock()
    d = _wake_daemon(1, ["stop listening"], clock, awake_s=60.0)

    d._wake_detected(0.6, "hey bol")
    await asyncio.sleep(0.05)

    assert d._asleep is True
    assert d.wake.muted is True
    assert d._awake() is False
    assert "paused. Press the key to resume" in capsys.readouterr().out


async def test_the_hotkey_is_the_way_back_from_a_pause(capsys):
    clock = Clock()
    d = _wake_daemon(0, [], clock)
    d._wake_detected(0.6, "stop listening")
    capsys.readouterr()

    d._hotkey_pressed()
    await asyncio.sleep(0.05)

    assert d._asleep is False
    assert d.wake.muted is False
    assert "bol: listening again." in capsys.readouterr().out


async def test_the_resume_line_is_printed_once(capsys):
    clock = Clock()
    d = _wake_daemon(0, [], clock)
    d._wake_detected(0.6, "stop listening")
    capsys.readouterr()

    d._hotkey_pressed()
    d._hotkey_pressed()
    await asyncio.sleep(0.05)

    assert capsys.readouterr().out.count("listening again") == 1


# ------------------------------------------------------------- cancelling


async def test_a_click_cancels_a_recording_a_trigger_word_started():
    clock = Clock()
    d = _wake_daemon(0, [], clock, awake_s=60.0)
    d.recorder = BlockingRecorder()

    d._wake_detected(0.6, "type")
    await asyncio.wait_for(d.recorder.started.wait(), timeout=1.0)
    session = d._active_session
    d._clicked()
    await asyncio.sleep(0.05)

    assert session.end_reason == "cancelled"
    assert d.bridge.injected == []
    assert d._pending_paste is False
    # The window that would have reopened the microphone is shut with it.
    assert d._awake() is False
    assert d.hud.states[-1] == "idle"


async def test_a_click_leaves_a_hotkey_recording_alone():
    # Clicking to put the cursor somewhere while dictating is a thing people
    # do on purpose, and the key in their other hand is the way out.
    clock = Clock()
    d = _wake_daemon(0, [], clock)
    d.recorder = BlockingRecorder()

    d._hotkey_pressed()
    await asyncio.wait_for(d.recorder.started.wait(), timeout=1.0)
    session = d._active_session
    d._clicked()
    await asyncio.sleep(0.02)

    assert session.stopped is False
    assert d._wake_session is None
    session.request_stop()
    await asyncio.sleep(0.05)


async def test_a_click_with_no_recording_running_does_nothing():
    clock = Clock()
    d = _wake_daemon(0, [], clock)

    d._clicked()

    assert d.recorder.calls == []


async def test_another_app_coming_forward_cancels_a_trigger_recording(monkeypatch):
    monkeypatch.setattr(daemon_mod, "FRONTMOST_POLL_S", 0.001)
    seen = ["com.apple.Terminal", "com.apple.Terminal", "com.google.Chrome"]

    async def frontmost():
        return seen.pop(0) if len(seen) > 1 else seen[0]

    monkeypatch.setattr(daemon_mod, "frontmost_bundle_id", frontmost)
    clock = Clock()
    d = _wake_daemon(0, [], clock, awake_s=60.0)
    d.recorder = BlockingRecorder()

    d._wake_detected(0.6, "type")
    await asyncio.wait_for(d.recorder.started.wait(), timeout=1.0)
    session = d._active_session
    for _ in range(200):
        await asyncio.sleep(0.005)
        if session.stopped:
            break

    assert session.end_reason == "cancelled"
    assert d.bridge.injected == []
    assert d._awake() is False


async def test_an_unreadable_frontmost_app_never_cancels_anything(monkeypatch):
    # A missing Automation permission reads as "", and a permission Bol does
    # not have must not cancel every recording it ever starts.
    monkeypatch.setattr(daemon_mod, "FRONTMOST_POLL_S", 0.001)

    async def frontmost():
        return ""

    monkeypatch.setattr(daemon_mod, "frontmost_bundle_id", frontmost)
    clock = Clock()
    d = _wake_daemon(0, [], clock)
    d.recorder = BlockingRecorder()

    d._wake_detected(0.6, "type")
    await asyncio.wait_for(d.recorder.started.wait(), timeout=1.0)
    session = d._active_session
    await asyncio.sleep(0.05)

    assert session.stopped is False
    session.request_stop()
    await asyncio.sleep(0.05)


async def test_a_frontmost_watcher_that_raises_never_ends_a_recording(monkeypatch):
    monkeypatch.setattr(daemon_mod, "FRONTMOST_POLL_S", 0.001)

    async def frontmost():
        raise OSError("osascript is not available")

    monkeypatch.setattr(daemon_mod, "frontmost_bundle_id", frontmost)
    clock = Clock()
    d = _wake_daemon(0, [], clock)
    d.recorder = BlockingRecorder()

    d._wake_detected(0.6, "type")
    await asyncio.wait_for(d.recorder.started.wait(), timeout=1.0)
    session = d._active_session
    await asyncio.sleep(0.05)

    assert session.stopped is False
    session.request_stop()
    await asyncio.sleep(0.05)


# ------------------------------------------------------ a model nobody has


async def test_a_missing_keyword_model_costs_one_info_line(monkeypatch, tmp_path, caplog):
    # Wake is on by default, so the first `bol run` on a machine nobody has
    # set up must not open with a warning about a feature they never asked
    # for. One line, and the hotkey.
    monkeypatch.setattr(wake, "wake_available", lambda: True)
    listener = WakeListener(Config().wake, lambda s, p: None, model_root=tmp_path)

    with caplog.at_level(logging.DEBUG, logger="bol.wake"):
        assert await listener.start() is False

    assert "keyword model not downloaded" in caplog.text
    assert "run `bol setup`" in caplog.text
    assert "hotkey still works" in caplog.text
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_a_daemon_with_no_keyword_model_still_arms_the_hotkey(monkeypatch, tmp_path):
    monkeypatch.setattr(wake_model, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(wake, "wake_available", lambda: True)
    d = Daemon(Config(), text_mode=False)
    d.transcriber = FakeTranscriber([])
    d.recorder = FakeRecorder(0)

    await d._start_wake()

    assert d.wake is None
    assert d._awake() is False
