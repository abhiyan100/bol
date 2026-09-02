"""Wake phrase tests: the child protocol, the daemon's use of it, and the two
rules ("the phrase comes back off the transcript", "Bol is deaf while it
speaks") that decide whether wake mode is usable or merely present.

Nothing here imports sherpa-onnx or sentencepiece. CI installs base
dependencies only, and the whole point of putting the keyword model in a
child process is that the daemon's own tests never have to load it: the
spotter, the BPE encoder, and the child process are all fakes.
"""

import asyncio
import io
import tarfile
import types

import numpy as np
import pytest

import bol.cli as cli
import bol.wake as wake
from bol.config import Config, validate_config
from bol.daemon import Daemon
from bol.wake import WakeListener, model as wake_model
from bol.wake.listener import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    RESET_GAP_S,
    WakeError,
    keywords_text,
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
        ("wake 0.6 hey bol\n", 0.6),
        ("wake 0.12\n", 0.12),
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
        on_wake or (lambda score: None),
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


async def test_a_wake_line_reaches_the_callback():
    heard = asyncio.Event()
    scores = []

    def on_wake(score):
        scores.append(score)
        heard.set()

    listener, proc = await _listener(on_wake=on_wake)
    assert await listener.start() is True
    proc.stdout.push(b"wake 0.6 hey bol\n")

    await asyncio.wait_for(heard.wait(), timeout=1.0)

    assert scores == [0.6]
    await listener.stop()


async def test_noise_on_the_child_pipe_is_ignored():
    scores = []
    listener, proc = await _listener(on_wake=scores.append)
    assert await listener.start() is True
    proc.stdout.push(b"onnxruntime: using CPU\n")
    proc.stdout.push(b"wake\n")
    proc.stdout.push(b"wake 0.5 hey bowl\n")

    for _ in range(20):
        await asyncio.sleep(0)
        if scores:
            break

    assert scores == [0.5]
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


def _wake_daemon(utterances, texts, clock, awake_s=60.0, armed=True):
    cfg = Config()
    cfg.sound_cues = False
    cfg.hands_free = False
    cfg.hotkey.submit = "voice"
    cfg.wake.enabled = True
    cfg.wake.awake_s = awake_s
    d = Daemon(cfg, text_mode=False, clock=clock)
    d.recorder = TickingRecorder(utterances, clock)
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
    # Ended by the silence gate, not by a key, and pasted under submit=voice.
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
    assert d.cfg.hands_free is False  # nothing to do with hands-free


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


async def test_the_pill_keeps_a_dot_up_while_the_window_is_open():
    clock = Clock()
    d = _wake_daemon(1, ["add a login test"], clock, awake_s=60.0)

    d._wake_detected(0.6)
    await asyncio.sleep(0.05)

    # Pasted, so the pill has nothing left to say, but the window is still
    # open and that is worth a dot rather than a blank screen. The blank
    # screen comes back exactly once, when the window finally closes.
    states = d.hud.states
    assert states[:3] == ["listening", "finalizing", "awake"]
    assert states.count("idle") == 1
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


def test_wake_is_off_by_default():
    cfg = Config().wake
    assert cfg.enabled is False
    assert cfg.phrases == ["hey bol"]
    assert cfg.threshold == 0.12
    assert cfg.awake_s == 60.0


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
        "Turn your wifi off and try it",
    ):
        assert promise in section


def test_load_config_reads_the_wake_section(tmp_path):
    from bol.config import load_config

    path = tmp_path / "config.toml"
    path.write_text(
        "[wake]\nenabled = true\nphrases = [\"computer\"]\n"
        "threshold = 0.3\nawake_s = 15\n"
    )

    cfg = load_config(path)

    assert cfg.wake.enabled is True
    assert cfg.wake.phrases == ["computer"]
    assert cfg.wake.threshold == 0.3
    assert cfg.wake.awake_s == 15


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("phrases", [], "at least one phrase"),
        ("phrases", ["  "], "at least one phrase"),
        ("threshold", 0.0, "above 0"),
        ("threshold", 1.5, "at most 1"),
        ("threshold", "loud", "must be a number"),
        ("awake_s", -1, "cannot be negative"),
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
    cfg.wake.threshold = 99
    cfg.wake.phrases = []

    validate_config(cfg)


# ----------------------------------------------------------------- the doctor


def _install_model(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in wake_model.WANTED.values():
        (root / name).write_bytes(b"x" * 1000)


def test_doctor_stays_quiet_when_wake_is_off():
    rows = cli.probe_wake(Config())

    assert len(rows) == 1
    status, label, _hint = rows[0]
    assert status == cli.INFO
    assert "off" in label


def test_doctor_names_the_missing_extra(monkeypatch, tmp_path):
    monkeypatch.setattr(wake_model, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(wake, "wake_available", lambda: False)
    cfg = Config()
    cfg.wake.enabled = True

    rows = cli.probe_wake(cfg)

    bad = [row for row in rows if row[0] == cli.BAD]
    assert len(bad) == 2  # the extra and the model it never downloaded
    assert "bol[stt,llm,wake]" in bad[0][2]


def test_doctor_reports_the_model_and_what_it_listens_for(monkeypatch, tmp_path):
    monkeypatch.setattr(wake_model, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(wake, "wake_available", lambda: True)
    _install_model(tmp_path / "models" / "kws")
    cfg = Config()
    cfg.wake.enabled = True

    rows = cli.probe_wake(cfg)

    assert [row[0] for row in rows] == [cli.OK, cli.OK, cli.INFO]
    assert "on disk" in rows[1][1]
    assert "hey bol, hey bowl, hey ball" in rows[2][1]
    assert "threshold 0.12" in rows[2][1]


def test_setup_downloads_nothing_when_wake_is_off(capsys):
    assert cli._setup_wake(Config()) is True
    assert "off" in capsys.readouterr().out


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
