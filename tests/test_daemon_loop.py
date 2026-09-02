"""Regression tests for the listening loop: the races and dead paths the code
reviews confirmed (session-token stops, hands-free chaining, multi-session
hook affinity, mic failures)."""

import asyncio
import logging
import queue as _queue

import numpy as np
import pytest

from bol.audio.capture import RecordingSession
from bol import daemon as daemon_mod
from bol.config import Config
from bol.daemon import Daemon


class FakeRecorder:
    """Scripted recorder: returns queued 'audio' markers, no hardware."""

    def __init__(self, utterance_count):
        self.remaining = utterance_count
        self.calls = []

    def begin(self):
        return RecordingSession()

    async def record(self, session, until_silence):
        self.calls.append(until_silence)
        if session.stopped or self.remaining == 0:
            return None
        self.remaining -= 1
        return np.zeros(16000, dtype=np.float32)


class FakeTranscriber:
    def __init__(self, texts):
        self.texts = list(texts)

    async def warmup(self):
        pass

    async def transcribe(self, audio, sample_rate):
        return self.texts.pop(0) if self.texts else ""


class FakeBridge:
    def __init__(self):
        self.injected = []
        self.keys = []

    async def inject(self, text, submit):
        self.injected.append((text, submit))

    async def inject_keys(self, *keys):
        self.keys.append(keys)

    async def interrupt(self):
        self.keys.append(("Escape",))


class FakeSpeaker:
    def __init__(self):
        self.spoken = []

    async def speak(self, text):
        self.spoken.append(text)

    async def stop(self):
        pass


class FakeSummarizer:
    async def summarize(self, event):
        return f"summary of {event.session_id}"


class FakeHud:
    """The pill, without a window: it just records what it was told to show."""

    def __init__(self, order=None):
        self.calls = []
        self.order = order

    async def start(self):
        pass

    async def stop(self):
        pass

    def set(self, state, text="", detail=""):
        self.calls.append((state, text, detail))
        if self.order is not None:
            self.order.append(f"pill:{state}")

    @property
    def states(self):
        return [state for state, _text, _detail in self.calls]


def _daemon(utterances, texts, submit="voice"):
    cfg = Config()
    cfg.sound_cues = False
    cfg.hands_free = True
    # These loop tests predate auto-send and are about mic ownership, so they
    # keep the explicit "send it" grammar. Auto-send has its own tests below.
    cfg.hotkey.submit = submit
    d = Daemon(cfg, text_mode=False)
    d.recorder = FakeRecorder(utterances)
    d.transcriber = FakeTranscriber(texts)
    d.bridge = FakeBridge()
    d.speaker = FakeSpeaker()
    d.hud = FakeHud()
    return d


@pytest.mark.asyncio
async def test_hands_free_chains_dictation_then_send():
    d = _daemon(2, ["add a login test", "send it"])
    await d._listen_session(d.recorder.begin(), until_silence=False)
    # Both utterances were captured in ONE mic ownership (the reopen loop),
    # and the second one submitted.
    assert d.recorder.calls == [False, True]
    assert d.bridge.injected == [("add a login test ", False), ("", True)]


@pytest.mark.asyncio
async def test_send_closes_the_mic():
    d = _daemon(5, ["ship it send", "should never be heard"])
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.recorder.calls == [False]
    assert d.bridge.injected == [("ship it", True)]


@pytest.mark.asyncio
async def test_hands_free_off_never_chains():
    d = _daemon(3, ["hello there", "more words"])
    d.cfg.hands_free = False
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.recorder.calls == [False]


@pytest.mark.asyncio
async def test_sleep_stops_chaining():
    d = _daemon(3, ["stop listening", "unheard"])
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.recorder.calls == [False]
    assert d._asleep is True


@pytest.mark.asyncio
async def test_release_before_record_stops_only_its_session():
    d = _daemon(1, ["unused"])
    session = d.recorder.begin()
    session.request_stop()  # released before the mic opened
    await d._listen_session(session, until_silence=False)
    # The stopped session yields no audio and nothing is injected...
    assert d.bridge.injected == []
    # ...and a FRESH session is unaffected by the stale stop.
    assert not d.recorder.begin().stopped


@pytest.mark.asyncio
async def test_stale_hotkey_release_is_inert():
    d = _daemon(1, ["unused"])
    d._ptt_session = None
    d._hotkey_released()  # no session in flight, must not raise or affect anything


@pytest.mark.asyncio
async def test_auto_listen_noop_while_mic_owned():
    d = _daemon(1, ["hello"])
    async with d._listen_lock:
        await d._auto_listen()  # lock held elsewhere → must return, not deadlock
    assert d.recorder.calls == []


@pytest.mark.asyncio
async def test_permission_yes_presses_enter_and_yields_to_claude():
    d = _daemon(1, ["yes"])
    d._permission_session = "s1"
    reopen = await d._handle_utterance("yes")
    assert reopen is False
    assert ("Enter",) in d.bridge.keys


# ------------------------------------------------------------------ A1 sessions


def _stop(session_id, message="Claude finished the job it was given.", cwd="/w/repo"):
    return {
        "hook_event_name": "Stop",
        "session_id": session_id,
        "prompt_id": f"p-{session_id}",
        "last_assistant_message": message,
        "cwd": cwd,
    }


def _hook_daemon():
    d = _daemon(0, [])
    d.cfg.hands_free = False  # keep the mic out of the hook tests
    d.summarizer = FakeSummarizer()
    return d


@pytest.mark.asyncio
async def test_only_the_first_session_is_narrated():
    d = _hook_daemon()
    await d._on_stop(_stop("A", cwd="/work/alpha"))
    await d._on_stop(_stop("B", cwd="/work/beta"))
    await d._on_stop(_stop("C", cwd="/work/gamma"))
    assert d.speaker.spoken == ["summary of A"]
    assert d._bound_session == "A"


@pytest.mark.asyncio
async def test_follow_all_narrates_every_session():
    d = _hook_daemon()
    d.cfg.server.follow = "all"
    await d._on_stop(_stop("A"))
    await d._on_stop(_stop("B"))
    await d._on_stop(_stop("C"))
    assert d.speaker.spoken == ["summary of A", "summary of B", "summary of C"]


@pytest.mark.asyncio
async def test_permission_from_a_foreign_session_never_presses_enter():
    d = _hook_daemon()
    await d._on_stop(_stop("A"))  # Bol latches onto session A
    await d._on_notification(
        {
            "hook_event_name": "Notification",
            "session_id": "B",
            "notification_type": "permission_prompt",
            "message": "Claude wants to run rm -rf.",
        }
    )
    # Session B's prompt was never announced, so nothing is armed...
    assert d._permission_session is None
    reopen = await d._handle_utterance("yes")
    # ...and "yes" is plain dictation, not an approval keystroke.
    assert ("Enter",) not in d.bridge.keys
    assert reopen is True


@pytest.mark.asyncio
async def test_permission_answer_checks_the_prompting_session():
    d = _hook_daemon()
    await d._on_stop(_stop("A"))
    d._permission_session = "B"  # e.g. armed under follow="all", then rebound
    reopen = await d._handle_utterance("yes")
    assert ("Enter",) not in d.bridge.keys
    assert reopen is True
    assert d._permission_session is None


@pytest.mark.asyncio
async def test_ignored_session_still_drains_its_tool_log():
    d = _hook_daemon()
    await d._on_stop(_stop("A"))
    d.tracker.record_tool(
        {"prompt_id": "p-B", "tool_name": "Edit",
         "tool_input": {"file_path": "/x.py"}, "tool_response": {}}
    )
    await d._on_stop(_stop("B"))
    assert d.tracker._tools == {}


# ------------------------------------------------------------------ A3 failures


class ExplodingRecorder(FakeRecorder):
    """Fails the first record() the way a yanked input device does."""

    def __init__(self, utterance_count):
        super().__init__(utterance_count)
        self.exploded = False

    async def record(self, session, until_silence):
        if not self.exploded:
            self.exploded = True
            self.calls.append(until_silence)
            raise OSError("PortAudioError: device unavailable")
        return await super().record(session, until_silence)


@pytest.mark.asyncio
async def test_mic_failure_is_spoken_and_the_hotkey_survives():
    d = _daemon(1, ["add a login test"])
    d.recorder = ExplodingRecorder(1)
    d.transcriber = FakeTranscriber(["add a login test"])
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.speaker.spoken == ["Lost the microphone. Check your input device."]
    assert d.bridge.injected == []
    # The next press still records and injects: the loop is not wedged.
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.bridge.injected == [("add a login test ", False)]


@pytest.mark.asyncio
async def test_empty_dictation_is_not_injected():
    d = _daemon(1, ["clean it up"])
    reopen = await d._handle_utterance("clean it up")
    assert reopen is True
    assert d.bridge.injected == []


# ------------------------------------------------------------------ auto-send


@pytest.mark.asyncio
async def test_dictation_auto_submits():
    d = _daemon(1, ["add a login test"], submit="auto")
    reopen = await d._handle_utterance("add a login test")
    # One guarded paste plus Enter, and the turn passes to Claude.
    assert d.bridge.injected == [("add a login test", True)]
    assert reopen is False


@pytest.mark.asyncio
async def test_dictation_stays_paste_only_in_voice_mode():
    d = _daemon(1, ["add a login test"], submit="voice")
    reopen = await d._handle_utterance("add a login test")
    assert d.bridge.injected == [("add a login test ", False)]
    assert reopen is True


@pytest.mark.asyncio
async def test_short_dictation_is_pasted_not_sent():
    # A one or two word transcript is as likely to be a misfire as an
    # instruction, and a wrong send costs a whole Claude turn.
    d = _daemon(1, ["run tests"], submit="auto")
    reopen = await d._handle_utterance("run tests")
    assert d.bridge.injected == [("run tests ", False)]
    assert reopen is True


@pytest.mark.asyncio
async def test_the_word_floor_is_configurable():
    d = _daemon(1, ["run tests"], submit="auto")
    d.cfg.hotkey.auto_send_min_words = 2
    await d._handle_utterance("run tests")
    assert d.bridge.injected == [("run tests", True)]


@pytest.mark.asyncio
async def test_send_it_submits_however_short_the_text_is():
    d = _daemon(1, ["go send it"], submit="auto")
    await d._handle_utterance("go send it")
    assert d.bridge.injected == [("go", True)]


@pytest.mark.asyncio
async def test_type_prefix_never_submits():
    d = _daemon(1, ["type hello world there"], submit="auto")
    reopen = await d._handle_utterance("type hello world there")
    assert d.bridge.injected == [("hello world there", False)]
    assert reopen is True


@pytest.mark.asyncio
async def test_scratch_that_still_discards_under_auto_send():
    d = _daemon(1, ["scratch that"], submit="auto")
    await d._handle_utterance("scratch that")
    assert d.bridge.injected == []
    assert d.bridge.keys == [("C-u",)]


@pytest.mark.asyncio
async def test_close_still_discards_under_auto_send():
    d = _daemon(1, ["close"], submit="auto")
    await d._handle_utterance("close")
    assert d.bridge.keys == [("C-u",)]


@pytest.mark.asyncio
async def test_interrupt_still_interrupts_under_auto_send():
    d = _daemon(1, ["stop claude"], submit="auto")
    await d._handle_utterance("stop claude")
    assert d.bridge.keys == [("Escape",)]
    assert d.bridge.injected == []


# ----------------------------------------------------------------------- cues


class OrderedRecorder(FakeRecorder):
    def __init__(self, order):
        super().__init__(1)
        self.order = order

    async def record(self, session, until_silence):
        self.order.append("record")
        return await super().record(session, until_silence)


@pytest.mark.asyncio
async def test_the_listen_cue_never_delays_the_microphone(monkeypatch):
    # The regression: play_cue awaits afplay's proc.wait(), so awaiting it
    # before record() put the whole chime in front of the open mic.
    order = []

    async def fake_cue(name):
        order.append(f"cue:{name}")

    monkeypatch.setattr(daemon_mod, "play_cue", fake_cue)
    d = _daemon(1, ["add a login test"], submit="voice")
    d.cfg.sound_cues = True
    d.cfg.hands_free = False
    d.recorder = OrderedRecorder(order)

    await d._listen_session(d.recorder.begin(), until_silence=False)
    await asyncio.sleep(0)  # let the fired-and-forgotten cue task run

    assert order == ["record", "cue:listen"]


@pytest.mark.asyncio
async def test_the_done_cue_never_delays_the_paste(monkeypatch):
    order = []

    async def fake_cue(name):
        order.append(f"cue:{name}")

    monkeypatch.setattr(daemon_mod, "play_cue", fake_cue)
    d = _daemon(1, ["add a login test"], submit="auto")
    d.cfg.sound_cues = True

    class OrderedBridge(FakeBridge):
        async def inject(self, text, submit):
            order.append("inject")
            await super().inject(text, submit)

    d.bridge = OrderedBridge()
    await d._handle_utterance("add a login test")
    await asyncio.sleep(0)

    assert order == ["inject", "cue:done"]


# --------------------------------------------------------------- tap sessions


class FakeHotkey:
    def __init__(self):
        self.cleared = 0

    def clear_tap(self):
        self.cleared += 1


@pytest.mark.asyncio
async def test_a_tap_switches_the_recording_to_end_on_silence():
    d = _daemon(1, ["unused"])
    d.hotkey = FakeHotkey()
    session = d.recorder.begin()
    d._ptt_session = session

    d._hotkey_released("tap")

    assert session.until_silence is True
    assert session.stopped is False   # still recording, the gate ends it
    assert d._tap_session is session


@pytest.mark.asyncio
async def test_a_second_tap_stops_that_recording():
    d = _daemon(1, ["unused"])
    d.hotkey = FakeHotkey()
    session = d.recorder.begin()
    d._ptt_session = session
    d._hotkey_released("tap")

    d._hotkey_released("tap")

    assert session.stopped is True
    assert d._tap_session is None
    assert d.hotkey.cleared >= 1      # the listener is re-armed


@pytest.mark.asyncio
async def test_a_hold_release_still_stops_immediately():
    d = _daemon(1, ["unused"])
    d.hotkey = FakeHotkey()
    session = d.recorder.begin()
    d._ptt_session = session

    d._hotkey_released("hold")

    assert session.stopped is True
    assert session.until_silence is False


@pytest.mark.asyncio
async def test_a_tap_after_the_recording_ended_rearms_the_key():
    # The race: the utterance endpointed on its own before the key came up.
    d = _daemon(1, ["unused"])
    d.hotkey = FakeHotkey()
    d._ptt_session = None

    d._hotkey_released("tap")

    assert d._tap_session is None
    assert d.hotkey.cleared == 1


@pytest.mark.asyncio
async def test_a_finished_recording_clears_the_tap_state():
    d = _daemon(1, ["add a login test"])
    d.hotkey = FakeHotkey()
    d.cfg.hands_free = False
    session = d.recorder.begin()
    d._ptt_session = session
    d._tap_session = session

    await d._listen_session(session, until_silence=False)

    assert d._tap_session is None
    assert d._ptt_session is None
    assert d.hotkey.cleared >= 1


# -------------------------------------------------------------------- defaults


def test_hands_free_is_off_by_default():
    # With auto-send on, a mic that reopens unasked can send room noise.
    assert Config().hands_free is False


# ------------------------------------------------------------------ the pill


@pytest.mark.asyncio
async def test_the_pill_shows_listening_on_the_keystroke():
    # The whole point of the pill: the user presses the key and sees it, with
    # nothing (prewarm, mic, TTS stop) allowed in front.
    order = []
    d = _daemon(1, ["add a login test"])
    d.cfg.hands_free = False
    d.recorder = OrderedRecorder(order)
    d.hud = FakeHud(order)

    d._hotkey_pressed()

    assert order == ["pill:listening"]
    assert d.hud.calls[0] == ("listening", "Listening", "")

    await asyncio.sleep(0.05)  # let the press's own tasks finish
    assert order[:3] == ["pill:listening", "record", "pill:finalizing"]


@pytest.mark.asyncio
async def test_the_pill_says_finalizing_then_clears_after_a_paste():
    d = _daemon(1, ["add a login test"])
    d.cfg.hands_free = False

    await d._listen_session(d.recorder.begin(), until_silence=False)

    # The words are in Claude's box now, so the pill has nothing left to say.
    assert d.hud.states == ["finalizing", "idle"]


@pytest.mark.asyncio
async def test_the_pill_says_sent_when_the_turn_goes_to_claude():
    d = _daemon(1, ["ship it send"])
    await d._handle_utterance("ship it send")
    assert d.hud.calls == [("sending", "Sent", "")]


@pytest.mark.asyncio
async def test_the_pill_shows_the_tool_claude_just_ran():
    d = _hook_daemon()
    await d._on_tool(
        {
            "session_id": "A",
            "prompt_id": "p-A",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"},
            "tool_response": {},
        }
    )
    assert d.hud.calls[-1] == ("thinking", "Thinking", "Bash: pytest -q")


@pytest.mark.asyncio
async def test_another_sessions_tool_never_reaches_the_pill():
    d = _hook_daemon()
    await d._on_stop(_stop("A"))  # Bol latches onto session A
    d.hud.calls.clear()

    await d._on_tool(
        {
            "session_id": "B",
            "prompt_id": "p-B",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "tool_response": {},
        }
    )

    assert d.hud.calls == []
    # And the pill's read of the session never binds Bol to one.
    assert d._bound_session == "A"


@pytest.mark.asyncio
async def test_a_permission_prompt_becomes_a_question_on_the_pill():
    d = _hook_daemon()
    await d._on_notification(
        {
            "hook_event_name": "Notification",
            "session_id": "A",
            "notification_type": "permission_prompt",
            "message": "Claude wants to run rm -rf build.",
        }
    )
    # The pill keeps the question, not the whole spoken sentence, and it
    # stays up: there is an answer outstanding.
    assert d.hud.calls == [("permission", "Claude wants to run rm -rf build.", "")]


@pytest.mark.asyncio
async def test_the_pill_follows_bol_speaking_and_then_clears():
    d = _hook_daemon()
    await d._speak("Tests pass, all thirty of them.")
    assert d.hud.calls == [
        ("speaking", "Tests pass, all thirty of them.", ""),
        ("idle", "", ""),
    ]


@pytest.mark.asyncio
async def test_a_mic_failure_turns_the_pill_red_and_leaves_it_up():
    d = _daemon(1, ["add a login test"])
    d.recorder = ExplodingRecorder(1)

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.hud.calls == [("error", "Lost the microphone. Check your input device.", "")]
    # No idle behind it: the remedy has to stay readable, and the pill takes
    # itself down after three seconds.
    assert "idle" not in d.hud.states


@pytest.mark.asyncio
async def test_a_transcription_failure_turns_the_pill_red():
    class DeafTranscriber(FakeTranscriber):
        async def transcribe(self, audio, sample_rate):
            raise RuntimeError("model went away")

    d = _daemon(1, ["unused"])
    d.transcriber = DeafTranscriber([])

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.hud.calls == [
        ("finalizing", "Finalizing", ""),
        ("error", "Couldn't transcribe that one. Try again.", ""),
    ]


# ------------------------------------------------------------- live words


class LiveRecorder(FakeRecorder):
    """A recorder that takes long enough for partials to arrive, and that
    remembers whether the daemon handed the recording a tap."""

    def __init__(self, utterance_count, pause=0.02):
        super().__init__(utterance_count)
        self.pause = pause
        self.taps = []

    async def record(self, session, until_silence):
        self.taps.append(session.tap)
        await asyncio.sleep(self.pause)
        return await super().record(session, until_silence)


class LiveTranscriber(FakeTranscriber):
    """Reports scripted partials, then waits for the daemon's sentinel the
    way the real streaming loop does."""

    def __init__(self, texts, partials=(("add a", "login"), ("add a login", "test"))):
        super().__init__(texts)
        self.partials = list(partials)
        self.stream_args = []
        self.ended = False
        self.fail = False

    async def stream(self, blocks, emit, *, context_size, chunk_ms, sample_rate):
        self.stream_args.append((context_size, chunk_ms, sample_rate))
        if self.fail:
            raise RuntimeError("this model does not stream")
        for committed, draft in self.partials:
            emit(committed, draft)
            await asyncio.sleep(0)
        while True:
            try:
                item = blocks.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.001)
                continue
            if item is None:
                self.ended = True
                return


def _live_daemon(monkeypatch, texts, partials=None, pause=0.02):
    d = _daemon(1, list(texts))
    # One recording, so the assertions are about one pill and one decode.
    d.cfg.hands_free = False
    # The throttle has its own tests; here it must not swallow a partial.
    monkeypatch.setattr(daemon_mod, "LIVE_PILL_HZ", 1000.0)
    d.recorder = LiveRecorder(1, pause=pause)
    kwargs = {} if partials is None else {"partials": partials}
    d.transcriber = LiveTranscriber(list(texts), **kwargs)
    return d


@pytest.mark.asyncio
async def test_the_pill_shows_words_while_you_are_still_talking(monkeypatch):
    d = _live_daemon(monkeypatch, ["add a login test for the parser"])

    await d._listen_session(d.recorder.begin(), until_silence=False)

    # Committed and draft arrive apart, so the panel can dim the unsure half.
    listening = [c for c in d.hud.calls if c[0] == "listening"]
    assert listening == [
        ("listening", "add a", "login"),
        ("listening", "add a login", "test"),
    ]
    # And they are on screen before the mic closes, not after.
    assert d.hud.states.index("finalizing") > d.hud.states.index("listening")


@pytest.mark.asyncio
async def test_partial_words_never_reach_claude(monkeypatch):
    # The whole safety property of the phase: the pill is a mirror, and the
    # bridge only ever sees the full-buffer decode.
    d = _live_daemon(monkeypatch, ["add a login test for the parser"])

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.bridge.injected == [("add a login test for the parser ", False)]
    await asyncio.sleep(0.01)  # the fake decoder polls its queue
    assert d.transcriber.ended  # the sentinel was delivered
    assert d.recorder.taps and d.recorder.taps[0] is not None


@pytest.mark.asyncio
async def test_the_live_decoder_gets_the_configured_stream_settings(monkeypatch):
    d = _live_daemon(monkeypatch, ["add a login test"])
    d.cfg.stt.stream_context = [256, 8]
    d.cfg.stt.stream_chunk_ms = 160

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.transcriber.stream_args == [([256, 8], 160, d.cfg.audio.sample_rate)]


@pytest.mark.asyncio
async def test_nothing_streamed_lands_on_the_pill_after_the_recording(monkeypatch):
    # A partial that arrives late would overwrite "Finalizing" with a
    # half-decoded sentence, right when the user is waiting on the real one.
    d = _live_daemon(monkeypatch, ["add a login test"], partials=[])

    async def stream(blocks, emit, **kwargs):
        while True:
            try:
                item = blocks.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.001)
                continue
            if item is None:
                # The decoder finishing one step behind the sentinel: this
                # word is already stale and must go nowhere.
                emit("too", "late")
                return

    d.transcriber.stream = stream
    await d._listen_session(d.recorder.begin(), until_silence=False)
    await asyncio.sleep(0.01)

    assert "listening" not in d.hud.states
    assert d.hud.states[0] == "finalizing"


@pytest.mark.asyncio
async def test_live_off_means_no_stream_and_no_tap(monkeypatch):
    d = _live_daemon(monkeypatch, ["add a login test"])
    d.cfg.stt.live = False

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.transcriber.stream_args == []
    assert d.recorder.taps == [None]
    assert d.bridge.injected == [("add a login test ", False)]


@pytest.mark.asyncio
async def test_a_transcriber_that_cannot_stream_is_simply_not_asked(monkeypatch):
    # [stt] engine = "none" leaves no transcriber at all, and an engine
    # without a streaming decoder must not break the recording either.
    d = _daemon(1, ["add a login test"])
    d.cfg.hands_free = False
    monkeypatch.setattr(daemon_mod, "LIVE_PILL_HZ", 1000.0)

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert not hasattr(d.transcriber, "stream")
    assert d.bridge.injected == [("add a login test ", False)]


@pytest.mark.asyncio
async def test_a_broken_live_decoder_costs_the_words_and_nothing_else(caplog):
    d = _daemon(1, ["add a login test"])
    d.cfg.hands_free = False
    d.recorder = LiveRecorder(1)
    d.transcriber = LiveTranscriber(["add a login test"])
    d.transcriber.fail = True

    with caplog.at_level(logging.DEBUG, logger="bol"):
        await d._listen_session(d.recorder.begin(), until_silence=False)

    assert "listening" not in d.hud.states
    assert d.bridge.injected == [("add a login test ", False)]
    failures = [r for r in caplog.records if "live words failed" in r.getMessage()]
    assert len(failures) == 1  # once, at debug, per recording


# ----------------------------------------------------------- the throttle


class FakeLoop:
    """Just the clock and the timer _LiveWords uses, under the test's thumb."""

    def __init__(self):
        self.now = 0.0
        self.timers = []

    def time(self):
        return self.now

    def call_later(self, delay, fn):
        timer = _Timer(fn)
        self.timers.append((self.now + delay, timer))
        return timer

    def advance(self, seconds):
        self.now += seconds
        due = [(at, t) for at, t in self.timers if at <= self.now]
        self.timers = [(at, t) for at, t in self.timers if at > self.now]
        for _at, timer in due:
            timer.fire()


class _Timer:
    def __init__(self, fn):
        self.fn = fn
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.fn()


def _live_words(interval=0.25):
    hud = FakeHud()
    loop = FakeLoop()
    return hud, loop, daemon_mod._LiveWords(hud, loop, interval)


def test_a_burst_of_partials_redraws_the_pill_once():
    hud, loop, live = _live_words()
    live.emit("add", "a")
    live.emit("add a", "login")
    live.emit("add a login", "test")

    assert hud.calls == [("listening", "add", "a")]  # the first one, at once


def test_the_last_partial_of_a_burst_is_always_delivered():
    hud, loop, live = _live_words()
    live.emit("add", "a")
    live.emit("add a", "login")
    live.emit("add a login", "test")
    loop.advance(0.25)

    # Not the middle one: the newest text is the only one worth showing.
    assert hud.calls == [
        ("listening", "add", "a"),
        ("listening", "add a login", "test"),
    ]


def test_closing_drops_the_pending_redraw_and_ends_the_decode():
    hud, loop, live = _live_words()
    live.blocks = _queue.Queue()
    live.emit("add", "a")
    live.emit("add a", "login")
    live.close()
    loop.advance(1.0)

    assert hud.calls == [("listening", "add", "a")]
    assert live.blocks is None
    assert live.emit("anything", "at all") is None
    assert hud.calls == [("listening", "add", "a")]


def test_closing_twice_is_harmless():
    hud, loop, live = _live_words()
    live.blocks = _queue.Queue()
    blocks = live.blocks
    live.close()
    live.close()

    assert blocks.get_nowait() is None
    assert blocks.empty()


def test_an_empty_partial_leaves_listening_alone():
    # Before the first word is decoded there is nothing to say, and blanking
    # the pill would read as the microphone having died.
    hud, loop, live = _live_words()
    live.emit("", "")

    assert hud.calls == []
