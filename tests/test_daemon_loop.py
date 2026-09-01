"""Regression tests for the listening loop: the races and dead paths the code
reviews confirmed (session-token stops, hands-free chaining, multi-session
hook affinity, mic failures)."""

import asyncio

import numpy as np
import pytest

from bol.audio.capture import RecordingSession
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


def _daemon(utterances, texts):
    cfg = Config()
    cfg.sound_cues = False
    cfg.hands_free = True
    d = Daemon(cfg, text_mode=False)
    d.recorder = FakeRecorder(utterances)
    d.transcriber = FakeTranscriber(texts)
    d.bridge = FakeBridge()
    d.speaker = FakeSpeaker()
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
    # The stopped session yields no audio and nothing is injected…
    assert d.bridge.injected == []
    # …and a FRESH session is unaffected by the stale stop.
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
    # Session B's prompt was never announced, so nothing is armed…
    assert d._permission_session is None
    reopen = await d._handle_utterance("yes")
    # …and "yes" is plain dictation, not an approval keystroke.
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
