"""Regression tests for the listening loop — the races and dead paths the
first code review confirmed (session-token stops, hands-free chaining)."""

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
    d._hotkey_released()  # no session in flight — must not raise or affect anything


@pytest.mark.asyncio
async def test_auto_listen_noop_while_mic_owned():
    d = _daemon(1, ["hello"])
    async with d._listen_lock:
        await d._auto_listen()  # lock held elsewhere → must return, not deadlock
    assert d.recorder.calls == []


@pytest.mark.asyncio
async def test_permission_yes_presses_enter_and_yields_to_claude():
    d = _daemon(1, ["yes"])
    d._awaiting_permission = True
    reopen = await d._handle_utterance("yes")
    assert reopen is False
    assert ("Enter",) in d.bridge.keys
