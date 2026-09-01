"""Capture tests: the energy gate must never eat a push-to-talk utterance,
and hands-free must still endpoint on its own. No hardware, no real time:
a fake InputStream pumps scripted blocks at the recorder's callback.
"""

import asyncio
import types

import numpy as np
import pytest

from bol.audio import capture
from bol.audio.capture import Recorder, _resolve_input_device
from bol.config import AudioConfig

BLOCK = 480  # 30ms @ 16kHz, the recorder's block size
_RNG = np.random.default_rng(7)

_DEVICES = [
    {"name": "MacBook Pro Microphone", "max_input_channels": 1},
    {"name": "MacBook Pro Speakers", "max_input_channels": 0},
    {"name": "Shure MV7", "max_input_channels": 2},
]


def _silence(n):
    return [_RNG.normal(0, 0.001, BLOCK).astype(np.float32) for _ in range(n)]


def _speech(n):
    return [_RNG.normal(0, 0.3, BLOCK).astype(np.float32) for _ in range(n)]


def _speech_int16(n):
    return [(_RNG.normal(0, 0.3, BLOCK) * 32767).astype(np.int16) for _ in range(n)]


def _cfg(**over):
    cfg = AudioConfig()
    cfg.silence_ms = 90       # 3 blocks of trailing silence ends an utterance
    cfg.min_speech_ms = 60    # 2 speech blocks clears the noise gate
    cfg.listen_window_s = 1   # 33 blocks of nobody talking gives the mic up
    cfg.max_utterance_s = 2   # backstop so a broken test can't hang
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


class FakeStream:
    """Stands in for sd.InputStream: replays scripted blocks into the
    callback from the event loop, one per tick."""

    def __init__(self, blocks, on_exhausted, kwargs):
        self.blocks = blocks
        self.kwargs = kwargs
        self.on_exhausted = on_exhausted
        self.closed = False

    def __enter__(self):
        self._task = asyncio.get_running_loop().create_task(self._pump())
        return self

    def __exit__(self, *_exc):
        self.closed = True
        self._task.cancel()
        return False

    async def _pump(self):
        for chunk in self.blocks:
            if self.closed:
                return
            self.kwargs["callback"](chunk.reshape(-1, 1), len(chunk), None, None)
            await asyncio.sleep(0)
        if self.on_exhausted is not None:
            self.on_exhausted()


def _fake_sd(monkeypatch, blocks=(), on_exhausted=None, devices=_DEVICES):
    """Swap sounddevice out for the fake; returns the list of opened streams."""
    opened = []

    def input_stream(**kwargs):
        stream = FakeStream(list(blocks), on_exhausted, kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(
        capture,
        "sd",
        types.SimpleNamespace(InputStream=input_stream, query_devices=lambda: devices),
    )
    return opened


async def _record(recorder, session, until_silence):
    audio = await recorder.record(session, until_silence=until_silence)
    await asyncio.sleep(0)  # let the pump task settle before the loop closes
    return audio


# ------------------------------------------------------------------ push-to-talk


async def test_push_to_talk_keeps_speech_from_the_first_block(monkeypatch):
    # The regression: the noise floor used to be sampled from the opening
    # blocks, so a user who talks the instant the key goes down set the floor
    # to their own voice and the whole utterance was dropped.
    recorder = Recorder(_cfg())
    session = recorder.begin()
    _fake_sd(monkeypatch, _speech(10), on_exhausted=session.request_stop)

    audio = await _record(recorder, session, until_silence=False)

    assert audio is not None
    assert audio.dtype == np.float32
    assert audio.size >= 8 * BLOCK


async def test_push_to_talk_returns_silence_too(monkeypatch):
    # Held the key, said nothing audible: hand it to the transcriber anyway,
    # an empty transcript is already handled downstream.
    recorder = Recorder(_cfg())
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(20), on_exhausted=session.request_stop)

    audio = await _record(recorder, session, until_silence=False)

    assert audio is not None
    assert audio.size >= 15 * BLOCK


async def test_push_to_talk_converts_int16_frames(monkeypatch):
    recorder = Recorder(_cfg())
    session = recorder.begin()
    _fake_sd(monkeypatch, _speech_int16(6), on_exhausted=session.request_stop)

    audio = await _record(recorder, session, until_silence=False)

    assert audio is not None
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0


async def test_tap_released_before_the_mic_opens(monkeypatch):
    recorder = Recorder(_cfg())
    session = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(10))
    session.request_stop()

    assert await _record(recorder, session, until_silence=False) is None
    assert opened == []  # the mic was never even opened


# -------------------------------------------------------------------- hands-free


async def test_hands_free_gives_the_mic_up_after_the_window(monkeypatch):
    recorder = Recorder(_cfg())
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(40))

    assert await _record(recorder, session, until_silence=True) is None


async def test_hands_free_endpoints_on_trailing_silence(monkeypatch):
    recorder = Recorder(_cfg())
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(5) + _speech(10) + _silence(10))

    audio = await _record(recorder, session, until_silence=True)

    assert audio is not None
    blocks = audio.size // BLOCK
    assert 15 <= blocks <= 21  # 5 + 10 + the 3 silence blocks that ended it


async def test_hands_free_stopped_midway_returns_none(monkeypatch):
    # Barge-in: the hotkey stops a hands-free recording that heard nothing.
    recorder = Recorder(_cfg())
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(10), on_exhausted=session.request_stop)

    assert await _record(recorder, session, until_silence=True) is None


async def test_hands_free_drops_a_blip_under_min_speech_ms(monkeypatch):
    recorder = Recorder(_cfg(min_speech_ms=300))
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(5) + _speech(2) + _silence(10))

    assert await _record(recorder, session, until_silence=True) is None


# ----------------------------------------------------------------- input device


@pytest.mark.parametrize(
    "spec, expected",
    [("", None), ("  ", None), ("2", 2), ("shure", 2), ("MACBOOK PRO MIC", 0)],
)
def test_resolve_input_device(monkeypatch, spec, expected):
    _fake_sd(monkeypatch)
    assert _resolve_input_device(spec) == expected


def test_resolve_input_device_rejects_unknown_name(monkeypatch):
    _fake_sd(monkeypatch)
    with pytest.raises(ValueError) as err:
        _resolve_input_device("webcam")
    message = str(err.value)
    assert "webcam" in message
    assert "Shure MV7" in message
    assert "Speakers" not in message  # output-only devices aren't offered


async def test_configured_device_reaches_the_stream(monkeypatch):
    recorder = Recorder(_cfg(input_device="shure"))
    session = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=session.request_stop)

    await _record(recorder, session, until_silence=False)

    assert opened[0].kwargs["device"] == 2


async def test_default_device_is_left_to_sounddevice(monkeypatch):
    recorder = Recorder(_cfg())
    session = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=session.request_stop)

    await _record(recorder, session, until_silence=False)

    assert opened[0].kwargs["device"] is None
