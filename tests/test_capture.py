"""Capture tests: the energy gate must never eat a push-to-talk utterance,
until-silence must still endpoint on its own, and the prepared mic must be
built once, kept warm, and rebuilt when the device dies. No hardware: a fake
InputStream pumps scripted blocks at the recorder's callback.
"""

import asyncio
import logging
import queue as _queue
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
    """Stands in for sd.InputStream: a long-lived object that replays
    scripted blocks into the callback while it is started, one per tick.
    Tests feed it more blocks to simulate a mic that keeps running between
    recordings."""

    def __init__(self, blocks, on_exhausted, kwargs):
        self.blocks = list(blocks)
        self.kwargs = kwargs
        self.on_exhausted = on_exhausted
        self.active = False
        self.closed = False
        self.starts = 0
        self.fail_start = False
        self._task = None

    def feed(self, blocks, on_exhausted=None):
        self.blocks.extend(blocks)
        if on_exhausted is not None:
            self.on_exhausted = on_exhausted

    def start(self):
        if self.fail_start:
            raise OSError("PortAudioError: device unavailable")
        self.starts += 1
        self.active = True
        self._task = asyncio.get_running_loop().create_task(self._pump())

    def stop(self):
        self.active = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def close(self):
        self.stop()
        self.closed = True

    async def _pump(self):
        while self.active:
            if self.blocks:
                chunk = self.blocks.pop(0)
                self.kwargs["callback"](chunk.reshape(-1, 1), len(chunk), None, None)
                await asyncio.sleep(0)
                continue
            if self.on_exhausted is not None:
                fire, self.on_exhausted = self.on_exhausted, None
                fire()
            await asyncio.sleep(0.001)


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


async def _record(recorder, session, until_silence, close=True):
    audio = await recorder.record(session, until_silence=until_silence)
    await asyncio.sleep(0)  # let the pump task settle before the loop closes
    if close:
        # Releases the device and cancels the warm-window task, so nothing is
        # left pending when the test's event loop closes.
        await recorder.close()
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


# ---------------------------------------------------------------- prepared mic


async def test_open_builds_the_stream_without_taking_the_device(monkeypatch):
    # Construction is the expensive half (about 33 ms on an M-series Mac) and
    # it must be paid at startup, not at the press. Starting it is separate,
    # so nothing holds the mic until someone actually talks.
    recorder = Recorder(_cfg())
    opened = _fake_sd(monkeypatch)

    await recorder.open()

    assert len(opened) == 1
    assert opened[0].starts == 0
    await recorder.close()


async def test_the_stream_is_built_once_and_reused(monkeypatch):
    recorder = Recorder(_cfg())
    first = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=first.request_stop)

    await _record(recorder, first, until_silence=False, close=False)
    stream = opened[0]
    second = recorder.begin()
    stream.feed(_speech(4), on_exhausted=second.request_stop)
    audio = await _record(recorder, second, until_silence=False)

    assert opened == [stream]   # one InputStream across both recordings
    assert stream.starts == 1   # and the hardware was only started once
    assert audio is not None


async def test_the_warm_window_releases_the_microphone(monkeypatch):
    # Codex's Bluetooth note: a stream held open forever keeps a headset in
    # its tinny headset profile, so the warm window has to actually expire.
    recorder = Recorder(_cfg(warm_s=0.01))
    first = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=first.request_stop)

    await _record(recorder, first, until_silence=False, close=False)
    assert opened[0].active
    await asyncio.sleep(0.05)
    assert not opened[0].active   # released, but not thrown away

    second = recorder.begin()
    opened[0].feed(_speech(4), on_exhausted=second.request_stop)
    audio = await _record(recorder, second, until_silence=False)

    assert opened == [opened[0]]  # restarted, not rebuilt
    assert opened[0].starts == 2
    assert audio is not None


async def test_the_start_latency_is_measured_once(monkeypatch, caplog):
    recorder = Recorder(_cfg(warm_s=0.01))
    first = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=first.request_stop)

    with caplog.at_level(logging.DEBUG, logger="bol.audio"):
        await _record(recorder, first, until_silence=False, close=False)
        await asyncio.sleep(0.05)  # warm window elapses, next press restarts
        second = recorder.begin()
        opened[0].feed(_speech(4), on_exhausted=second.request_stop)
        await _record(recorder, second, until_silence=False)

    assert opened[0].starts == 2
    measured = [r for r in caplog.records if "start latency" in r.getMessage()]
    assert len(measured) == 1  # one debug line, not one per press


async def test_pre_roll_from_a_warm_stream_is_prepended(monkeypatch):
    recorder = Recorder(_cfg(pre_roll_ms=90))  # three blocks of pre-roll
    first = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=first.request_stop)

    await _record(recorder, first, until_silence=False, close=False)
    stream = opened[0]
    assert stream.active  # warm window holds it open
    stream.feed(_silence(6))  # room tone, with nobody recording
    await asyncio.sleep(0.05)

    second = recorder.begin()
    stream.feed(_speech(8), on_exhausted=second.request_stop)
    audio = await _record(recorder, second, until_silence=False)

    assert opened == [stream]
    assert audio.size > 3 * BLOCK
    # The first three blocks are the room tone the ring was holding when the
    # press landed: audio from BEFORE record() was called.
    head = audio[: 3 * BLOCK]
    assert float(np.sqrt(np.mean(head**2))) < 0.01
    assert float(np.sqrt(np.mean(audio[3 * BLOCK:] ** 2))) > 0.05


async def test_a_dead_stream_is_dropped_and_rebuilt(monkeypatch):
    stop = {"fn": lambda: None}
    recorder = Recorder(_cfg(warm_s=0.01))
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=lambda: stop["fn"]())

    first = recorder.begin()
    stop["fn"] = first.request_stop
    await _record(recorder, first, until_silence=False, close=False)
    await asyncio.sleep(0.05)     # warm window elapses; the stream goes cold
    opened[0].fail_start = True   # and the device disappears in the meantime

    dead = recorder.begin()
    with pytest.raises(OSError):
        await recorder.record(dead, until_silence=False)
    assert opened[0].closed       # dropped, not kept around broken
    assert len(opened) == 1

    third = recorder.begin()
    stop["fn"] = third.request_stop
    audio = await _record(recorder, third, until_silence=False)

    assert len(opened) == 2       # the next press gets a fresh stream
    assert audio is not None


# ------------------------------------------------------------- mid-flight gate


async def test_until_silence_set_mid_recording_endpoints_the_utterance(
    monkeypatch,
):
    # The tap: the recording starts as push-to-talk, and the key release a
    # moment later hands the ending over to the energy gate. Levels are
    # measured from the first block, so the floor is ready when the flag flips.
    recorder = Recorder(_cfg())
    session = recorder.begin()
    opened = _fake_sd(monkeypatch, _silence(4) + _speech(8))

    async def release():
        await asyncio.sleep(0.03)
        session.until_silence = True   # the tap
        opened[0].feed(_silence(10))   # and then the speaker stops talking

    asyncio.get_running_loop().create_task(release())
    audio = await _record(recorder, session, until_silence=False)

    assert audio is not None
    blocks = audio.size // BLOCK
    # 4 + 8 plus the three silence blocks that endpointed it, not the cap.
    assert 12 <= blocks <= 18


# ------------------------------------------------------------------- the tap


def _tapped(tap):
    """Everything a queue tap has been handed so far, as one array."""
    blocks = []
    while True:
        try:
            blocks.append(tap.get_nowait())
        except _queue.Empty:
            break
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


async def test_the_tap_sees_the_same_audio_as_the_recording(monkeypatch):
    # The live decoder's copy. It may run a block ahead of the buffer (the
    # sink hops through the event loop and the tap does not), but it must
    # never be a different recording.
    recorder = Recorder(_cfg())
    session = recorder.begin()
    session.tap = _queue.Queue()
    _fake_sd(monkeypatch, _speech(10), on_exhausted=session.request_stop)

    audio = await _record(recorder, session, until_silence=False)
    heard = _tapped(session.tap)

    assert audio is not None
    assert heard.size >= audio.size
    assert np.array_equal(heard[: audio.size], audio)


async def test_the_tap_gets_the_pre_roll_before_the_live_blocks(monkeypatch):
    # The word that started before the key went down is exactly the word a
    # live view must not open by missing.
    recorder = Recorder(_cfg(pre_roll_ms=90))
    first = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=first.request_stop)

    await _record(recorder, first, until_silence=False, close=False)
    stream = opened[0]
    stream.feed(_silence(6))  # room tone, with nobody recording
    await asyncio.sleep(0.05)

    second = recorder.begin()
    second.tap = _queue.Queue()
    stream.feed(_speech(8), on_exhausted=second.request_stop)
    audio = await _record(recorder, second, until_silence=False)
    heard = _tapped(second.tap)

    assert audio is not None
    assert np.array_equal(heard[: audio.size], audio)
    # The first three blocks are the ring's room tone, ahead of the speech.
    assert float(np.sqrt(np.mean(heard[: 3 * BLOCK] ** 2))) < 0.01
    assert float(np.sqrt(np.mean(heard[3 * BLOCK : 6 * BLOCK] ** 2))) > 0.05


async def test_a_tap_that_raises_never_costs_the_recording(monkeypatch):
    # The tap runs on PortAudio's thread. A listener that died has to be a
    # non-event: the recording is the thing the user actually asked for.
    def angry(_block):
        raise RuntimeError("the live decoder fell over")

    recorder = Recorder(_cfg())
    session = recorder.begin()
    session.tap = angry
    _fake_sd(monkeypatch, _speech(10), on_exhausted=session.request_stop)

    audio = await _record(recorder, session, until_silence=False)

    assert audio is not None
    assert audio.size >= 8 * BLOCK


async def test_the_tap_stops_when_its_recording_does(monkeypatch):
    # The stream stays warm between recordings; the tap must not, or the
    # decoder would keep transcribing the room after the key came up.
    recorder = Recorder(_cfg())
    session = recorder.begin()
    session.tap = _queue.Queue()
    opened = _fake_sd(monkeypatch, _speech(6), on_exhausted=session.request_stop)

    await _record(recorder, session, until_silence=False, close=False)
    during = _tapped(session.tap).size
    opened[0].feed(_speech(8))  # the warm stream keeps running
    await asyncio.sleep(0.05)
    after = _tapped(session.tap).size
    await recorder.close()

    assert during > 0
    assert after == 0
