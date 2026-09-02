"""Capture tests: the speech gate must never eat a push-to-talk utterance,
until-silence must still endpoint on its own, and the prepared mic must be
built once, kept warm, and rebuilt when the device dies. No hardware: a fake
InputStream pumps scripted blocks at the recorder's callback.

These run on the energy gate, which turns scripted audio into a decision, so
the whole path from the callback to the endpoint is exercised. The hysteresis
itself is tested against scripted probabilities further down, and the gates
themselves in test_vad.py.
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

BLOCK = 512  # 32 ms @ 16 kHz: Silero's window, and the recorder's block
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
    # Scripted noise is not speech to a real VAD, so these run the energy
    # gate: it is the one that turns this audio into a decision.
    cfg.vad = "energy"
    cfg.silence_ms = 96       # 3 blocks of trailing silence ends an utterance
    cfg.min_speech_ms = 60    # 2 speech blocks clears the noise gate
    cfg.listen_window_s = 1   # 31 blocks of nobody talking gives the mic up
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


async def test_a_key_released_before_the_mic_opens(monkeypatch):
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
    recorder = Recorder(_cfg(pre_roll_ms=96))  # three blocks of pre-roll
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
    # A recording that starts as push-to-talk and is handed over to the
    # energy gate mid-flight. Levels are measured from the first block, so the
    # floor is ready when the flag flips.
    recorder = Recorder(_cfg())
    session = recorder.begin()
    opened = _fake_sd(monkeypatch, _silence(4) + _speech(8))

    async def release():
        await asyncio.sleep(0.03)
        session.until_silence = True   # handed to the gate
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
    recorder = Recorder(_cfg(pre_roll_ms=96))
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


# ------------------------------------------------------- endpointing decisions


class ScriptedGate:
    """A speech gate that reads its answers off a list.

    Endpointing is a decision about probabilities, so it is tested with
    probabilities. Scripted noise can only ever test the energy gate's idea
    of what a probability is, which test_vad.py does separately.
    """

    def __init__(self, script=(), tail=0.0):
        self.script = list(script)
        self.tail = tail
        self.seen = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def probability(self, _block):
        self.seen += 1
        return self.script.pop(0) if self.script else self.tail


async def test_one_loud_block_does_not_open_an_utterance(monkeypatch):
    # A door, a keyboard, a chair. Speech has to be confirmed by a second
    # block before the recording is allowed to be about anything.
    gate = ScriptedGate([0.1, 0.99, 0.1], tail=0.1)
    recorder = Recorder(_cfg(), gate=gate)
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(40))

    assert await _record(recorder, session, until_silence=True) is None
    assert gate.seen >= 31  # it gave the mic up at listen_window_s, not at max


async def test_two_blocks_over_the_start_threshold_open_it(monkeypatch):
    gate = ScriptedGate([0.1, 0.99, 0.99], tail=0.05)
    recorder = Recorder(_cfg(), gate=gate)
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(40))

    audio = await _record(recorder, session, until_silence=True)

    # Opened on blocks 2 and 3, then ended after three blocks under the
    # release threshold: 6 blocks, not the whole listen window.
    assert audio is not None
    assert audio.size // BLOCK == 6


async def test_a_dip_inside_a_word_does_not_end_the_utterance(monkeypatch):
    # The hysteresis: 0.4 is under the threshold that STARTS speech and over
    # the one that ends it, so a quiet syllable keeps the recording open.
    gate = ScriptedGate([0.9, 0.9, 0.4, 0.4, 0.4, 0.9, 0.2, 0.2, 0.2], tail=0.2)
    recorder = Recorder(_cfg(), gate=gate)
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(40))

    audio = await _record(recorder, session, until_silence=True)

    assert audio is not None
    assert audio.size // BLOCK == 9  # six of speech plus the three that ended it


async def test_the_gate_is_reset_for_every_recording(monkeypatch):
    # Silero is recurrent and the energy gate carries a noise floor. Either
    # one, left unreset, would judge this utterance by the last one.
    gate = ScriptedGate(tail=0.9)
    recorder = Recorder(_cfg(), gate=gate)
    first = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=first.request_stop)

    await _record(recorder, first, until_silence=False, close=False)
    second = recorder.begin()
    opened[0].feed(_speech(4), on_exhausted=second.request_stop)
    await _record(recorder, second, until_silence=False)

    assert gate.resets == 2


async def test_the_pre_roll_primes_the_gate_without_voting(monkeypatch):
    # The ring holds audio from before the press. It has to reach the gate,
    # or a Silero that has heard nothing judges the first syllable cold. It
    # must not be allowed to START the utterance: a door slam three hundred
    # milliseconds before the key went down is not the user talking.
    gate = ScriptedGate(tail=0.9)
    recorder = Recorder(_cfg(pre_roll_ms=96, warm_s=1.0), gate=gate)
    first = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=first.request_stop)

    await _record(recorder, first, until_silence=False, close=False)
    opened[0].feed(_silence(6))
    await asyncio.sleep(0.05)

    # Three blocks of pre-roll that all scream speech, then a room that says
    # nothing at all for the rest of the recording.
    gate.script = [0.99, 0.99, 0.99]
    gate.tail = 0.05
    second = recorder.begin()
    opened[0].feed(_silence(40))

    assert await _record(recorder, second, until_silence=True) is None
    assert gate.script == []  # the pre-roll was fed, and then ignored


# ------------------------------------------------------------ the wake monitor


async def test_the_monitor_hears_every_block_the_stream_produces(monkeypatch):
    # Wake mode's one-microphone rule: the keyword listener reads the blocks
    # this stream is already producing, with no recording in flight and no
    # second device opened.
    heard = []
    recorder = Recorder(_cfg())
    opened = _fake_sd(monkeypatch, _speech(6))

    await recorder.open()
    await recorder.monitor(heard.append)
    await asyncio.sleep(0.05)
    await recorder.close()

    assert opened[0].starts == 1
    assert len(heard) >= 5
    assert all(block.dtype == np.float32 for block in heard)


async def test_the_monitor_suspends_the_warm_window(monkeypatch):
    # warm_s exists to give a Bluetooth headset its profile back between
    # recordings. Wake mode is a deliberate decision to hold the mic instead,
    # so the timer that would release it must not fire.
    recorder = Recorder(_cfg(warm_s=0.01))
    session = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(4), on_exhausted=session.request_stop)

    await recorder.open()
    await recorder.monitor(lambda _block: None)
    await _record(recorder, session, until_silence=False, close=False)
    await asyncio.sleep(0.05)

    assert opened[0].active  # still listening, warm window or not
    await recorder.close()


async def test_dropping_the_monitor_hands_the_device_back(monkeypatch):
    recorder = Recorder(_cfg(warm_s=0.01))
    opened = _fake_sd(monkeypatch, _speech(4))

    await recorder.open()
    await recorder.monitor(lambda _block: None)
    await asyncio.sleep(0.02)
    assert opened[0].active

    await recorder.monitor(None)
    await asyncio.sleep(0.05)

    assert not opened[0].active
    await recorder.close()


async def test_a_monitor_that_raises_never_costs_the_recording(monkeypatch):
    # Same rule as the live-words tap: a spectator that throws must not take
    # the audio callback, and with it the microphone, down with it.
    def explode(_block):
        raise RuntimeError("the wake child died mid-frame")

    recorder = Recorder(_cfg())
    session = recorder.begin()
    opened = _fake_sd(monkeypatch, _speech(8), on_exhausted=session.request_stop)

    await recorder.open()
    await recorder.monitor(explode)
    audio = await _record(recorder, session, until_silence=False)

    assert audio is not None
    assert audio.size >= 6 * BLOCK
    assert opened[0].starts == 1


# ----------------------------------------------------------- how it ended

# end_reason is how a recording finished, and the daemon spends it on whether
# the user walked away from it. Every path that ends a recording has to name
# itself, or a cancelled one reads like a finished sentence.


async def test_a_release_names_its_ending(monkeypatch):
    recorder = Recorder(_cfg())
    session = recorder.begin()
    _fake_sd(monkeypatch, _speech(6), on_exhausted=lambda: session.request_stop("release"))

    audio = await _record(recorder, session, until_silence=False)

    assert audio is not None
    assert session.end_reason == "release"


async def test_an_unnamed_stop_is_just_stop(monkeypatch):
    # Barge-in and shutdown call request_stop() with nothing to say. That is
    # not the user finishing a sentence, so it must not read as one.
    recorder = Recorder(_cfg())
    session = recorder.begin()
    _fake_sd(monkeypatch, _speech(6), on_exhausted=session.request_stop)

    await _record(recorder, session, until_silence=False)

    assert session.end_reason == "stop"


async def test_the_gate_names_a_silence_ending(monkeypatch):
    # Two blocks of speech, then three under the release threshold: the
    # endpoint the user did not ask for.
    gate = ScriptedGate([0.1, 0.99, 0.99], tail=0.05)
    recorder = Recorder(_cfg(), gate=gate)
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(40))

    audio = await _record(recorder, session, until_silence=True)

    assert audio is not None
    assert session.end_reason == "silence"


async def test_the_listen_window_names_its_ending(monkeypatch):
    # Nobody spoke inside listen_window_s, so the mic was given up. No audio,
    # and an ending that was never the user's.
    gate = ScriptedGate(tail=0.05)
    recorder = Recorder(_cfg(), gate=gate)
    session = recorder.begin()
    _fake_sd(monkeypatch, _silence(60))

    assert await _record(recorder, session, until_silence=True) is None
    assert session.end_reason == "window"


async def test_a_session_pause_overrides_the_configured_one(monkeypatch):
    # A "type" dictation pauses for thinking where a conversational turn does
    # not, so the pause that ends it is the session's, not the config's.
    gate = ScriptedGate([0.99, 0.99] + [0.05] * 20, tail=0.05)
    recorder = Recorder(_cfg(silence_ms=96), gate=gate)
    session = recorder.begin()
    session.silence_ms = 320  # 10 blocks, not 3
    _fake_sd(monkeypatch, _silence(40))

    audio = await _record(recorder, session, until_silence=True)

    assert session.end_reason == "silence"
    # Two speech blocks then ten of silence, where the config would have cut
    # it at three.
    assert audio is not None
    assert audio.size // BLOCK == 12


async def test_a_session_window_overrides_the_configured_one(monkeypatch):
    # A microphone a trigger word opened has to give up in seconds, not in
    # the eight the hotkey's own hands-free reopen is allowed.
    gate = ScriptedGate(tail=0.05)
    recorder = Recorder(_cfg(listen_window_s=10), gate=gate)
    session = recorder.begin()
    session.window_ms = 320  # 10 blocks of nobody talking
    _fake_sd(monkeypatch, _silence(60))

    assert await _record(recorder, session, until_silence=True) is None
    assert session.end_reason == "window"
    assert gate.seen < 20  # it gave up at the session's window, not the config's


async def test_a_cancelled_recording_keeps_its_name(monkeypatch):
    # A click, or another app coming forward. The daemon reads this name and
    # drops the audio rather than transcribing it.
    gate = ScriptedGate([0.99] * 10, tail=0.99)
    recorder = Recorder(_cfg(), gate=gate)
    session = recorder.begin()
    _fake_sd(monkeypatch, _speech(40))

    session.request_stop(capture.CANCELLED)
    await _record(recorder, session, until_silence=True)

    assert session.end_reason == "cancelled"


async def test_the_utterance_cap_names_its_ending(monkeypatch):
    # max_utterance_s ran out mid-sentence, which is the one ending that is
    # certainly not the end of what someone was saying.
    recorder = Recorder(_cfg(max_utterance_s=1))
    session = recorder.begin()
    _fake_sd(monkeypatch, _speech(60))

    audio = await _record(recorder, session, until_silence=False)

    assert audio is not None
    assert session.end_reason == "max"


async def test_a_fresh_session_has_not_ended():
    assert Recorder(_cfg()).begin().end_reason == ""


async def test_the_first_answer_wins():
    # A hold released in the same breath as the gate endpointing. Whatever
    # actually ended the recording got there first; the loser must not
    # relabel a pause as a deliberate finish.
    session = Recorder(_cfg()).begin()
    session.note_end("silence")
    session.request_stop("release")

    assert session.end_reason == "silence"
    assert session.stopped is True


async def test_every_reason_is_documented():
    # The daemon reads these names; a new one nobody wrote down is a rule
    # nobody applied.
    assert set(capture.END_REASONS) == {
        "release", "silence", "window", "max", "cancelled", "stop",
    }
    assert capture.CANCELLED in capture.END_REASONS


# ------------------------------------------------------------- level meter

# The pill's meter reads the microphone from the audio callback, because that
# is the only place the audio exists in real time. Two things have to hold:
# the numbers are a usable 0..1 for a voice, and nothing is measured when
# nobody is recording.


def test_silence_and_a_voice_land_at_opposite_ends():
    quiet = capture.block_level(_silence(1)[0])
    loud = capture.block_level(_speech(1)[0])
    assert 0.0 <= quiet < 0.15
    assert loud > quiet
    assert 0.0 <= loud <= 1.0


def test_an_empty_or_broken_block_reads_as_silence():
    assert capture.block_level(np.zeros(BLOCK, dtype=np.float32)) == 0.0
    assert capture.block_level(np.zeros(0, dtype=np.float32)) == 0.0
    assert 0.0 <= capture.block_level(np.full(BLOCK, np.nan, dtype=np.float32)) <= 1.0


def test_the_meter_rises_faster_than_it_falls():
    # A meter that tracked every block would strobe at the block rate; one
    # that lagged the attack would miss the syllable it exists to show.
    speech = _speech(1)[0]
    rise = capture.block_level(speech, 0.0)
    settled = capture.block_level(speech, rise)
    assert 0.0 < rise < settled  # still climbing towards the same block
    fall = capture.block_level(np.zeros(BLOCK, dtype=np.float32), settled)
    assert 0.0 < fall < settled
    # 120 ms of silence is about four blocks: most of the way down, not all.
    assert fall > settled * 0.5


async def test_a_recording_feeds_the_level_hook(monkeypatch):
    levels = []
    recorder = Recorder(_cfg())
    recorder.on_level = levels.append
    session = recorder.begin()
    _fake_sd(monkeypatch, _speech(10), on_exhausted=session.request_stop)

    await _record(recorder, session, until_silence=False)

    assert len(levels) >= 5
    assert all(0.0 <= level <= 1.0 for level in levels)
    assert max(levels) > 0.1  # a voice moved it


async def test_nothing_is_measured_between_recordings(monkeypatch):
    # The stream stays warm after a recording. The meter must not keep
    # reporting into a pill that has moved on to thinking.
    levels = []
    recorder = Recorder(_cfg())
    recorder.on_level = levels.append
    session = recorder.begin()
    stream = _fake_sd(monkeypatch, _speech(6), on_exhausted=session.request_stop)

    await _record(recorder, session, until_silence=False, close=False)
    during = len(levels)
    stream[0].feed(_speech(10))
    await asyncio.sleep(0.05)  # the warm stream keeps calling the callback
    await recorder.close()

    assert during >= 3
    assert len(levels) == during


async def test_the_meter_costs_nothing_when_nobody_asked(monkeypatch):
    # on_level unset is the ordinary case (the pill is off, or too old a
    # daemon): the callback must not measure anything at all.
    recorder = Recorder(_cfg())
    session = recorder.begin()
    _fake_sd(monkeypatch, _speech(6), on_exhausted=session.request_stop)

    audio = await _record(recorder, session, until_silence=False)

    assert audio is not None
    assert recorder.on_level is None
