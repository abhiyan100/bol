"""Microphone capture with two stop conditions:

- push-to-talk: caller stops the recording explicitly (hotkey released);
- until-silence: an energy gate ends the utterance after trailing silence.

Stop signalling uses per-recording session tokens, not shared state: begin()
mints a token synchronously at the triggering event, and request_stop() on a
token only ever ends *that* recording. A release for a session that never got
the mic, or that already finished, is inert by construction: no cross-talk,
no lost-stop races. The token also carries the mutable until_silence flag, so
a tap can switch a running push-to-talk recording over to the energy gate.

The stream is prepared once and kept. Building an sd.InputStream costs about
33 ms on an M-series Mac and start() another 10 to 25 ms, and both used to sit
between the hotkey and the first syllable. Now the object is built once, the
callback feeds a 2 s ring buffer whenever the stream runs, and a recording
prepends pre_roll_ms of that ring so the opening word survives the press. The
stream is released warm_s after the last recording rather than held forever:
a Bluetooth headset pinned open stays in its tinny headset profile.

The energy gate runs only while session.until_silence is set. Holding the key
is already an explicit request to record, so push-to-talk hands back whatever
it captured and lets the transcriber decide whether there were words in it.
Gating there dropped the utterance of anyone who starts talking the instant
the key goes down, because their own voice defined the noise floor. Levels are
measured for every block regardless, so the gate is ready the moment the flag
flips mid-recording.

Energy VAD is deliberately simple (RMS vs. an adaptive noise floor). It's a
protocol seam: swap in Silero later without touching the daemon.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from time import perf_counter

import numpy as np
import sounddevice as sd

from ..config import AudioConfig

log = logging.getLogger("bol.audio")

_BLOCK_MS = 30
# Once speech has started, a block only counts as silence below this fraction
# of the start threshold. Hysteresis, so mid-word dips don't end the utterance.
_RELEASE_RATIO = 0.6
# The noise floor is the quiet fifth of what has been heard so far, not the
# first few blocks: a user who speaks immediately can't raise their own floor.
_FLOOR_PERCENTILE = 20
# How much recent audio the always-on ring keeps. Caps pre_roll_ms and keeps
# the buffer's memory flat no matter how long the stream stays warm.
_RING_MS = 2000


def _resolve_input_device(spec: str) -> int | None:
    """Map [audio] input_device to a sounddevice id. Accepts an index or a
    case-insensitive substring of the device name; empty means system default."""
    spec = spec.strip()
    if not spec:
        return None
    if spec.lstrip("-").isdigit():
        return int(spec)
    wanted = spec.lower()
    names: list[str] = []
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) < 1:
            continue
        name = device.get("name", "")
        if wanted in name.lower():
            return index
        names.append(name)
    available = ", ".join(names) if names else "none found"
    raise ValueError(
        f"no input device matches {spec!r}. Available: {available}. "
        "Set [audio] input_device to one of those names (or its index)."
    )


class RecordingSession:
    """Stop token for one recording. Mint via Recorder.begin() at the event
    that starts the recording (hotkey press / auto-listen decision).

    until_silence is mutable and read every loop iteration: a tap sets it
    after the recording is already running, which turns "record while held"
    into "record until they stop talking" without restarting the mic.
    """

    __slots__ = ("_stop", "until_silence")

    def __init__(self, until_silence: bool = False) -> None:
        self._stop = asyncio.Event()
        self.until_silence = until_silence

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()


class Recorder:
    def __init__(self, cfg: AudioConfig) -> None:
        self._cfg = cfg
        self._device: int | None = None
        self._device_resolved = False
        self._stream = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # Set for the duration of one recording; the callback reads it once.
        self._sink: asyncio.Queue[np.ndarray] | None = None
        self._ring: deque[np.ndarray] = deque(maxlen=max(1, _RING_MS // _BLOCK_MS))
        self._warm_task: asyncio.Task | None = None
        self._logged_latency = False

    def begin(self, until_silence: bool = False) -> RecordingSession:
        return RecordingSession(until_silence)

    def _input_device(self) -> int | None:
        # Resolved once: the config is static, and a name lookup on every
        # hotkey press would add latency at the worst possible moment.
        if not self._device_resolved:
            self._device = _resolve_input_device(self._cfg.input_device)
            self._device_resolved = True
        return self._device

    # ------------------------------------------------------------------ stream

    def _callback(self, indata, _frames, _time, status) -> None:
        # PortAudio's thread, not the event loop. deque.append on a bounded
        # deque is atomic and never grows, so the ring needs no lock.
        if status:
            log.debug("audio status: %s", status)
        mono = indata[:, 0]
        if np.issubdtype(mono.dtype, np.integer):
            # The stream asks for float32, but a backend that hands back
            # int16 would overflow the RMS and break the transcriber.
            mono = mono.astype(np.float32) / float(np.iinfo(mono.dtype).max)
        block = np.asarray(mono, dtype=np.float32).copy()
        self._ring.append(block)
        sink, loop = self._sink, self._loop
        if sink is not None and loop is not None:
            loop.call_soon_threadsafe(sink.put_nowait, block)

    def _build(self) -> None:
        cfg = self._cfg
        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype="float32",
            blocksize=int(cfg.sample_rate * _BLOCK_MS / 1000),
            device=self._input_device(),
            callback=self._callback,
        )
        self._running = False

    async def open(self) -> None:
        """Build the stream before the hotkey is armed.

        Construction is the expensive half and it does not touch the device's
        record light, so it is safe to pay it at startup and not at the press.
        """
        self._loop = asyncio.get_running_loop()
        if self._stream is None:
            self._build()

    def _start_stream(self) -> None:
        if self._running:
            return
        self._ring.clear()  # whatever is in there predates this device session
        started = perf_counter()
        self._stream.start()
        self._running = True
        if not self._logged_latency:
            self._logged_latency = True
            log.debug(
                "input stream start latency: %.1f ms",
                (perf_counter() - started) * 1000,
            )

    def _stop_stream(self) -> None:
        stream, self._running = self._stream, False
        self._ring.clear()
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as exc:  # a yanked device: rebuild on the next press
            log.debug("could not stop the input stream: %s", exc)
            self._discard()

    def _discard(self) -> None:
        """Drop the stream so the next record() builds a fresh one."""
        stream, self._stream = self._stream, None
        self._running = False
        self._ring.clear()
        if stream is None:
            return
        try:
            stream.close()
        except Exception as exc:
            log.debug("could not close the input stream: %s", exc)

    def _cancel_warm(self) -> None:
        task, self._warm_task = self._warm_task, None
        if task is not None:
            task.cancel()

    def _schedule_warm_release(self) -> None:
        if self._stream is None:
            return  # already discarded; nothing to hold open
        warm = float(self._cfg.warm_s)
        if warm <= 0 or self._loop is None:
            self._stop_stream()
            return
        self._warm_task = self._loop.create_task(self._release_after(warm))

    async def _release_after(self, warm: float) -> None:
        try:
            await asyncio.sleep(warm)
        except asyncio.CancelledError:
            return
        log.debug("warm window elapsed, releasing the microphone")
        self._stop_stream()

    async def close(self) -> None:
        """Release the device and forget the stream (daemon shutdown)."""
        self._cancel_warm()
        self._sink = None
        self._discard()

    def _pre_roll(self) -> list[np.ndarray]:
        count = int(self._cfg.pre_roll_ms) // _BLOCK_MS
        if count <= 0 or not self._ring:
            return []
        return list(self._ring)[-count:]

    # ----------------------------------------------------------------- capture

    async def record(
        self, session: RecordingSession, until_silence: bool
    ) -> np.ndarray | None:
        """Record one utterance; returns float32 mono @ sample_rate, or None.

        until_silence only ever turns the gate ON: the caller's argument and a
        later tap both set the session flag, and the loop re-reads it every
        block. With the gate off the recording returns everything captured;
        only an empty recording (a tap released before the mic opened) gives
        None. With it on, the recording gives the mic up if no speech starts
        inside listen_window_s, endpoints after silence_ms, and discards
        anything under min_speech_ms.
        """
        if session.stopped:
            return None  # quick tap: released before the mic even opened
        if until_silence:
            session.until_silence = True
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._cancel_warm()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        try:
            if self._stream is None:
                self._build()
            was_running = self._running
            # Snapshot before the sink is armed, so a block cannot land in
            # both. Losing the seam block costs 30 ms; doubling it stutters.
            pre_roll = self._pre_roll() if was_running else []
            self._sink = queue
            if not was_running:
                self._start_stream()
        except Exception:
            self._sink = None
            self._discard()  # dead device: the next press builds a new stream
            raise

        try:
            return await self._pump(session, queue, pre_roll)
        except Exception:
            self._discard()
            raise
        finally:
            self._sink = None
            self._schedule_warm_release()

    async def _pump(
        self,
        session: RecordingSession,
        queue: asyncio.Queue[np.ndarray],
        pre_roll: list[np.ndarray],
    ) -> np.ndarray | None:
        cfg = self._cfg
        loop = asyncio.get_running_loop()
        blocks: list[np.ndarray] = list(pre_roll)
        # Pre-roll feeds the noise floor but never the gate: it is room tone
        # from before the press, and a door slam in it must not count as the
        # start of speech.
        levels: list[float] = [
            float(np.sqrt(np.mean(block**2)) + 1e-9) for block in pre_roll
        ]
        speech_blocks = 0
        silence_blocks = 0
        silence_limit = max(1, cfg.silence_ms // _BLOCK_MS)
        max_blocks = cfg.max_utterance_s * 1000 // _BLOCK_MS
        window_blocks = cfg.listen_window_s * 1000 // _BLOCK_MS
        heard_speech = False
        # Wall clock as well as block count: a warm stream that goes silent
        # because the device vanished delivers nothing, and a block-only cap
        # would then spin here forever.
        deadline = loop.time() + cfg.max_utterance_s + 1

        while len(blocks) < max_blocks and loop.time() < deadline:
            if session.stopped:
                break
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            blocks.append(chunk)
            # Measured for every block, gate or no gate, so a tap that flips
            # until_silence mid-recording finds a floor already built.
            rms = float(np.sqrt(np.mean(chunk**2)) + 1e-9)
            levels.append(rms)
            floor = max(float(np.percentile(levels, _FLOOR_PERCENTILE)), 1e-4)
            start = floor * cfg.energy_threshold
            if rms > (start * _RELEASE_RATIO if heard_speech else start):
                heard_speech = True
                speech_blocks += 1
                silence_blocks = 0
            elif heard_speech:
                silence_blocks += 1
                if session.until_silence and silence_blocks >= silence_limit:
                    break
            elif session.until_silence and len(blocks) >= window_blocks:
                break  # nobody spoke, give the mic up

        if not blocks:
            return None
        if session.until_silence and speech_blocks * _BLOCK_MS < cfg.min_speech_ms:
            return None
        return np.concatenate(blocks)
