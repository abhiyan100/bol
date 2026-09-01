"""Microphone capture with two stop conditions:

- push-to-talk: caller stops the recording explicitly (hotkey released);
- hands-free: an energy gate ends the utterance after trailing silence.

Stop signalling uses per-recording session tokens, not shared state: begin()
mints a token synchronously at the triggering event, and request_stop() on a
token only ever ends *that* recording. A release for a session that never got
the mic, or that already finished, is inert by construction: no cross-talk,
no lost-stop races.

The energy gate runs hands-free only. Holding the key is already an explicit
request to record, so push-to-talk hands back whatever it captured and lets
the transcriber decide whether there were words in it. Gating there dropped
the utterance of anyone who starts talking the instant the key goes down,
because their own voice defined the noise floor.

Energy VAD is deliberately simple (RMS vs. an adaptive noise floor). It's a
protocol seam: swap in Silero later without touching the daemon.
"""

from __future__ import annotations

import asyncio
import logging

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
    that starts the recording (hotkey press / auto-listen decision)."""

    __slots__ = ("_stop",)

    def __init__(self) -> None:
        self._stop = asyncio.Event()

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

    def begin(self) -> RecordingSession:
        return RecordingSession()

    def _input_device(self) -> int | None:
        # Resolved once: the config is static, and a name lookup on every
        # hotkey press would add latency at the worst possible moment.
        if not self._device_resolved:
            self._device = _resolve_input_device(self._cfg.input_device)
            self._device_resolved = True
        return self._device

    async def record(
        self, session: RecordingSession, until_silence: bool
    ) -> np.ndarray | None:
        """Record one utterance; returns float32 mono @ sample_rate, or None.

        Push-to-talk (until_silence=False) returns everything captured; only an
        empty recording (a tap released before the mic opened) gives None.
        Hands-free (until_silence=True) nobody asked for, so it keeps the energy
        gate: it gives the mic up if no speech starts inside listen_window_s,
        endpoints after silence_ms, and discards anything under min_speech_ms.
        """
        cfg = self._cfg
        if session.stopped:
            return None  # quick tap: released before the mic even opened
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        block = int(cfg.sample_rate * _BLOCK_MS / 1000)

        def callback(indata, _frames, _time, status):
            if status:
                log.debug("audio status: %s", status)
            mono = indata[:, 0]
            if np.issubdtype(mono.dtype, np.integer):
                # The stream asks for float32, but a backend that hands back
                # int16 would overflow the RMS and break the transcriber.
                mono = mono.astype(np.float32) / float(np.iinfo(mono.dtype).max)
            loop.call_soon_threadsafe(
                queue.put_nowait, np.asarray(mono, dtype=np.float32).copy()
            )

        blocks: list[np.ndarray] = []
        levels: list[float] = []  # per-block RMS, feeds the adaptive floor
        speech_blocks = 0
        silence_blocks = 0
        silence_limit = max(1, cfg.silence_ms // _BLOCK_MS)
        max_blocks = cfg.max_utterance_s * 1000 // _BLOCK_MS
        window_blocks = cfg.listen_window_s * 1000 // _BLOCK_MS
        heard_speech = False

        stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype="float32",
            blocksize=block,
            device=self._input_device(),
            callback=callback,
        )
        with stream:
            while len(blocks) < max_blocks:
                if session.stopped:
                    break
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                blocks.append(chunk)
                if not until_silence:
                    continue  # the key release ends this one, not the energy
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
                    if silence_blocks >= silence_limit:
                        break
                elif len(blocks) >= window_blocks:
                    break  # hands-free reopen, nobody spoke, give the mic up

        if not blocks:
            return None
        if until_silence and speech_blocks * _BLOCK_MS < cfg.min_speech_ms:
            return None
        return np.concatenate(blocks)
