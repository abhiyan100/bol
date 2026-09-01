"""Microphone capture with two stop conditions:

- push-to-talk: caller stops the recording explicitly (hotkey released);
- hands-free: an energy gate ends the utterance after trailing silence.

Energy VAD is deliberately simple (RMS vs. rolling noise floor). It's a
protocol seam — swap in Silero later without touching the daemon.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque

import numpy as np
import sounddevice as sd

from ..config import AudioConfig

log = logging.getLogger("bol.audio")

_BLOCK_MS = 30


class Recorder:
    def __init__(self, cfg: AudioConfig) -> None:
        self._cfg = cfg
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        """Push-to-talk release: end the current recording."""
        self._stop.set()

    async def record(self, until_silence: bool) -> np.ndarray | None:
        """Record one utterance; returns float32 mono @ sample_rate, or None
        if nothing above the noise floor was captured."""
        cfg = self._cfg
        self._stop.clear()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        block = int(cfg.sample_rate * _BLOCK_MS / 1000)

        def callback(indata, _frames, _time, status):
            if status:
                log.debug("audio status: %s", status)
            loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())

        blocks: list[np.ndarray] = []
        noise_floor = deque(maxlen=20)  # first ~600ms establishes the floor
        speech_blocks = 0
        silence_blocks = 0
        silence_limit = max(1, cfg.silence_ms // _BLOCK_MS)
        max_blocks = cfg.max_utterance_s * 1000 // _BLOCK_MS
        heard_speech = False

        stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype="float32",
            blocksize=block,
            callback=callback,
        )
        with stream:
            while len(blocks) < max_blocks:
                if self._stop.is_set():
                    break
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                blocks.append(chunk)
                rms = float(np.sqrt(np.mean(chunk**2)) + 1e-9)
                if len(noise_floor) < noise_floor.maxlen and not heard_speech:
                    noise_floor.append(rms)
                floor = max(np.median(noise_floor) if noise_floor else 1e-4, 1e-4)
                if rms > floor * self._cfg.energy_threshold:
                    heard_speech = True
                    speech_blocks += 1
                    silence_blocks = 0
                elif heard_speech:
                    silence_blocks += 1
                    if until_silence and silence_blocks >= silence_limit:
                        break

        if not blocks:
            return None
        audio = np.concatenate(blocks)
        if speech_blocks * _BLOCK_MS < cfg.min_speech_ms:
            return None
        return audio
