"""Kokoro-82M speaker via mlx-audio: local neural TTS, Apache-2.0 model.

Loaded lazily in a worker thread: model init costs a few seconds once, then
generation is faster than real-time on Apple Silicon.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

from .. import mlx_thread
from ..config import Config
from .base import clamp_speech

log = logging.getLogger("bol.speak")


class KokoroSpeaker:
    def __init__(self, cfg: Config) -> None:
        from mlx_audio.tts.utils import load_model  # noqa: F401  (import check)

        self._cfg = cfg
        self._model = None
        self._lock = asyncio.Lock()
        self._playing = False

    def _ensure_model(self):
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            log.info("loading Kokoro model %s ...", self._cfg.tts.kokoro_model)
            self._model = load_model(self._cfg.tts.kokoro_model)
        return self._model

    def _generate(self, text: str) -> tuple[np.ndarray, int]:
        model = self._ensure_model()
        chunks = []
        sample_rate = 24000
        for result in model.generate(
            text=text,
            voice=self._cfg.tts.kokoro_voice,
            speed=self._cfg.tts.kokoro_speed,
        ):
            chunks.append(np.asarray(result.audio))
            sample_rate = getattr(result, "sample_rate", sample_rate)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio.astype(np.float32), sample_rate

    async def speak(self, text: str) -> None:
        async with self._lock:
            audio, rate = await mlx_thread.run(self._generate, clamp_speech(text))
            if audio.size == 0:
                return
            self._playing = True
            try:
                await asyncio.to_thread(self._play, audio, rate)
            finally:
                self._playing = False

    def _play(self, audio: np.ndarray, rate: int) -> None:
        sd.play(audio, samplerate=rate, blocking=True)

    async def stop(self) -> None:
        if self._playing:
            sd.stop()
