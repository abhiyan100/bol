"""Parakeet TDT via parakeet-mlx, the low-latency local engine FluidVoice
ships. ~0.05 RTF on Apple Silicon; model downloads once from Hugging Face."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

from ..config import Config

log = logging.getLogger("bol.stt")


class ParakeetTranscriber:
    def __init__(self, cfg: Config) -> None:
        import parakeet_mlx  # noqa: F401  (fail fast if missing)

        self._model_name = cfg.stt.parakeet_model
        self._model = None
        self._lock = asyncio.Lock()

    def _ensure_model(self):
        if self._model is None:
            from parakeet_mlx import from_pretrained

            log.info("loading Parakeet model %s ...", self._model_name)
            self._model = from_pretrained(self._model_name)
        return self._model

    async def warmup(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._ensure_model)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe, audio, sample_rate)

    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # parakeet-mlx takes a file path; hand it a temp 16-bit PCM WAV.
        model = self._ensure_model()
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16.tobytes())
            result = model.transcribe(path)
        finally:
            path.unlink(missing_ok=True)
        return result.text.strip()
