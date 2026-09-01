from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from ..config import Config

log = logging.getLogger("bol.stt")


class Transcriber(Protocol):
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...
    async def warmup(self) -> None: ...


def build_transcriber(cfg: Config) -> Transcriber | None:
    if cfg.stt.engine == "none":
        return None
    if cfg.stt.engine == "parakeet":
        try:
            from .parakeet import ParakeetTranscriber

            return ParakeetTranscriber(cfg)
        except ImportError:
            log.error(
                "parakeet-mlx not installed — run: uv sync --extra stt "
                "(voice input disabled; text mode still works)"
            )
            return None
    raise ValueError(f"unknown stt engine: {cfg.stt.engine}")
