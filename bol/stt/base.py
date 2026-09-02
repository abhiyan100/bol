from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from ..config import Config

log = logging.getLogger("bol.stt")


# Put this on a live stream's block queue to end the loop. None rather than a
# sentinel object so anything that can hold blocks can also carry the stop.
STREAM_END = None


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
                "parakeet-mlx not installed; voice input disabled, text mode "
                "still works. Reinstall with the stt extra: uv tool install "
                "--force \"bol[stt,llm] @ git+https://github.com/abhiyan100/bol\" "
                "(from a clone: uv sync --extra stt)"
            )
            return None
    raise ValueError(f"unknown stt engine: {cfg.stt.engine}")
