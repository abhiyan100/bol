from __future__ import annotations

from typing import Protocol

from ..config import Config
from ..hooks.events import StopEvent


class Summarizer(Protocol):
    async def summarize(self, event: StopEvent) -> str:
        """Return the sentence(s) Bol should speak for this finished turn."""
        ...


def build_summarizer(cfg: Config, engine=None) -> Summarizer:
    from .template import TemplateSummarizer

    if cfg.summarizer.engine == "auto" and engine is not None:
        from .llm import LlmSummarizer

        # LlmSummarizer checks engine availability per call and falls back
        # to the template itself, so it's safe even before the engine warms.
        return LlmSummarizer(cfg, engine)
    return TemplateSummarizer(cfg)
