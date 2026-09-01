from __future__ import annotations

from typing import Protocol

from ..config import Config
from ..hooks.events import StopEvent


class Summarizer(Protocol):
    async def summarize(self, event: StopEvent) -> str:
        """Return the sentence(s) Bol should speak for this finished turn."""
        ...


def build_summarizer(cfg: Config) -> Summarizer:
    from .template import TemplateSummarizer

    if cfg.summarizer.engine == "openrouter" and cfg.openrouter_key:
        from .openrouter import OpenRouterSummarizer

        return OpenRouterSummarizer(cfg)
    return TemplateSummarizer(cfg)
