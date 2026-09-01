"""Persona summarizer over the shared LLM engine (local model or user's API).
Falls back to the deterministic template on any failure — the loop never goes
silent because a model hiccuped."""

from __future__ import annotations

import logging

from ..config import Config
from ..hooks.events import StopEvent
from .persona import PERSONA_SYSTEM_PROMPT, build_user_prompt
from .template import TemplateSummarizer, describe_tools

log = logging.getLogger("bol.summarize")


class LlmSummarizer:
    def __init__(self, cfg: Config, engine) -> None:
        self._cfg = cfg
        self._engine = engine
        self._fallback = TemplateSummarizer(cfg)
        name = cfg.summarizer.user_name
        self.system_prompt = PERSONA_SYSTEM_PROMPT.format(
            name_suffix=f", {name}" if name else ""
        )

    async def summarize(self, event: StopEvent) -> str:
        try:
            return await self._engine.complete(
                system=self.system_prompt,
                user=build_user_prompt(
                    describe_tools(event.tools), event.last_assistant_message
                ),
                max_tokens=120,
                deadline_s=self._cfg.summarizer.timeout_s,
            )
        except Exception as exc:
            log.debug("llm summary failed (%s); using template", exc)
            return await self._fallback.summarize(event)
