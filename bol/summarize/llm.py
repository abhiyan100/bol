"""Persona summarizer over the shared LLM engine (local model or user's API).
Falls back to the deterministic template on any failure, so the loop never
goes silent because a model hiccuped."""

from __future__ import annotations

import logging
import re

from ..config import Config
from ..hooks.events import StopEvent
from .persona import PERSONA_EXAMPLES, PERSONA_SYSTEM_PROMPT, build_user_prompt
from .template import TemplateSummarizer, describe_tools

log = logging.getLogger("bol.summarize")


class LlmSummarizer:
    def __init__(self, cfg: Config, engine) -> None:
        self._cfg = cfg
        self._engine = engine
        self._fallback = TemplateSummarizer(cfg)
        name = cfg.summarizer.user_name
        self.system_prompt = PERSONA_SYSTEM_PROMPT + (
            f"\n- The developer's name is {name}. Drop it in occasionally, "
            "not every reply." if name else ""
        )

    async def summarize(self, event: StopEvent) -> str:
        # Thin turns starve the model into inventing detail; the
        # deterministic template is honest by construction.
        if len(event.last_assistant_message) < 40 and len(event.tools) < 2:
            return await self._fallback.summarize(event)
        try:
            reply = await self._engine.complete(
                system=self.system_prompt,
                user=build_user_prompt(
                    describe_tools(event.tools), event.last_assistant_message
                ),
                max_tokens=120,
                deadline_s=self._cfg.summarizer.timeout_s,
                history=PERSONA_EXAMPLES,
            )
            return _spoken_sanitize(reply)
        except Exception as exc:
            log.debug("llm summary failed (%s); using template", exc)
            return await self._fallback.summarize(event)


_SANITIZE = re.compile(r"\s*[—–]\s*")
_MARKDOWN = re.compile(r"[*_`#]+")


def _spoken_sanitize(text: str) -> str:
    text = _SANITIZE.sub(", ", text)
    return _MARKDOWN.sub("", text).strip()
