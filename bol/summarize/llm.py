"""Persona summarizer over the shared LLM engine (local model or user's API).
Falls back to the deterministic template on any failure, so the loop never
goes silent because a model hiccuped."""

from __future__ import annotations

import logging
import re

from ..config import Config
from ..hooks.events import CLAUDE, StopEvent, display_name
from .persona import build_user_prompt, persona_examples, persona_system_prompt
from .template import TemplateSummarizer, describe_tools

log = logging.getLogger("bol.summarize")


class LlmSummarizer:
    def __init__(self, cfg: Config, engine) -> None:
        self._cfg = cfg
        self._engine = engine
        self._fallback = TemplateSummarizer(cfg)
        self._name = cfg.summarizer.user_name
        # Which agent the next reply is about. Updated by summarize(), and read
        # by system_prompt so the daemon's prewarm warms the prompt that is
        # actually coming rather than the one from before Codex showed up.
        self.agent = CLAUDE

    @property
    def system_prompt(self) -> str:
        prompt = persona_system_prompt(display_name(self.agent))
        if self._name:
            prompt += (
                f"\n- The developer's name is {self._name}. Drop it in "
                "occasionally, not every reply."
            )
        return prompt

    async def summarize(self, event: StopEvent) -> str:
        self.agent = event.agent
        # Thin turns starve the model into inventing detail; the
        # deterministic template is honest by construction.
        if len(event.last_assistant_message) < 40 and len(event.tools) < 2:
            return await self._fallback.summarize(event)
        name = display_name(event.agent)
        try:
            reply = await self._engine.complete(
                system=self.system_prompt,
                user=build_user_prompt(
                    describe_tools(event.tools), event.last_assistant_message, name
                ),
                max_tokens=120,
                deadline_s=self._cfg.summarizer.timeout_s,
                history=persona_examples(name),
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
