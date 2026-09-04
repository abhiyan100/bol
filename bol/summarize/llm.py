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
        # The few-shot turns teach shape, but a 1B model recites them: it once
        # told the user about a Docker build and react 19 that never happened.
        # Only a big API model gets the examples; the local one gets the
        # instructions alone, and every reply is checked against the turn.
        history = persona_examples(name) if self._cfg.llm.provider == "api" else []
        try:
            reply = await self._engine.complete(
                system=self.system_prompt,
                user=build_user_prompt(
                    describe_tools(event.tools), event.last_assistant_message, name
                ),
                max_tokens=120,
                deadline_s=self._cfg.summarizer.timeout_s,
                history=history,
            )
            reply = _spoken_sanitize(reply)
            if not grounded(reply, event):
                log.debug("llm summary not about this turn; using template: %r", reply)
                return await self._fallback.summarize(event)
            return reply
        except Exception as exc:
            log.debug("llm summary failed (%s); using template", exc)
            return await self._fallback.summarize(event)


# Words from the few-shot examples that have no business in a real reply
# unless the turn itself mentioned them.
_EXAMPLE_TELLS = ("react 19", "npm install", "docker", "pricing card", "styles.css",
                  "mockup", "peer dependency", "dark mode")
_CONTENT_WORD = re.compile(r"[a-z][a-z0-9]{4,}")


def _content_words(text: str) -> set[str]:
    return set(_CONTENT_WORD.findall((text or "").lower()))


def grounded(reply: str, event: StopEvent) -> bool:
    """Is this reply about the turn it was asked about?

    A small model handed a few-shot history sometimes answers with the
    example instead of the input. Two checks: no telltale example phrase
    the turn never mentioned, and at least one content word (five letters
    or more) shared with the agent's message or its tool log. A reply that
    fails is discarded for the template, which cannot invent anything.
    """
    if not reply or not reply.strip():
        return False
    low = reply.lower()
    source_text = " ".join(
        [event.last_assistant_message or ""]
        + [f"{t.tool_name} {t.detail or ''}" for t in event.tools]
    ).lower()
    for tell in _EXAMPLE_TELLS:
        if tell in low and tell not in source_text:
            return False
    shared = _content_words(reply) & _content_words(source_text)
    return bool(shared)


_SANITIZE = re.compile(r"\s*[—–]\s*")
_MARKDOWN = re.compile(r"[*_`#]+")


def _spoken_sanitize(text: str) -> str:
    text = _SANITIZE.sub(", ", text)
    return _MARKDOWN.sub("", text).strip()
