"""Persona summarizer via OpenRouter (free-tier models by default).

Falls back to the template summarizer on any error or timeout — the loop must
keep talking even when the free tier rate-limits (50 req/day without credits).
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from ..config import Config
from ..hooks.events import StopEvent
from .persona import PERSONA_SYSTEM_PROMPT, build_user_prompt
from .template import TemplateSummarizer, describe_tools

log = logging.getLogger("bol.summarize")

_API = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterSummarizer:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._fallback = TemplateSummarizer(cfg)
        name = cfg.summarizer.user_name
        self._system = PERSONA_SYSTEM_PROMPT.format(
            name_suffix=f", {name}" if name else ""
        )

    async def summarize(self, event: StopEvent) -> str:
        try:
            return await asyncio.wait_for(
                self._call(event), timeout=self._cfg.summarizer.timeout_s
            )
        except Exception as exc:
            log.warning("openrouter summarizer failed (%s); using template", exc)
            return await self._fallback.summarize(event)

    async def _call(self, event: StopEvent) -> str:
        payload = {
            "model": self._cfg.summarizer.openrouter_model,
            "max_tokens": 150,
            "messages": [
                {"role": "system", "content": self._system},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        describe_tools(event.tools), event.last_assistant_message
                    ),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._cfg.openrouter_key}",
            "HTTP-Referer": "https://github.com/bol-voice/bol",
            "X-Title": "Bol",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(_API, json=payload, headers=headers) as resp:
                data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {str(data)[:200]}")
        text = data["choices"][0]["message"]["content"].strip()
        if not text:
            raise RuntimeError("empty completion")
        return text
