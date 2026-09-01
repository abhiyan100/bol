"""Tier-0 summarizer: free, instant, deterministic.

Builds the spoken reply from the tool log + Claude's own last message. No
network, no model. Covers the common case; the persona LLM is optional polish.
"""

from __future__ import annotations

import re
from collections import Counter

from ..config import Config
from ..hooks.events import StopEvent, ToolUse

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_MARKDOWN_NOISE = re.compile(r"[*_`#>|]+")
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)

_EDIT_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}
_READ_TOOLS = {"Read", "Grep", "Glob", "LS"}


def _spoken(text: str, max_sentences: int = 2) -> str:
    text = _CODE_BLOCK.sub(" code block ", text)
    text = _MARKDOWN_NOISE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = _SENTENCE_END.split(text)
    return " ".join(sentences[:max_sentences]).strip()


def describe_tools(tools: list[ToolUse]) -> str:
    if not tools:
        return ""
    edited = {t.detail.rsplit("/", 1)[-1] for t in tools if t.tool_name in _EDIT_TOOLS and t.detail}
    ran = sum(1 for t in tools if t.tool_name == "Bash")
    failed = [t for t in tools if not t.ok]
    other = Counter(
        t.tool_name for t in tools
        if t.tool_name not in _EDIT_TOOLS | _READ_TOOLS | {"Bash"}
    )

    parts: list[str] = []
    if edited:
        names = sorted(edited)
        if len(names) <= 3:
            parts.append(f"edited {', '.join(names)}")
        else:
            parts.append(f"edited {len(names)} files")
    if ran:
        parts.append(f"ran {ran} command{'s' if ran > 1 else ''}")
    for name, count in other.most_common(2):
        parts.append(f"used {name}{f' {count} times' if count > 1 else ''}")

    line = ", ".join(parts)
    if failed:
        flag = f"{len(failed)} of those failed" if len(failed) > 1 else "one of those failed"
        line = f"{line} — heads up, {flag}" if line else f"heads up, a tool call failed"
    return line


class TemplateSummarizer:
    def __init__(self, cfg: Config) -> None:
        self._name = cfg.summarizer.user_name

    async def summarize(self, event: StopEvent) -> str:
        activity = describe_tools(event.tools)
        gist = _spoken(event.last_assistant_message)
        pieces = [p for p in (activity.capitalize() if activity else "", gist) if p]
        if not pieces:
            body = "Claude's done, but it didn't say much."
        else:
            body = ". ".join(pieces).rstrip(".") + "."
        ask = f"What next{', ' + self._name if self._name else ''}?"
        return f"{body} {ask}"
