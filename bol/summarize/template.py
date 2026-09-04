"""Tier-0 summarizer: free, instant, deterministic.

Builds the spoken reply from the tool log + the agent's own last message. No
network, no model. Covers the common case; the persona LLM is optional polish.

Which agent it names comes from the event, so a Codex turn is never reported
as Claude's.
"""

from __future__ import annotations

import re
from collections import Counter

from ..config import Config
from ..hooks.events import StopEvent, ToolUse, display_name

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_MARKDOWN_NOISE = re.compile(r"[*_`#>|]+")
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)

# apply_patch is how Codex edits a file; events.py gives it the touched path
# as its detail, so it belongs with the rest of the edit tools here.
_EDIT_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit", "apply_patch"}
_READ_TOOLS = {"Read", "Grep", "Glob", "LS"}

# A markdown table or a log dump has no sentence punctuation, so the split
# below returns the whole message as one "sentence". This is spoken aloud, so
# cap it: 400 characters is about 25 seconds, already generous.
SPOKEN_CHARS = 400


def _cap(text: str, limit: int = SPOKEN_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.")
    return f"{cut or text[:limit]}, and more"


def _spoken(text: str, max_sentences: int = 2) -> str:
    text = _CODE_BLOCK.sub(" code block ", text)
    text = _MARKDOWN_NOISE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = _SENTENCE_END.split(text)
    return _cap(" ".join(sentences[:max_sentences]).strip())


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
        line = f"{line}, heads up, {flag}" if line else "heads up, a tool call failed"
    return line


class TemplateSummarizer:
    def __init__(self, cfg: Config) -> None:
        self._name = cfg.summarizer.user_name

    async def summarize(self, event: StopEvent) -> str:
        """State, then what it did, then what it says. No question of Bol's own.

        The relay shape the user asked for: "Claude is done. It edited two
        files and ran the tests. It says the login test passes." A failed tool
        call turns the opening into "hit a problem"; a question mark in the
        agent's message turns it into "needs you".
        """
        agent = display_name(event.agent)
        activity = describe_tools(event.tools)
        gist = _spoken(event.last_assistant_message)
        failed = any(not tool.ok for tool in event.tools)
        asks = gist.rstrip().endswith("?")
        if failed:
            state = f"{agent} hit a problem."
        elif asks:
            state = f"{agent} needs you."
        else:
            state = f"{agent} is done."
        if self._name:
            state = f"{self._name}, {state[0].lower()}{state[1:]}"
        parts = [state]
        if activity:
            parts.append(f"It {activity}.")
        if gist:
            parts.append(f"It's asking: {gist}" if asks else f"It says: {gist}")
        elif not activity:
            parts.append("It didn't say much.")
        return " ".join(parts)
