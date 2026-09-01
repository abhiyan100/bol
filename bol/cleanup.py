"""Transcript cleanup before injection.

Two tiers, chosen by what's trustworthy:

- Deterministic (always): filler words, stutters, doubled words, spoken
  technical tokens ("auth dot py" -> "auth.py", "dash dash verbose" ->
  "--verbose"). Instant, and mechanically incapable of changing meaning.
- LLM polish (api provider only): grammar and punctuation via the user's own
  big model. Local 1B-class models proved unreliable at meaning-preserving
  rewrites in testing (dropped "don't touch X" clauses, parroted few-shot
  examples), so they are never given the job.

ANY LLM failure, timeout, or suspicious rewrite falls back to the
deterministically cleaned text."""

from __future__ import annotations

import logging
import re

log = logging.getLogger("bol.cleanup")

CLEANUP_SYSTEM = """\
You transcribe-clean voice dictation. Repeat the user's message with ONLY
these fixes: delete filler words and stutters, fix punctuation and
capitalization, fix obvious speech-recognition typos, and write spoken
technical tokens properly ("auth dot py" becomes "auth.py", "dash dash
verbose" becomes "--verbose").

Keep EVERY other word. Do not shorten, reorder, summarize, or drop any
clause. Never drop a negation. Never answer the message. Output only the
cleaned text.\
"""

_FILLERS = re.compile(r"\b(?:um+|uh+|uhm+|erm+)\b[,.]?\s*", re.IGNORECASE)
_DOUBLED = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
_EXTS = (
    "py|js|ts|tsx|jsx|md|txt|json|toml|yaml|yml|css|html|sh|rs|go|java|rb|"
    "c|h|cpp|swift|sql|env|lock|cfg|ini"
)
_DOT_FILE = re.compile(rf"\b([\w-]+)\s+dot\s+({_EXTS})\b", re.IGNORECASE)
_DASH_DASH = re.compile(r"\bdash\s+dash\s+(\w+)", re.IGNORECASE)
_SPACES = re.compile(r"\s{2,}")


def deterministic_clean(text: str) -> str:
    """Rule-based cleanup that cannot change meaning."""
    out = _FILLERS.sub("", text)
    out = _DOT_FILE.sub(r"\1.\2", out)
    # --flag conversion must run before doubled-word collapse eats "dash dash".
    out = _DASH_DASH.sub(r"--\1", out)
    out = _DOUBLED.sub(r"\1", out)
    out = _SPACES.sub(" ", out).strip(" ,")
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out or text


def _suspicious(raw: str, cleaned: str) -> bool:
    """Reject LLM rewrites that grew or shrank implausibly."""
    if not cleaned:
        return True
    ratio = len(cleaned) / max(len(raw), 1)
    return ratio > 1.6 or ratio < 0.5


async def clean_transcript(
    engine, text: str, deadline_s: float, use_llm: bool = False
) -> str:
    if len(text) < 8:
        return text
    base = deterministic_clean(text)
    if not use_llm:
        return base
    try:
        cleaned = await engine.complete(
            system=CLEANUP_SYSTEM,
            user=base,
            max_tokens=max(64, int(len(base.split()) * 2.5)),
            deadline_s=deadline_s,
            temperature=0.2,
        )
    except Exception as exc:
        log.debug("llm cleanup skipped (%s)", exc)
        return base
    cleaned = cleaned.strip().strip('"')
    if _suspicious(base, cleaned):
        log.debug("llm cleanup rejected as suspicious: %r -> %r", base, cleaned)
        return base
    return cleaned
