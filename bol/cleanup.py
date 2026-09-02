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

from . import mlx_thread

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


class TunedCleaner:
    """Bol's own fine-tuned 350M cleanup model, loaded in-process (it's tiny)
    on first use. Trained on exactly this task, so it gets the rewrite job we
    refuse to give generic small models."""

    SYSTEM = "Clean this voice dictation for a coding agent. Output only the cleaned text."

    def __init__(self, model: str) -> None:
        self._model_name = model
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            from mlx_lm import load

            log.info("loading cleanup model %s ...", self._model_name)
            self._model, self._tokenizer = load(self._model_name)
        return self._model, self._tokenizer

    def _generate(self, text: str) -> str:
        from mlx_lm import generate

        model, tokenizer = self._load()
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": text},
            ],
            add_generation_prompt=True,
        )
        return generate(model, tokenizer, prompt=prompt, max_tokens=120).strip()

    async def warmup(self) -> None:
        import asyncio

        await mlx_thread.run(self._load)

    async def clean(self, text: str, deadline_s: float) -> str:
        import asyncio

        try:
            out = await asyncio.wait_for(
                mlx_thread.run(self._generate, text), timeout=deadline_s
            )
        except Exception as exc:
            log.debug("tuned cleanup skipped (%s)", exc)
            return text
        out = out.strip().strip('"')
        if _suspicious(text, out):
            log.debug("tuned cleanup rejected: %r -> %r", text, out)
            return text
        return out


def build_cleaner(cfg) -> TunedCleaner | None:
    if not cfg.cleanup.model:
        return None
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        log.warning("cleanup model configured but mlx-lm not installed")
        return None
    return TunedCleaner(cfg.cleanup.model)


async def clean_transcript(
    engine, text: str, deadline_s: float, use_llm: bool = False, cleaner=None
) -> str:
    if len(text) < 8:
        return text
    base = deterministic_clean(text)
    # The tuned local model handles the polish only when there is no better
    # option; in api mode the user's own big model does it instead.
    if cleaner is not None and not use_llm:
        return await cleaner.clean(base, deadline_s)
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
