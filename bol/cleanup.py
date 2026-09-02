"""Transcript cleanup before injection.

Two tiers, chosen by what's trustworthy:

- Deterministic (always): filler words, stutters, doubled words, spoken
  technical tokens ("auth dot py" -> "auth.py", "dash dash verbose" ->
  "--verbose"), then spelling: the built-in tool names and whatever is in
  [vocabulary] words. Instant, and mechanically incapable of changing meaning.
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


# --------------------------------------------------------------- vocabulary
#
# Spelling, not rewriting. Two passes, both reversible by eye and neither of
# them able to change a word into a different word:
#
# 1. TOOL_NAMES, always on: phrases every transcriber gets wrong the same way.
# 2. [vocabulary] words: the user's own names, matched by edit distance.
#
# Both are deliberately timid. A wrong correction costs more than a missed
# one, because a missed one is still the word the user said. So: no bare
# "jason" (that is a person), no "cursor" (that is a text caret far more often
# than an editor), and nothing that is a common English word.

# phrase -> spelling. Matched case-insensitively on whole words, with any
# amount of whitespace between the parts of a phrase.
TOOL_NAMES = (
    ("claude code", "Claude Code"),
    ("cloud code", "Claude Code"),
    ("jason file", "JSON file"),
    ("jason format", "JSON format"),
    ("git hub", "GitHub"),
    ("pie test", "pytest"),
    ("you vee", "uv"),
    ("o auth", "OAuth"),
    ("github", "GitHub"),
    ("pytest", "pytest"),
    ("oauth", "OAuth"),
    ("codex", "Codex"),
)


def _phrase(words: str) -> re.Pattern:
    return re.compile(
        r"\b" + r"\s+".join(re.escape(part) for part in words.split()) + r"\b",
        re.IGNORECASE,
    )


# Longest first, so "claude code" is spelled before anything can claim
# "claude" on its own.
_TOOL_RULES = tuple(
    (_phrase(phrase), spelling)
    for phrase, spelling in sorted(TOOL_NAMES, key=lambda pair: -len(pair[0]))
)

# Words that are English before they are anything else. A vocabulary entry
# never rewrites one of these, which is what keeps "clause" from becoming
# "Claude" and "cursor" from becoming an editor. Exact matches still pass:
# if the entry IS the word, only its capitalization changes.
_COMMON_WORDS = frozenset(
    """
    a about after all also an and any are as ask at back be because been before
    being best better between both build built but by call called can case
    change check class clause clean cloud close code coder come commit could
    course cursor cut data day deal did do does doing done down each end even
    every fact feel few file files find first fix for from full get give go
    going good got great had half hand has have he her here him his hold home
    hook how i if in into is it its just keep kind know last late later least
    left less let life like line list little long look lot made make man many
    may me mean might mind mine more most move much must my name near need new
    next no not note now number of off often old on once one only open or order
    other our out over own page part pass past people place plan play please
    point put question quite rather read real really right run said same say
    school see seem send set she should show side since so some sort sound
    speak start state still stop such sure take talk tell test text than that
    the their them then there these they thing think this those though thought
    three through time to today together too took top turn two type under up
    us use used using very want was watch way we week well went were what when
    where which while who why will with word work world would write year yes
    yet you your
    """.split()
)

_TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")


def _edit_limit(entry: str) -> int:
    """How far a dictated token may be from an entry and still be it.

    Long words survive a bad syllable and are hard to hit by accident, short
    ones are one edit away from half the language. Under five characters the
    entry only ever fixes capitalization.
    """
    if len(entry) >= 8:
        return 2
    if len(entry) >= 5:
        return 1
    return 0


def _distance(a: str, b: str, limit: int) -> int | None:
    """Levenshtein distance, or None as soon as it is known to exceed limit."""
    if abs(len(a) - len(b)) > limit:
        return None
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        best = i
        for j, right in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left == right else 1),
                )
            )
            best = min(best, current[-1])
        if best > limit:
            return None
        previous = current
    return previous[-1] if previous[-1] <= limit else None


def _match_entry(span: str, entries: list[str], fuzzy: bool) -> str | None:
    """The vocabulary entry this span of text is, or None.

    fuzzy is False when every word in the span is a common English word: then
    only an exact match counts, so a real word can have its capitalization
    fixed but can never be turned into a different word.
    """
    lowered = span.lower()
    best: str | None = None
    best_distance: int | None = None
    for entry in entries:
        target = entry.lower()
        if lowered == target:
            return entry
        if not fuzzy:
            continue
        limit = _edit_limit(entry)
        if limit == 0:
            continue
        distance = _distance(lowered, target, limit)
        if distance is None:
            continue
        if best_distance is None or distance < best_distance:
            best, best_distance = entry, distance
    return best


def _apply_words(text: str, entries: list[str]) -> str:
    """Replace tokens and adjacent token pairs that are vocabulary entries.

    Pairs are tried first and matched with the space still in them, because
    the mistake this fixes is a transcriber splitting one name in two:
    "abhi yan" is one edit from "Abhiyan".
    """
    spans = [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(text)]
    out: list[str] = []
    last = 0
    index = 0
    while index < len(spans):
        start, end, token = spans[index]
        if index + 1 < len(spans):
            next_start, next_end, next_token = spans[index + 1]
            if not text[end:next_start].strip():
                pair = text[start:next_end]
                fuzzy = not (
                    token.lower() in _COMMON_WORDS
                    and next_token.lower() in _COMMON_WORDS
                )
                match = _match_entry(pair, entries, fuzzy)
                if match is not None:
                    out.append(text[last:start])
                    out.append(match)
                    last = next_end
                    index += 2
                    continue
        match = _match_entry(token, entries, token.lower() not in _COMMON_WORDS)
        if match is not None and match != token:
            out.append(text[last:start])
            out.append(match)
            last = end
        index += 1
    out.append(text[last:])
    return "".join(out)


def apply_vocabulary(text: str, words=()) -> str:
    """Spell tool names and the user's own words the way they are written.

    Runs after the deterministic rules and before any model, in every cleanup
    mode: it is a lookup table, so there is nothing for a model to improve
    and nothing for a timeout to lose.
    """
    if not text:
        return text
    for pattern, spelling in _TOOL_RULES:
        text = pattern.sub(lambda _m, value=spelling: value, text)
    entries = [str(word).strip() for word in (words or ())]
    entries = [entry for entry in entries if entry]
    if not entries:
        return text
    return _apply_words(text, entries)


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
    engine,
    text: str,
    deadline_s: float,
    use_llm: bool = False,
    cleaner=None,
    vocabulary=(),
) -> str:
    if len(text) < 8:
        return text
    base = apply_vocabulary(deterministic_clean(text), vocabulary)
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
