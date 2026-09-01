"""Trailing/leading voice-command grammar.

Commands ride on the final transcript itself — no second model, no intent
classifier. A short fuzzy match against the last few words decides what to do,
mirroring the infina.so UX ("say 'send'", "say 'type ...'"):

  "refactor the auth module send it"  -> SEND  text="refactor the auth module"
  "type hello world"                  -> TYPE  text="hello world"   (no Enter)
  "scratch that" / "close" / "cancel" -> DISCARD
  "stop listening"                    -> SLEEP  (disables hands-free loop)
  "interrupt" / "stop claude"         -> INTERRUPT (Escape to the pane)
  anything else                       -> DICTATE (buffered; "send it" submits)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class Action(Enum):
    SEND = auto()       # inject text (if any) and press Enter
    TYPE = auto()       # inject text without Enter
    DICTATE = auto()    # plain speech, buffer it
    DISCARD = auto()    # throw away the current buffer
    SLEEP = auto()      # stop hands-free listening
    INTERRUPT = auto()  # send Escape to Claude
    REPEAT = auto()     # re-speak the last reply


@dataclass
class Parsed:
    action: Action
    text: str = ""
    # "clean it up" rider: run the LLM cleanup pass on text before injecting.
    clean: bool = False


_PUNCT = re.compile(r"[.,!?;:]+")

# Trailing phrases → action. Longest match wins. Checked against the
# normalized tail of the utterance.
_TRAILING: list[tuple[str, Action]] = [
    ("send it off", Action.SEND),
    ("and send it", Action.SEND),
    ("send it", Action.SEND),
    ("send that", Action.SEND),
    ("send", Action.SEND),
    ("go ahead", Action.SEND),
    ("submit", Action.SEND),
]

# Whole-utterance phrases → action (no payload text).
_STANDALONE: dict[str, Action] = {
    "close": Action.DISCARD,
    "cancel": Action.DISCARD,
    "scratch that": Action.DISCARD,
    "never mind": Action.DISCARD,
    "nevermind": Action.DISCARD,
    "discard": Action.DISCARD,
    "stop listening": Action.SLEEP,
    "go to sleep": Action.SLEEP,
    "sleep": Action.SLEEP,
    "interrupt": Action.INTERRUPT,
    "stop claude": Action.INTERRUPT,
    "stop stop": Action.INTERRUPT,
    "say that again": Action.REPEAT,
    "repeat that": Action.REPEAT,
    "what did you say": Action.REPEAT,
}

_TYPE_PREFIX = re.compile(r"^type\s+", re.IGNORECASE)
_WRITE_PREFIX = re.compile(r"^write\s+", re.IGNORECASE)

_CLEAN_PHRASES = [
    "clean it up",
    "clean that up",
    "clean up",
    "fix it up",
    "tidy it up",
]


def _norm(text: str) -> str:
    return _PUNCT.sub("", text.strip().lower()).strip()


def parse_transcript(transcript: str) -> Parsed:
    raw = transcript.strip()
    if not raw:
        return Parsed(Action.DICTATE, "")

    norm = _norm(raw)

    if norm in _STANDALONE:
        return Parsed(_STANDALONE[norm])

    # Leading "type ..." → literal text, no submit, never cleaned (the user
    # asked for these exact characters).
    m = _TYPE_PREFIX.match(raw) or _WRITE_PREFIX.match(raw)
    if m:
        payload = raw[m.end():].strip()
        parsed_tail = _strip_trailing_send(payload)
        if parsed_tail is not None:
            return Parsed(Action.SEND, parsed_tail)
        return Parsed(Action.TYPE, payload)

    stripped = _strip_trailing_send(raw)
    if stripped is not None:
        payload, clean = _strip_trailing_clean(stripped)
        return Parsed(Action.SEND, payload, clean=clean)

    payload, clean = _strip_trailing_clean(raw)
    return Parsed(Action.DICTATE, payload, clean=clean)


def _strip_trailing_clean(raw: str) -> tuple[str, bool]:
    """Strip a trailing "clean it up" phrase (with an optional joining
    "and"/comma); returns (payload, clean_requested)."""
    norm = _norm(raw)
    for phrase in _CLEAN_PHRASES:
        for suffix in (" and " + phrase, " " + phrase):
            if norm.endswith(suffix) or norm == suffix.strip():
                n_words = len(suffix.split())
                words = raw.split()
                payload = " ".join(words[:-n_words]).rstrip(" ,.;")
                return payload, True
    return raw, False


def _strip_trailing_send(raw: str) -> str | None:
    """If the utterance ends with a send phrase, return the text before it."""
    norm = _norm(raw)
    for phrase, action in _TRAILING:
        if action is not Action.SEND:
            continue
        if norm == phrase:
            return ""
        if norm.endswith(" " + phrase):
            # Cut the phrase off the raw string by word count, preserving
            # original casing/punctuation of the payload.
            n_words = len(phrase.split())
            words = raw.split()
            payload = " ".join(words[:-n_words]).strip()
            return _PUNCT.sub("", payload) if not payload else payload.rstrip(" ,.;")
    return None
