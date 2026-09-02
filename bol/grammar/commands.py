"""Trailing/leading voice-command grammar.

Commands ride on the final transcript itself, no second model, no intent
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

# Default phrases per command. Every list is user-remappable via
# [commands] in config.toml; see Grammar below.
DEFAULT_PHRASES: dict[str, list[str]] = {
    "send": [
        "send it off", "and send it", "send it", "send that", "send",
        "go ahead", "submit",
    ],
    "type": ["type", "write"],
    "discard": [
        "close", "cancel", "scratch that", "never mind", "nevermind",
        "discard",
    ],
    # "pause" reads better than "sleep" now that the microphone is open by
    # default: what it stops is the listening, and it comes back on a keypress.
    "sleep": ["stop listening", "go to sleep", "pause bol", "pause", "sleep"],
    "interrupt": ["interrupt", "stop claude", "stop stop"],
    "repeat": ["say that again", "repeat that", "what did you say"],
    "clean": ["clean it up", "clean that up", "clean up", "fix it up",
              "tidy it up"],
}

_STANDALONE_ACTIONS = {
    "discard": Action.DISCARD,
    "sleep": Action.SLEEP,
    "interrupt": Action.INTERRUPT,
    "repeat": Action.REPEAT,
}


def _norm(text: str) -> str:
    return _PUNCT.sub("", text.strip().lower()).strip()


class Grammar:
    """Command matcher built from (possibly user-remapped) phrase lists."""

    def __init__(self, overrides: dict[str, list[str]] | None = None) -> None:
        phrases = dict(DEFAULT_PHRASES)
        for key, value in (overrides or {}).items():
            if key in phrases and isinstance(value, list) and value:
                phrases[key] = [str(p).lower() for p in value]
        # Longer phrases must match before their prefixes ("send it" > "send").
        self._send = sorted(phrases["send"], key=len, reverse=True)
        self._clean = sorted(phrases["clean"], key=len, reverse=True)
        self._type_prefix = re.compile(
            r"^(?:" + "|".join(re.escape(p) for p in phrases["type"]) + r")\s+",
            re.IGNORECASE,
        )
        self._standalone: dict[str, Action] = {}
        for key, action in _STANDALONE_ACTIONS.items():
            for phrase in phrases[key]:
                self._standalone[_norm(phrase)] = action

    def parse(self, transcript: str) -> Parsed:
        raw = transcript.strip()
        if not raw:
            return Parsed(Action.DICTATE, "")

        norm = _norm(raw)
        if norm in self._standalone:
            return Parsed(self._standalone[norm])

        # Leading type-prefix → literal text, no submit, never cleaned (the
        # user asked for these exact characters).
        m = self._type_prefix.match(raw)
        if m:
            payload = raw[m.end():].strip()
            parsed_tail = self._strip_trailing_send(payload)
            if parsed_tail is not None:
                return Parsed(Action.SEND, parsed_tail)
            return Parsed(Action.TYPE, payload)

        stripped = self._strip_trailing_send(raw)
        if stripped is not None:
            payload, clean = self._strip_trailing_clean(stripped)
            return Parsed(Action.SEND, payload, clean=clean)

        payload, clean = self._strip_trailing_clean(raw)
        return Parsed(Action.DICTATE, payload, clean=clean)

    def _strip_trailing_clean(self, raw: str) -> tuple[str, bool]:
        """Strip a trailing clean phrase (with an optional joining "and");
        returns (payload, clean_requested)."""
        norm = _norm(raw)
        for phrase in self._clean:
            for suffix in (" and " + phrase, " " + phrase):
                if norm.endswith(suffix) or norm == suffix.strip():
                    n_words = len(suffix.split())
                    words = raw.split()
                    payload = " ".join(words[:-n_words]).rstrip(" ,.;")
                    return payload, True
        return raw, False

    def _strip_trailing_send(self, raw: str) -> str | None:
        """If the utterance ends with a send phrase, return the text before."""
        norm = _norm(raw)
        for phrase in self._send:
            if norm == phrase:
                return ""
            if norm.endswith(" " + phrase):
                # Cut the phrase off by word count, preserving the payload's
                # original casing/punctuation.
                n_words = len(phrase.split())
                words = raw.split()
                payload = " ".join(words[:-n_words]).strip()
                return payload.rstrip(" ,.;") if payload else ""
        return None


_DEFAULT_GRAMMAR = Grammar()


def parse_transcript(transcript: str) -> Parsed:
    """Parse with the default phrase set."""
    return _DEFAULT_GRAMMAR.parse(transcript)
