"""What the pill says, as pure functions.

Deliberately free of AppKit: the state table, the wording, and the truncation
are the parts that decide whether the pill tells the truth, and they have to
be testable on any machine, in any process, without a window server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Every state the daemon is allowed to put on screen. Anything else is
# dropped rather than shown, so a typo in a caller can never wedge the pill
# on a line nobody can explain.
STATES = (
    "idle",
    "awake",
    "listening",
    "finalizing",
    "sending",
    "thinking",
    "permission",
    "speaking",
    "error",
)

# The 8 px dot next to the text. An empty name means no dot at all: "Sent"
# and a permission question are statements, not activity.
COLORS = {
    "idle": "",
    # Wake mode's awake window: nothing is being recorded, but the next thing
    # said needs no wake phrase, and that is worth one dot on screen.
    "awake": "green",
    "listening": "green",
    "finalizing": "blue",
    "sending": "",
    "thinking": "blue",
    "permission": "",
    "speaking": "amber",
    "error": "red",
}

# Transient states take themselves off the screen after this long, so the
# daemon never has to hold a timer open just to retract a word.
HOLD_S = {"sending": 1.0, "error": 3.0}

# Used when the caller sends a state with no text of its own.
DEFAULTS = {
    # One word, because the pill hides itself when a state has nothing to
    # say: an awake window drawn as a bare dot would be an unexplained light.
    "awake": "Awake",
    "listening": "Listening",
    "finalizing": "Finalizing",
    "sending": "Sent",
    "thinking": "Thinking",
    "speaking": "Bol is speaking",
    "error": "Something went wrong",
}

PERMISSION_HINT = "tap and say yes or no"
SEPARATOR = " · "
TRUNCATION = "..."
# Hard ceiling on the label. The panel narrows it again to whatever 60% of
# the active screen can actually hold.
MAX_CHARS = 120


@dataclass(frozen=True)
class Update:
    """One line of truth about what Bol is doing right now."""

    state: str
    text: str = ""
    detail: str = ""


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_line(line: str) -> Update | None:
    """Read one JSON line from the daemon, or return None if it is not usable.

    The child is fed by a pipe it does not control, so a blank line, a half
    written object, or an unknown state has to be ignored rather than raise
    on the reader thread and take the pill down with it.
    """
    if not isinstance(line, str):
        return None
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("state") not in STATES:
        return None
    return Update(
        state=data["state"],
        text=_clean(data.get("text")),
        detail=_clean(data.get("detail")),
    )


def truncate_middle(text: str, limit: int) -> str:
    """Shorten to `limit` characters, keeping both ends.

    The ends are what identify a line: "Bash: pytest -q ..." alone loses the
    command, and a path truncated at the tail loses the file name.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(TRUNCATION):
        return text[:limit]
    keep = limit - len(TRUNCATION)
    head = (keep + 1) // 2
    tail = keep - head
    return text[:head] + TRUNCATION + (text[-tail:] if tail else "")


def label_for(state: str, text: str = "", detail: str = "") -> str:
    """The single line the pill shows. Empty means the pill is hidden."""
    if state not in STATES or state == "idle":
        return ""
    text = _clean(text)
    detail = _clean(detail)
    if state == "permission":
        head = f"Claude asks: {text}" if text else "Claude needs your permission"
        return truncate_middle(f"{head}{SEPARATOR}{PERMISSION_HINT}", MAX_CHARS)
    if state == "listening" and detail:
        # Live dictation: text is what the decoder has committed and detail is
        # the tail it is still deciding on. One sentence, so they join with a
        # space rather than the field separator, and the words replace the
        # "Listening" default instead of being appended to it.
        return truncate_middle(f"{text} {detail}".strip(), MAX_CHARS)
    head = text or DEFAULTS.get(state, "")
    if detail:
        head = f"{head}{SEPARATOR}{detail}" if head else detail
    return truncate_middle(head, MAX_CHARS)


def draft_span(state: str, label: str, detail: str) -> int:
    """How many characters at the end of the label are unconfirmed draft.

    The panel dims exactly this many. Zero whenever the answer is not certain:
    the label may have been truncated in the middle, and dimming the wrong run
    of a sentence is worse than dimming none of it.
    """
    detail = _clean(detail)
    if state != "listening" or not detail or not isinstance(label, str):
        return 0
    return len(detail) if label.endswith(detail) else 0


def color_for(state: str) -> str:
    """Dot colour name, or "" for the states that show no dot."""
    return COLORS.get(state, "")


def hold_for(state: str) -> float:
    """Seconds this state stays up on its own. 0 means until told otherwise."""
    return HOLD_S.get(state, 0.0)


def render(update: Update) -> tuple[str, str]:
    """(label, colour) for one update."""
    return label_for(update.state, update.text, update.detail), color_for(update.state)


def tool_line(name: str, detail: str = "") -> str:
    """One line for the tool Claude just ran, e.g. "Bash: pytest -q"."""
    name = _clean(name) or "Tool"
    detail = _clean(detail)
    return f"{name}: {detail}" if detail else name
