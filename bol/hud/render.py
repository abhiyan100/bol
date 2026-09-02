"""What the pill shows, as pure functions.

Deliberately free of AppKit: the state table, the five-dot patterns, the
wording and the truncation are the parts that decide whether the pill tells
the truth, and they have to be testable on any machine, in any process,
without a window server.

The pill is a fixed-width capsule: a mark on the left, five dots on the
right, and no text unless [ui] text asks for it. So the dots are the whole
message, and which dots are alight for which state is data here rather than
a branch in the drawing code.
"""

from __future__ import annotations

import json
import math
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

# The capsule always shows this many dots. Some are alight and some are not;
# the count never changes, so the pill never resizes and never jumps.
DOT_COUNT = 5

# White over the dark capsule. A dot that is off is still drawn: five dots
# always sit there, and the message is which of them are alight.
DIM_ALPHA = 0.22
LIT_ALPHA = 0.95
# The bottom of the breathing swell, per the design: 40 to 95 percent.
BREATHE_LOW = 0.40
# How lit the dot behind a travelling head is. Enough to read as a tail,
# not enough to be mistaken for a second head.
TAIL = 0.35

# Every way the dots can move. The panel owns the drawing and the clock;
# this owns which motion belongs to which state.
MOTIONS = (
    "hidden",   # no pill at all
    "dim",      # pill up, nothing alight
    "level",    # a live meter: n dots from the left, n from the mic level
    "sweep",    # one dot travelling left to right, over and over
    "bounce",   # one dot running left, right, left
    "breathe",  # all five swelling together
    "all",      # all five alight, still
    "blink",    # all five, on and off
    "steady",   # all five alight, still, in a colour that means stop
)


@dataclass(frozen=True)
class Dots:
    """What the five dots do while one state is on screen.

    motion is the pattern, color names the tint ("" is plain white), period
    is one cycle of an animated motion in seconds, alpha is how bright a lit
    dot gets, and icon is the mark's own alpha.
    """

    motion: str
    color: str = ""
    period: float = 0.0
    alpha: float = LIT_ALPHA
    icon: float = 1.0


HIDDEN = Dots("hidden", icon=0.0)

# The state table. Colour is spent sparingly on purpose: white for everything
# Bol is doing under its own steam, and a colour only where the user has to
# do something (answer a prompt, read an error, say "send it").
DOTS = {
    "idle": HIDDEN,
    # Wake mode's awake window: nothing is being recorded, but the next thing
    # said needs no wake phrase. The pill is up and the mark is dimmed, which
    # is the difference between "ready" and "working".
    "awake": Dots("dim", icon=0.6),
    "listening": Dots("level"),
    "finalizing": Dots("sweep", period=0.4),
    "thinking": Dots("bounce", period=0.9),
    "speaking": Dots("breathe", period=1.2),
    "sending": Dots("all"),
    "permission": Dots("blink", color="amber", period=1.6),
    "error": Dots("steady", color="red"),
}

# The hint after a paste, which is after every paste: nothing is ever sent by
# itself. It rides in on "sending" with a hold of its own (see hold_for):
# "Sent" is one word that takes itself down, and a line the user has to act on
# is neither. Dim blue rather than white, so a pill that is waiting on the
# user never looks like one that is finished.
PASTED = Dots("all", color="blue", alpha=0.60)

# Transient states take themselves off the screen after this long, so the
# daemon never has to hold a timer open just to retract a word. A caller can
# override its own line (see Update.hold): "Sent" is one word and one second,
# and a sentence the user has to read is neither.
HOLD_S = {"sending": 1.0, "error": 3.0}

# Used when the caller sends a state with no text of its own. Only reachable
# with [ui] text = true; the dots say all of this on their own.
DEFAULTS = {
    "awake": "Awake",
    "listening": "Listening",
    "finalizing": "Finalizing",
    "sending": "Sent",
    "thinking": "Thinking",
    "speaking": "Bol is speaking",
    "error": "Something went wrong",
}

PERMISSION_HINT = "say yes or no"
SEPARATOR = " · "
TRUNCATION = "..."
# Hard ceiling on the label. The panel narrows it again to whatever 60% of
# the active screen can actually hold.
MAX_CHARS = 120
# Ceiling on a per-line hold. A transient line that outstays this is a pill
# stuck on a stale sentence, which is the failure the whole state table
# exists to avoid.
MAX_HOLD_S = 30.0


@dataclass(frozen=True)
class Update:
    """One line of truth about what Bol is doing right now."""

    state: str
    text: str = ""
    detail: str = ""
    # Seconds this line stays up, overriding HOLD_S for this line only.
    # 0 means "whatever the state says".
    hold: float = 0.0
    # Microphone level, 0 to 1, for the listening meter. Absent on every
    # other state and on every daemon too old to send it.
    level: float = 0.0


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_line(line: str) -> Update | None:
    """Read one JSON line from the daemon, or return None if it is not usable.

    The child is fed by a pipe it does not control, so a blank line, a half
    written object, or an unknown state has to be ignored rather than raise
    on the reader thread and take the pill down with it. Unknown keys are
    ignored rather than refused, so a newer daemon can talk to this child.
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
        hold=_hold(data.get("hold")),
        level=clamp_level(data.get("level")),
    )


def _hold(value: object) -> float:
    """A per-line hold from the pipe, or 0 for anything unusable.

    Same rule as every other field the child reads: a nonsense value costs
    the override, never the line.
    """
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return seconds if 0.0 < seconds < MAX_HOLD_S else 0.0


def clamp_level(value: object) -> float:
    """A microphone level from the pipe as 0..1. Anything unusable is 0.

    A missing level is the ordinary case, not an error: every state but
    listening leaves it out, and so does any daemon that predates the meter.
    """
    try:
        level = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(level):
        return 0.0
    return min(1.0, max(0.0, level))


# ------------------------------------------------------------------- the dots


def dots_for(state: str, hold: float = 0.0) -> Dots:
    """Which dot pattern one update draws.

    A "sending" that carries its own hold is the paste hint rather than a
    finished send, and it is the one thing on the pill the user is meant to
    answer, so it gets its own colour. Deriving it here keeps the wire
    protocol exactly what it was.
    """
    if state == "sending" and hold and hold > 0:
        return PASTED
    return DOTS.get(state, HIDDEN)


def listening_dots(level: float) -> int:
    """How many of the five dots a microphone level lights.

    At least one: a listening pill with every dot dark reads as a pill that
    has stopped listening, which is the one thing it must never say while the
    microphone is open. Rounded half up by hand rather than with round(),
    whose banker's rule lights a different number of dots either side of a
    half.
    """
    return max(1, min(DOT_COUNT, int(clamp_level(level) * DOT_COUNT + 0.5)))


def _mix(low: float, high: float, amount: float) -> float:
    """low at 0, high at 1, and exactly those at the ends: a dot the pattern
    calls fully lit has to be the same number as a dot that is simply lit."""
    if amount <= 0.0:
        return low
    if amount >= 1.0:
        return high
    return low + (high - low) * amount


def phase(elapsed: float, period: float) -> float:
    """Where in one cycle a motion is, as 0..1. 0 whenever there is no cycle."""
    if period <= 0:
        return 0.0
    try:
        seconds = float(elapsed)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(seconds) or seconds <= 0:
        return 0.0
    return (seconds % period) / period


def dot_alphas(dots: Dots, elapsed: float = 0.0, level: float = 0.0) -> tuple[float, ...]:
    """The alpha of each of the five dots at this moment of the motion.

    Pure on purpose: the panel supplies a clock and a level and draws the
    numbers, so every pattern in the design can be checked here rather than
    by staring at a window.
    """
    lit, dim = dots.alpha, DIM_ALPHA
    motion = dots.motion
    if motion in ("hidden", "dim"):
        return (dim,) * DOT_COUNT
    if motion in ("all", "steady"):
        return (lit,) * DOT_COUNT
    if motion == "level":
        count = listening_dots(level)
        return tuple(lit if i < count else dim for i in range(DOT_COUNT))
    where = phase(elapsed, dots.period)
    if motion == "sweep":
        # One dot travelling left to right, with a tail on the dot behind it
        # so the eye reads a direction rather than a dot teleporting.
        head = min(DOT_COUNT - 1, int(where * DOT_COUNT))
        return tuple(
            lit if i == head else (_mix(dim, lit, TAIL) if i == head - 1 else dim)
            for i in range(DOT_COUNT)
        )
    if motion == "bounce":
        # Left to right and back again inside one period. The head is snapped
        # to a dot rather than left between two: an interpolated head washed
        # the whole row out to one grey and read as no motion at all.
        travel = where * 2.0
        span = DOT_COUNT - 1
        position = travel * span if travel <= 1.0 else (2.0 - travel) * span
        head = int(position + 0.5)
        return tuple(
            lit if i == head else (_mix(dim, lit, TAIL) if abs(i - head) == 1 else dim)
            for i in range(DOT_COUNT)
        )
    if motion == "breathe":
        swell = 0.5 - 0.5 * math.cos(2.0 * math.pi * where)
        return (_mix(BREATHE_LOW, lit, swell),) * DOT_COUNT
    if motion == "blink":
        return ((lit if where < 0.5 else dim),) * DOT_COUNT
    return (dim,) * DOT_COUNT


def animated(dots: Dots) -> bool:
    """Whether this pattern needs a repaint clock at all.

    The still states (a level meter the daemon drives, an error, a paste
    hint) run no timer: the pill is decoration, and decoration that wakes the
    CPU thirty times a second while nothing moves is not.
    """
    return dots.motion in ("sweep", "bounce", "breathe", "blink") and dots.period > 0


# ------------------------------------------------------------------- the text


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
    """The line the pill shows beside the dots, when [ui] text is on."""
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


def hold_for(state: str, hold: float = 0.0) -> float:
    """Seconds this line stays up on its own. 0 means until told otherwise.

    A caller's own hold wins when it has one: the state table sizes the
    default word for each state, and a line with more to say than "Sent"
    needs longer than "Sent" does.
    """
    if hold and hold > 0:
        return float(hold)
    return HOLD_S.get(state, 0.0)


def render(update: Update) -> tuple[str, Dots]:
    """(label, dot pattern) for one update."""
    return (
        label_for(update.state, update.text, update.detail),
        dots_for(update.state, update.hold),
    )


def tool_line(name: str, detail: str = "") -> str:
    """One line for the tool Claude just ran, e.g. "Bash: pytest -q"."""
    name = _clean(name) or "Tool"
    detail = _clean(detail)
    return f"{name}: {detail}" if detail else name
