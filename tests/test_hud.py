"""The pill: what it says (render) and how the daemon talks to it (Hud).

Nothing here opens a window. The AppKit half is one thin file whose only job
is to draw what these functions decide, and the client half is tested against
a fake process because the failure that matters is a child that died.
"""

import asyncio
import json

import pytest

from bol.hud import Hud
from bol.hud.render import (
    COLORS,
    MAX_CHARS,
    STATES,
    Update,
    color_for,
    draft_span,
    hold_for,
    label_for,
    parse_line,
    render,
    tool_line,
    truncate_middle,
)

# ------------------------------------------------------------------- render


def test_every_state_says_something():
    for state in STATES:
        label = label_for(state)
        if state == "idle":
            assert label == ""  # idle is the pill being gone
        else:
            assert label, state


def test_the_dot_colours_are_the_documented_ones():
    assert color_for("listening") == "green"
    assert color_for("finalizing") == "blue"
    assert color_for("thinking") == "blue"
    assert color_for("speaking") == "amber"
    assert color_for("error") == "red"
    # A statement, not activity: no dot.
    assert color_for("sending") == ""
    assert color_for("permission") == ""
    assert color_for("idle") == ""


def test_an_unknown_state_shows_nothing():
    assert label_for("dancing") == ""
    assert color_for("dancing") == ""
    assert set(COLORS) == set(STATES)


def test_the_default_wording():
    assert label_for("listening") == "Listening"
    assert label_for("finalizing") == "Finalizing"
    assert label_for("sending") == "Sent"
    assert label_for("thinking") == "Thinking"


def test_thinking_carries_the_tool_line():
    assert label_for("thinking", "Thinking", "Bash: pytest -q") == (
        "Thinking · Bash: pytest -q"
    )


def test_listening_shows_the_committed_words_then_the_draft():
    # One sentence being spoken, not two fields: the draft continues the
    # committed text, so it joins with a space and not the field separator.
    assert label_for("listening", "add a login", "test for the") == (
        "add a login test for the"
    )
    assert label_for("listening", "", "add a") == "add a"
    assert label_for("listening", "add a login test") == "add a login test"
    assert label_for("listening") == "Listening"


def test_the_draft_is_the_part_the_panel_dims():
    label = label_for("listening", "add a login", "test for the")
    assert draft_span("listening", label, "test for the") == len("test for the")
    # Nothing is dimmed on any other state, whatever the detail says.
    assert draft_span("thinking", "Thinking · Bash: pytest", "Bash: pytest") == 0
    assert draft_span("listening", label, "") == 0


def test_a_draft_the_panel_had_to_cut_is_not_dimmed():
    # truncate_middle keeps both ends, so a long draft comes back missing its
    # middle. Dimming the wrong run of a sentence is worse than dimming none.
    committed = "the quick brown fox " * 6
    draft = "jumped over the lazy dog and kept going " * 3
    label = label_for("listening", committed, draft)
    assert len(label) == MAX_CHARS
    assert draft_span("listening", label, draft) == 0


def test_speaking_shows_bols_own_sentence():
    assert label_for("speaking", "Tests pass, all thirty of them.") == (
        "Tests pass, all thirty of them."
    )


def test_permission_asks_for_a_yes_or_a_no():
    label = label_for("permission", "Claude wants to run rm -rf build.")
    assert label.startswith("Claude asks: Claude wants to run rm -rf build.")
    assert label.endswith("tap and say yes or no")


def test_permission_without_a_message_still_makes_sense():
    assert label_for("permission") == (
        "Claude needs your permission · tap and say yes or no"
    )


def test_error_shows_the_remedy_verbatim():
    remedy = "Lost the microphone. Check your input device."
    assert label_for("error", remedy) == remedy
    assert color_for("error") == "red"


def test_transient_states_take_themselves_down():
    assert hold_for("sending") == 1.0
    assert hold_for("error") == 3.0
    # Everything else stays until the daemon says otherwise.
    assert hold_for("listening") == 0.0
    assert hold_for("thinking") == 0.0


def test_render_returns_the_label_and_the_colour():
    assert render(Update("listening")) == ("Listening", "green")
    assert render(Update("idle")) == ("", "")


# ---------------------------------------------------------------- truncation


def test_short_text_is_left_alone():
    assert truncate_middle("Listening", 20) == "Listening"
    assert truncate_middle("Listening", 9) == "Listening"


def test_truncation_keeps_both_ends():
    out = truncate_middle("abcdefghijklmnop", 9)
    assert len(out) == 9
    assert out.startswith("abc") and out.endswith("nop")
    assert "..." in out


def test_truncation_handles_silly_limits():
    assert truncate_middle("abcdef", 0) == ""
    assert truncate_middle("abcdef", 2) == "ab"
    assert truncate_middle("abcdef", 3) == "abc"


def test_a_very_long_label_is_capped():
    label = label_for("speaking", "word " * 200)
    assert len(label) == MAX_CHARS


def test_tool_line_reads_like_a_command():
    assert tool_line("Bash", "pytest -q") == "Bash: pytest -q"
    assert tool_line("Read", "") == "Read"
    assert tool_line("", "") == "Tool"


# --------------------------------------------------------------- parse_line


def test_a_full_update_is_read():
    line = json.dumps({"state": "thinking", "text": "Thinking", "detail": "Bash: ls"})
    assert parse_line(line) == Update("thinking", "Thinking", "Bash: ls")


def test_a_bare_state_is_enough():
    assert parse_line('{"state": "idle"}') == Update("idle", "", "")


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   \n",
        "not json at all",
        '{"state": "listening"',  # a half-written line
        "[1, 2, 3]",
        '"listening"',
        "{}",
        '{"state": "dancing"}',
        '{"state": 7}',
        "null",
    ],
)
def test_junk_is_ignored_rather_than_raised(line):
    # The child reads a pipe it does not control; one bad line must never be
    # the thing that takes the pill down.
    assert parse_line(line) is None


def test_non_string_fields_are_dropped():
    assert parse_line('{"state": "error", "text": 7, "detail": null}') == Update(
        "error", "", ""
    )


# ------------------------------------------------------------------- client


class FakeStdin:
    def __init__(self, broken: bool = False, drain_broken: bool = False) -> None:
        self.lines = []
        self.broken = broken
        self.drain_broken = drain_broken
        self.closed = False
        self.drains = 0

    def write(self, data: bytes) -> None:
        if self.broken:
            raise BrokenPipeError("the child is gone")
        self.lines.append(json.loads(data.decode("utf-8")))

    async def drain(self) -> None:
        self.drains += 1
        if self.drain_broken:
            raise ConnectionResetError("the child is gone")

    def close(self) -> None:
        self.closed = True


class FakeProc:
    def __init__(self, broken: bool = False, drain_broken: bool = False) -> None:
        self.stdin = FakeStdin(broken, drain_broken)
        self.killed = False

    async def wait(self) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


class Spawner:
    """Hands out scripted child processes and counts the attempts."""

    def __init__(self, *procs) -> None:
        self.procs = list(procs) or [FakeProc()]
        self.count = 0

    async def __call__(self):
        self.count += 1
        return self.procs[min(self.count - 1, len(self.procs) - 1)]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def _hud(*procs, clock=None):
    spawner = Spawner(*procs)
    hud = Hud(spawn=spawner, clock=clock or Clock())
    await hud.start()
    return hud, spawner


async def test_set_writes_one_json_line_per_update():
    hud, spawner = await _hud()
    hud.set("listening", "Listening")
    hud.set("thinking", "Thinking", "Bash: pytest -q")
    await hud.idle()

    assert spawner.count == 1
    assert spawner.procs[0].stdin.lines == [
        {"state": "listening", "text": "Listening", "detail": ""},
        {"state": "thinking", "text": "Thinking", "detail": "Bash: pytest -q"},
    ]
    await hud.stop()


async def test_a_disabled_pill_never_starts_a_child():
    spawner = Spawner()
    hud = Hud(enabled=False, spawn=spawner)
    await hud.start()
    hud.set("listening")
    await hud.stop()
    assert spawner.count == 0


async def test_an_unknown_state_is_never_sent():
    hud, spawner = await _hud()
    hud.set("dancing", "nope")
    await hud.idle()
    assert spawner.procs[0].stdin.lines == []
    await hud.stop()


async def test_set_before_start_is_inert():
    spawner = Spawner()
    hud = Hud(spawn=spawner)
    hud.set("listening")  # must not raise, must not spawn anything
    await hud.idle()
    assert spawner.count == 0


async def test_a_broken_pipe_never_reaches_the_caller():
    dead, fresh = FakeProc(broken=True), FakeProc()
    hud, spawner = await _hud(dead, fresh)

    hud.set("listening", "Listening")  # writes into a dead pipe
    await hud.idle()

    # The line was resent to the child that replaced the dead one.
    assert spawner.count == 2
    assert fresh.stdin.lines == [
        {"state": "listening", "text": "Listening", "detail": ""}
    ]
    await hud.stop()


async def test_a_broken_drain_is_swallowed():
    hud, spawner = await _hud(FakeProc(drain_broken=True))
    hud.set("listening")
    await hud.idle()  # the drain raises in the background and is dropped
    assert spawner.procs[0].stdin.drains == 1
    await hud.stop()


async def test_it_respawns_at_most_once_a_minute():
    clock = Clock()
    dead_one, dead_two, fresh = FakeProc(True), FakeProc(True), FakeProc()
    hud, spawner = await _hud(dead_one, dead_two, fresh, clock=clock)

    hud.set("listening")
    await hud.idle()
    assert spawner.count == 2  # the first failure brings the child back

    hud.set("listening")
    await hud.idle()
    assert spawner.count == 2  # the second, seconds later, does not

    clock.now += 61.0
    hud.set("listening")
    await hud.idle()
    assert spawner.count == 3  # a minute on, it tries once more
    assert fresh.stdin.lines == [{"state": "listening", "text": "", "detail": ""}]
    await hud.stop()


async def test_stop_closes_stdin():
    hud, spawner = await _hud()
    await hud.stop()
    assert spawner.procs[0].stdin.closed is True


async def test_set_after_stop_is_ignored():
    hud, spawner = await _hud()
    await hud.stop()
    hud.set("listening")
    await hud.idle()
    assert spawner.procs[0].stdin.lines == []


async def test_a_child_that_never_starts_is_not_fatal(caplog):
    async def explode():
        raise OSError("no such executable")

    hud = Hud(spawn=explode)
    with caplog.at_level("INFO", logger="bol.hud"):
        await hud.start()
    hud.set("listening")
    await hud.stop()
    assert "pill unavailable, continuing without it" in caplog.text


async def test_a_hung_child_is_killed_on_stop():
    class Hung(FakeProc):
        async def wait(self):
            await asyncio.sleep(30)

    hung = Hung()
    hud = Hud(spawn=Spawner(hung))
    await hud.start()
    hud._close_timeout = 0.01
    await asyncio.wait_for(hud.stop(), timeout=5)
    assert hung.stdin.closed is True
    assert hung.killed is True


# ------------------------------------------------------------ per-line hold

# "Sent" is one word and one second. A line the user is meant to read (the
# hint after a paste auto-send held back) needs longer than the state table's
# default, and only that line does.


def test_a_caller_can_outstay_the_state_default():
    assert hold_for("sending", 2.5) == 2.5
    assert hold_for("sending") == 1.0  # every other "Sent" is unchanged
    assert hold_for("listening", 2.5) == 2.5


def test_a_hold_of_zero_leaves_the_state_default_alone():
    assert hold_for("sending", 0.0) == 1.0
    assert hold_for("thinking", 0.0) == 0.0


def test_the_hold_survives_the_pipe():
    line = json.dumps({"state": "sending", "text": "Pasted.", "hold": 2.5})
    update = parse_line(line)
    assert update == Update("sending", "Pasted.", "", 2.5)
    assert hold_for(update.state, update.hold) == 2.5


@pytest.mark.parametrize("value", ["soon", None, -1, 0, 10 ** 6, [2.5]])
def test_an_unusable_hold_costs_the_override_not_the_line(value):
    # Same rule as every other field: a nonsense value must never be the
    # thing that stops the pill from saying what Bol is doing. A hold with no
    # ceiling would wedge a transient line on screen, which is worse.
    update = parse_line(json.dumps({"state": "sending", "text": "Sent", "hold": value}))
    assert update == Update("sending", "Sent", "", 0.0)


async def test_set_sends_a_hold_only_when_there_is_one():
    hud, spawner = await _hud()
    hud.set("sending", "Sent")
    hud.set("sending", "Pasted.", hold=2.5)
    await hud.idle()

    assert spawner.procs[0].stdin.lines == [
        {"state": "sending", "text": "Sent", "detail": ""},
        {"state": "sending", "text": "Pasted.", "detail": "", "hold": 2.5},
    ]
    await hud.stop()
