"""The pill: what it shows (render) and how the daemon talks to it (Hud).

Nothing here opens a window. The AppKit half is one file whose only job is to
draw what these functions decide, so the state table, the five dot patterns
and the level meter are all checked here, on any machine, with no window
server. The client half is tested against a fake process because the failure
that matters is a child that died.
"""

import asyncio
import json

import pytest

from bol.hud import Hud
from bol.hud.render import (
    BREATHE_LOW,
    DIM_ALPHA,
    DOT_COUNT,
    DOTS,
    LIT_ALPHA,
    MAX_CHARS,
    MOTIONS,
    PASTED,
    STATES,
    Update,
    animated,
    clamp_level,
    dot_alphas,
    dots_for,
    draft_span,
    hold_for,
    label_for,
    listening_dots,
    parse_line,
    phase,
    render,
    tool_line,
    truncate_middle,
)

# ---------------------------------------------------------------- state table


def test_every_state_has_a_dot_pattern():
    assert set(DOTS) == set(STATES)
    for state, dots in DOTS.items():
        assert dots.motion in MOTIONS, state
        assert 0.0 <= dots.alpha <= 1.0, state
        assert 0.0 <= dots.icon <= 1.0, state


def test_idle_is_the_pill_being_gone():
    assert dots_for("idle").motion == "hidden"
    assert label_for("idle") == ""


def test_an_unknown_state_shows_nothing():
    assert dots_for("dancing").motion == "hidden"
    assert label_for("dancing") == ""


def test_the_documented_motion_for_each_state():
    assert dots_for("listening").motion == "level"
    assert dots_for("finalizing").motion == "sweep"
    assert dots_for("thinking").motion == "bounce"
    assert dots_for("speaking").motion == "breathe"
    assert dots_for("sending").motion == "all"
    assert dots_for("permission").motion == "blink"
    assert dots_for("error").motion == "steady"


def test_colour_is_spent_only_where_the_user_has_to_do_something():
    # White for everything Bol is doing under its own steam.
    for state in ("listening", "finalizing", "thinking", "speaking", "sending"):
        assert dots_for(state).color == "", state
    assert dots_for("permission").color == "amber"
    assert dots_for("error").color == "red"


def test_there_is_no_ready_state_on_the_screen():
    # The awake pill is gone. It stood for a window in which speech needed no
    # trigger word, that window is off by default now, and a capsule that
    # sits there between sentences is one nobody stops seeing.
    assert "awake" not in STATES
    assert dots_for("awake") is dots_for("dancing")
    assert label_for("awake") == ""


def test_a_paste_hint_is_a_sending_that_carries_its_own_hold():
    # The one line on the pill the user is meant to answer, so it gets its
    # own colour. Same wire protocol as it always had: state plus a hold.
    assert dots_for("sending", 2.5) == PASTED
    assert PASTED.color == "blue"
    assert PASTED.alpha < LIT_ALPHA
    # An ordinary "Sent" is unchanged.
    assert dots_for("sending").color == ""
    assert dots_for("sending", 0.0).color == ""


def test_only_the_moving_patterns_ask_for_a_clock():
    for state in ("finalizing", "thinking", "speaking", "permission"):
        assert animated(dots_for(state)) is True, state
    # A meter the daemon drives, an error, a paste hint and an idle pill all
    # sit still: none of them should wake the CPU thirty times a second.
    for state in ("idle", "listening", "sending", "error"):
        assert animated(dots_for(state)) is False, state
    assert animated(PASTED) is False


# ------------------------------------------------------------------ the dots


def test_the_level_meter_lights_dots_from_the_left():
    assert listening_dots(0.0) == 1
    assert listening_dots(0.3) == 2
    assert listening_dots(0.5) == 3
    assert listening_dots(0.7) == 4
    assert listening_dots(1.0) == DOT_COUNT


def test_a_listening_pill_never_goes_completely_dark():
    # Every dot dark reads as a pill that has stopped listening, which is the
    # one thing it must not say while the microphone is open.
    assert listening_dots(0.0) == 1
    assert listening_dots(-5.0) == 1
    assert listening_dots(float("nan")) == 1
    assert listening_dots("loud") == 1
    assert listening_dots(9.0) == DOT_COUNT


def test_the_level_pattern_is_lit_then_dim():
    alphas = dot_alphas(dots_for("listening"), level=0.5)
    assert alphas == (LIT_ALPHA, LIT_ALPHA, LIT_ALPHA, DIM_ALPHA, DIM_ALPHA)
    assert dot_alphas(dots_for("listening"), level=1.0) == (LIT_ALPHA,) * DOT_COUNT


def test_every_pattern_answers_for_all_five_dots():
    for state in STATES:
        for elapsed in (0.0, 0.13, 0.7, 3.4):
            alphas = dot_alphas(dots_for(state), elapsed, 0.4)
            assert len(alphas) == DOT_COUNT, state
            assert all(0.0 <= a <= 1.0 for a in alphas), (state, alphas)


def test_the_sweep_runs_left_to_right():
    dots = dots_for("finalizing")  # 400 ms, five dots, 80 ms each
    heads = [dot_alphas(dots, e).index(LIT_ALPHA) for e in (0.0, 0.1, 0.18, 0.26, 0.39)]
    assert heads == [0, 1, 2, 3, 4]
    # And starts over rather than stopping on the last dot: finalizing lasts
    # as long as the decoder does.
    assert dot_alphas(dots, 0.41).index(LIT_ALPHA) == 0


def test_the_bounce_goes_out_and_comes_back():
    dots = dots_for("thinking")  # 900 ms there and back
    assert dot_alphas(dots, 0.0)[0] == LIT_ALPHA
    assert dot_alphas(dots, 0.45)[DOT_COUNT - 1] == LIT_ALPHA
    assert dot_alphas(dots, 0.9)[0] == LIT_ALPHA


def test_a_travelling_head_is_always_one_whole_dot():
    # An interpolated head that fell between two dots washed the whole row
    # out to one grey, and a row of identical dots reads as no motion.
    for state in ("thinking", "finalizing"):
        dots = dots_for(state)
        for step in range(24):
            alphas = dot_alphas(dots, step * dots.period / 24.0)
            assert alphas.count(LIT_ALPHA) == 1, (state, alphas)


def test_the_breath_swells_together():
    dots = dots_for("speaking")  # 1.2 s, 40 to 95 percent
    low = dot_alphas(dots, 0.0)
    high = dot_alphas(dots, 0.6)
    assert low == (BREATHE_LOW,) * DOT_COUNT
    assert high == (LIT_ALPHA,) * DOT_COUNT
    # All five move as one: a breath, not a meter.
    assert len(set(dot_alphas(dots, 0.3))) == 1


def test_the_permission_blink_is_on_and_off():
    dots = dots_for("permission")
    assert dot_alphas(dots, 0.0) == (LIT_ALPHA,) * DOT_COUNT
    assert dot_alphas(dots, 1.0) == (DIM_ALPHA,) * DOT_COUNT


def test_a_phase_is_where_in_the_cycle_we_are():
    assert phase(0.0, 0.4) == 0.0
    assert phase(0.2, 0.4) == 0.5
    assert phase(0.6, 0.4) == pytest.approx(0.5)
    # A still pattern has no cycle to be part of, and nonsense costs nothing.
    assert phase(1.0, 0.0) == 0.0
    assert phase(float("inf"), 0.4) == 0.0
    assert phase("soon", 0.4) == 0.0


# ---------------------------------------------------------------- the level


def test_a_level_is_clamped_to_the_meter():
    assert clamp_level(0.4) == 0.4
    assert clamp_level(3.0) == 1.0
    assert clamp_level(-1.0) == 0.0


@pytest.mark.parametrize("value", [None, "loud", float("nan"), float("inf"), [0.5]])
def test_an_unusable_level_is_silence_rather_than_a_dropped_line(value):
    assert clamp_level(value) == 0.0


def test_a_level_survives_the_pipe():
    update = parse_line(json.dumps({"state": "listening", "level": 0.72}))
    assert update == Update("listening", "", "", 0.0, 0.72)
    assert listening_dots(update.level) == 4


def test_a_line_with_no_level_is_the_ordinary_case():
    # Every state but listening leaves it out, and so does any daemon that
    # predates the meter.
    assert parse_line('{"state": "thinking"}').level == 0.0


def test_a_key_the_child_does_not_know_is_ignored_not_refused():
    line = json.dumps({"state": "listening", "level": 0.5, "shimmer": True})
    assert parse_line(line) == Update("listening", "", "", 0.0, 0.5)


# ------------------------------------------------------------------- render


def test_every_state_says_something_when_text_is_on():
    for state in STATES:
        label = label_for(state)
        if state == "idle":
            assert label == ""  # idle is the pill being gone
        else:
            assert label, state


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
    assert label.endswith("say yes or no")


def test_permission_without_a_message_still_makes_sense():
    assert label_for("permission") == (
        "Claude needs your permission · say yes or no"
    )


def test_error_shows_the_remedy_verbatim():
    remedy = "Lost the microphone. Check your input device."
    assert label_for("error", remedy) == remedy
    assert dots_for("error").color == "red"


def test_transient_states_take_themselves_down():
    assert hold_for("sending") == 1.0
    assert hold_for("error") == 3.0
    # Everything else stays until the daemon says otherwise.
    assert hold_for("listening") == 0.0
    assert hold_for("thinking") == 0.0


def test_render_returns_the_label_and_the_pattern():
    label, dots = render(Update("listening"))
    assert (label, dots.motion) == ("Listening", "level")
    label, dots = render(Update("idle"))
    assert (label, dots.motion) == ("", "hidden")


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


async def _hud(*procs, clock=None, **kwargs):
    spawner = Spawner(*procs)
    hud = Hud(spawn=spawner, clock=clock or Clock(), **kwargs)
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


async def test_the_text_option_reaches_the_child(monkeypatch, tmp_path):
    argv = []

    async def fake_exec(*args, **kwargs):
        argv.extend(args)
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    hud = Hud(text=True, log_path=tmp_path / "hud.log")
    await hud.start()
    await hud.stop()
    assert "--text" in argv
    assert "--position" in argv


async def test_the_pill_is_wordless_by_default(monkeypatch, tmp_path):
    argv = []

    async def fake_exec(*args, **kwargs):
        argv.extend(args)
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    hud = Hud(log_path=tmp_path / "hud.log")
    await hud.start()
    await hud.stop()
    assert "--text" not in argv


# ------------------------------------------------------------- level meter


async def test_a_level_refreshes_the_listening_line():
    hud, spawner = await _hud()
    hud.set("listening", "add a login", "test")
    hud.level(0.8)
    await hud.idle()

    # The words the live decoder put on that line survive the refresh: the
    # meter is another field on the same line, not a line of its own.
    assert spawner.procs[0].stdin.lines == [
        {"state": "listening", "text": "add a login", "detail": "test"},
        {
            "state": "listening",
            "text": "add a login",
            "detail": "test",
            "level": 0.8,
        },
    ]
    await hud.stop()


async def test_a_level_measured_after_listening_is_dropped():
    # The recorder keeps measuring for as long as the microphone is open, and
    # a level must never drag the pill back to a state it has left.
    hud, spawner = await _hud()
    hud.set("finalizing", "Finalizing")
    hud.level(0.9)
    await hud.idle()
    assert [line["state"] for line in spawner.procs[0].stdin.lines] == ["finalizing"]
    await hud.stop()


async def test_a_level_before_anything_is_listening_is_dropped():
    hud, spawner = await _hud()
    hud.level(0.5)
    await hud.idle()
    assert spawner.procs[0].stdin.lines == []
    await hud.stop()


async def test_the_meter_is_throttled():
    # Blocks arrive about thirty times a second and the meter goes out
    # fifteen; the rest are dropped rather than queued.
    clock = Clock()
    hud, spawner = await _hud(clock=clock)
    hud.set("listening", "Listening")
    for _ in range(5):
        hud.level(0.6)
        clock.now += 0.02
    await hud.idle()

    levels = [line.get("level") for line in spawner.procs[0].stdin.lines[1:]]
    assert levels == [0.6, 0.6]  # at 0.00 s and 0.08 s, not five times over
    await hud.stop()


async def test_silence_is_a_line_with_no_level_on_it():
    # 0 is the default at the other end, so it costs a key rather than
    # meaning something different.
    hud, spawner = await _hud()
    hud.set("listening", "Listening")
    hud.level(0.0)
    await hud.idle()
    assert "level" not in spawner.procs[0].stdin.lines[-1]
    await hud.stop()


async def test_a_disabled_pill_measures_nothing():
    spawner = Spawner()
    hud = Hud(enabled=False, spawn=spawner)
    await hud.start()
    hud.set("listening")
    hud.level(0.9)
    await hud.idle()
    assert spawner.count == 0


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
