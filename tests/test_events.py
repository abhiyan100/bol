"""TurnTracker bounds.

Every PostToolUse is buffered until its Stop arrives. Turns that never stop
(interrupted, crashed, or from a Claude Code session Bol is not narrating)
would otherwise pin their tool lists for the life of the daemon.
"""

from bol.hooks.events import MAX_TOOLS_PER_TURN, MAX_TURNS, TurnTracker


def _tool(prompt_id, name="Edit", path="/a.py"):
    return {
        "prompt_id": prompt_id,
        "session_id": "s",
        "tool_name": name,
        "tool_input": {"file_path": path},
        "tool_response": {},
    }


def test_oldest_turn_is_evicted():
    tracker = TurnTracker()
    for i in range(MAX_TURNS + 8):
        tracker.record_tool(_tool(f"p{i}"))
    assert len(tracker._tools) == MAX_TURNS
    # The first eight turns are gone; the newest is intact.
    assert tracker.finish_turn({"prompt_id": "p0"}).tools == []
    assert len(tracker.finish_turn({"prompt_id": f"p{MAX_TURNS + 7}"}).tools) == 1


def test_a_live_turn_is_not_evicted_by_newer_ones():
    tracker = TurnTracker()
    tracker.record_tool(_tool("long-running"))
    for i in range(MAX_TURNS - 1):
        tracker.record_tool(_tool(f"p{i}"))
        tracker.record_tool(_tool("long-running"))  # keeps it fresh
    assert len(tracker.finish_turn({"prompt_id": "long-running"}).tools) == MAX_TURNS


def test_tools_per_turn_are_capped_to_a_rolling_window():
    tracker = TurnTracker()
    for i in range(MAX_TOOLS_PER_TURN + 5):
        tracker.record_tool(_tool("p", path=f"/f{i}.py"))
    tools = tracker.finish_turn({"prompt_id": "p"}).tools
    assert len(tools) == MAX_TOOLS_PER_TURN
    # The window keeps the most recent calls, not the first ones.
    assert tools[-1].detail == f"/f{MAX_TOOLS_PER_TURN + 4}.py"
    assert tools[0].detail == "/f5.py"


def test_finish_turn_pops_the_key():
    tracker = TurnTracker()
    tracker.record_tool(_tool("p"))
    tracker.finish_turn({"prompt_id": "p"})
    assert tracker._tools == {}
