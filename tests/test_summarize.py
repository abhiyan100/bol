import pytest

from bol.config import Config
from bol.hooks.events import StopEvent, ToolUse, TurnTracker
from bol.summarize.template import TemplateSummarizer, describe_tools


def _event(tools=None, message="All done."):
    return StopEvent(
        session_id="s",
        prompt_id="p",
        last_assistant_message=message,
        transcript_path="",
        cwd="",
        tools=tools or [],
    )


@pytest.mark.asyncio
async def test_template_mentions_edits_and_failures():
    tools = [
        ToolUse("Edit", "/repo/auth.py", True),
        ToolUse("Bash", "pytest", False),
    ]
    cfg = Config()
    cfg.summarizer.user_name = "Abhiyan"
    text = await TemplateSummarizer(cfg).summarize(_event(tools))
    assert "auth.py" in text
    assert "failed" in text
    assert "Abhiyan" in text


@pytest.mark.asyncio
async def test_template_handles_empty_turn():
    text = await TemplateSummarizer(Config()).summarize(_event(message=""))
    assert text
    assert "What next" in text


@pytest.mark.asyncio
async def test_markdown_stripped_from_spoken_text():
    msg = "**Done!** I edited `auth.py`.\n```py\nprint(1)\n```\nTests pass."
    text = await TemplateSummarizer(Config()).summarize(_event(message=msg))
    assert "`" not in text and "*" not in text


def test_describe_tools_many_files():
    tools = [ToolUse("Edit", f"/r/f{i}.py", True) for i in range(5)]
    assert "5 files" in describe_tools(tools)


def test_tracker_accumulates_by_prompt():
    tracker = TurnTracker()
    tracker.record_tool(
        {"prompt_id": "p1", "tool_name": "Edit",
         "tool_input": {"file_path": "/a.py"}, "tool_response": {}}
    )
    tracker.record_tool(
        {"prompt_id": "p2", "tool_name": "Bash",
         "tool_input": {"command": "ls"}, "tool_response": {}}
    )
    event = tracker.finish_turn(
        {"prompt_id": "p1", "session_id": "s", "last_assistant_message": "hi"}
    )
    assert len(event.tools) == 1
    assert event.tools[0].tool_name == "Edit"


def test_tracker_flags_error_responses():
    tracker = TurnTracker()
    tracker.record_tool(
        {"prompt_id": "p", "tool_name": "Bash",
         "tool_input": {"command": "false"},
         "tool_response": {"is_error": True}}
    )
    event = tracker.finish_turn({"prompt_id": "p"})
    assert event.tools[0].ok is False
