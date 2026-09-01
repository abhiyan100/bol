import pytest

from bol.config import Config
from bol.hooks.events import StopEvent, ToolUse, TurnTracker
from bol.speak.base import SPEECH_LIMIT, clamp_speech
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
    assert "What should Claude do next" in text


@pytest.mark.asyncio
async def test_markdown_stripped_from_spoken_text():
    msg = "**Done!** I edited `auth.py`.\n```py\nprint(1)\n```\nTests pass."
    text = await TemplateSummarizer(Config()).summarize(_event(message=msg))
    assert "`" not in text and "*" not in text


@pytest.mark.asyncio
async def test_punctuationless_dump_is_capped():
    # A markdown table has no sentence punctuation, so the sentence split
    # used to hand the whole 20KB back and `say` read it for ~9 minutes.
    row = "| src/handlers/user_profile_controller.py | 412 | ok |\n"
    msg = "| file | lines | status |\n|---|---|---|\n" + row * 400
    assert len(msg) > 20_000
    text = await TemplateSummarizer(Config()).summarize(_event(message=msg))
    assert len(text) < 600
    assert "and more" in text


@pytest.mark.asyncio
async def test_short_message_is_not_truncated():
    text = await TemplateSummarizer(Config()).summarize(_event(message="All good."))
    assert "and more" not in text
    assert "All good." in text


def test_clamp_speech_is_the_last_line_of_defence():
    assert clamp_speech("short") == "short"
    long = "word " * 500
    out = clamp_speech(long)
    assert len(out) <= SPEECH_LIMIT + len(", and more")
    assert out.endswith(", and more")


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
