"""The persona summarizer must talk about THIS turn, never recite its examples."""

import pytest
from test_summarize import _event

from bol.config import Config
from bol.hooks.events import ToolUse
from bol.summarize.llm import LlmSummarizer, grounded


class FakeEngine:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _cfg(provider="local"):
    cfg = Config()
    cfg.llm.provider = provider
    return cfg


def _turn():
    return _event(
        tools=[ToolUse("Edit", "parser.py", True), ToolUse("Bash", "pytest -q", True)],
        message="Fixed the split on trailing commas in parser.py; all three tests pass now.",
    )


@pytest.mark.asyncio
async def test_a_grounded_reply_is_spoken_as_is():
    engine = FakeEngine("Claude's done. It edited parser.py and ran the tests. It says the trailing comma split is fixed and all three tests pass.")
    text = await LlmSummarizer(_cfg(), engine).summarize(_turn())
    assert "parser.py" in text and "tests pass" in text


@pytest.mark.asyncio
async def test_a_recited_example_is_thrown_away_for_the_template():
    # The exact failure a user heard: the model answered with the few-shot
    # Docker example instead of the turn in front of it.
    engine = FakeEngine(
        "The build crashed during npm install, but the issue is still there. "
        "Claude needs to check the peer dependency conflict with react 19. Let's fix that."
    )
    text = await LlmSummarizer(_cfg(), engine).summarize(_turn())
    assert "react" not in text.lower() and "npm" not in text.lower()
    assert "parser.py" in text  # the template, built from the real tool log


@pytest.mark.asyncio
async def test_the_local_model_gets_no_examples_to_recite():
    engine = FakeEngine("Claude's done. It edited parser.py; the tests pass.")
    await LlmSummarizer(_cfg("local"), engine).summarize(_turn())
    assert engine.calls[0]["history"] == []


@pytest.mark.asyncio
async def test_an_api_model_still_gets_the_examples():
    engine = FakeEngine("Claude's done. It edited parser.py; the tests pass.")
    await LlmSummarizer(_cfg("api"), engine).summarize(_turn())
    assert len(engine.calls[0]["history"]) == 4


def test_grounded_needs_a_shared_content_word():
    turn = _turn()
    assert grounded("Claude wrapped that up: parser.py is fixed and the tests pass.", turn)
    assert not grounded("All sorted, nothing else to report today.", turn)
    assert not grounded("", turn)
    # A telltale from the examples is fine only when the turn itself said it.
    docker = _event(tools=[ToolUse("Bash", "docker build .", False)], message="The Docker build fails at npm install.")
    assert grounded("Claude hit a problem: the Docker build fails at npm install.", docker)
