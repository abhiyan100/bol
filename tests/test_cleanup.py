import pytest

from bol.cleanup import clean_transcript, deterministic_clean


class FakeEngine:
    def __init__(self, reply=None, error=False):
        self.reply = reply
        self.error = error

    async def complete(self, system, user, max_tokens, deadline_s,
                       temperature=None, history=None):
        if self.error:
            raise RuntimeError("boom")
        return self.reply


def test_deterministic_fillers_and_doubles():
    out = deterministic_clean("um refactor the the auth module and uh run tests")
    assert out == "Refactor the auth module and run tests"


def test_deterministic_spoken_tokens():
    assert deterministic_clean("open auth dot py and run pytest dash dash verbose now") == \
        "Open auth.py and run pytest --verbose now"


def test_deterministic_preserves_negation():
    out = deterministic_clean("um delete the cache but don't touch login dot py")
    assert "don't touch login.py" in out


class FakeCleaner:
    """Stands in for the tuned local model."""

    def __init__(self):
        self.calls = []

    async def clean(self, text, deadline_s):
        self.calls.append(text)
        return text + " [tuned]"


@pytest.mark.asyncio
async def test_local_mode_never_calls_llm():
    engine = FakeEngine(error=True)  # would raise if called
    out = await clean_transcript(engine, "um fix the the login bug", 2.0, use_llm=False)
    assert out == "Fix the login bug"


@pytest.mark.asyncio
async def test_local_mode_uses_tuned_cleaner_not_llm():
    engine = FakeEngine(error=True)  # would raise if called
    cleaner = FakeCleaner()
    out = await clean_transcript(
        engine, "um fix the the login bug", 2.0, use_llm=False, cleaner=cleaner
    )
    assert cleaner.calls == ["Fix the login bug"]
    assert out == "Fix the login bug [tuned]"


@pytest.mark.asyncio
async def test_api_mode_prefers_llm_over_tuned_cleaner():
    engine = FakeEngine(reply="Fix the login bug.")
    cleaner = FakeCleaner()
    out = await clean_transcript(
        engine, "um fix the the login bug", 2.0, use_llm=True, cleaner=cleaner
    )
    assert cleaner.calls == []       # the small model stays out of the way
    assert out == "Fix the login bug."


@pytest.mark.asyncio
async def test_api_mode_applies_llm_fix():
    engine = FakeEngine(reply="Refactor the auth module and run the tests.")
    out = await clean_transcript(
        engine, "refactor auth module and run tests pls", 2.0, use_llm=True
    )
    assert out == "Refactor the auth module and run the tests."


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_deterministic():
    out = await clean_transcript(
        engine := FakeEngine(error=True), "um refactor the auth module", 2.0, use_llm=True
    )
    assert out == "Refactor the auth module"


@pytest.mark.asyncio
async def test_runaway_rewrite_rejected():
    engine = FakeEngine(reply="Certainly! " + "x" * 400)
    raw = "fix the login bug in auth.py"
    assert await clean_transcript(engine, raw, 2.0, use_llm=True) == deterministic_clean(raw)


@pytest.mark.asyncio
async def test_tiny_input_untouched():
    assert await clean_transcript(FakeEngine(reply="X"), "hi", 2.0, use_llm=True) == "hi"
