import pytest

from bol.cleanup import apply_vocabulary, clean_transcript, deterministic_clean


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


# ------------------------------------------------------------------ vocabulary


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("open claude code and look at auth.py", "open Claude Code and look at auth.py"),
        ("cloud code keeps crashing", "Claude Code keeps crashing"),
        ("Claude Code is already right", "Claude Code is already right"),
        ("push it to git hub", "push it to GitHub"),
        ("check github actions", "check GitHub actions"),
        ("ask codex to look at it", "ask Codex to look at it"),
        ("run pie test on the login module", "run pytest on the login module"),
        ("install it with you vee", "install it with uv"),
        ("the o auth flow is broken", "the OAuth flow is broken"),
        ("the oauth flow is broken", "the OAuth flow is broken"),
        ("write it to a jason file", "write it to a JSON file"),
        ("keep the jason format", "keep the JSON format"),
    ],
)
def test_tool_names_are_spelled_the_way_the_tools_spell_them(raw, expected):
    assert apply_vocabulary(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # A wrong correction costs more than a missed one, and every line here
        # is a word that means something else far more often than it means a
        # tool. None of them may be touched.
        "the clause is confusing, rewrite it",
        "move the cursor to the end of the line",
        "jason said the build is green",
        "the cloud bill went up again",
        "close the file and open the next one",
        "read the codebase before you change it",
    ],
)
def test_a_word_that_is_not_a_tool_name_is_left_alone(raw):
    assert apply_vocabulary(raw, ["Claude", "Codex", "Kokoro"]) == raw


@pytest.mark.parametrize(
    "raw, words, expected",
    [
        # Capitalization, from an exact match.
        ("ask abhiyan about it", ["Abhiyan"], "ask Abhiyan about it"),
        # One edit: a syllable the transcriber heard wrong.
        ("ask abhiyen about it", ["Abhiyan"], "ask Abhiyan about it"),
        # Two edits, allowed because the entry is eight characters or more.
        ("the paraqueet model", ["Parakeet"], "the Parakeet model"),
        # A pair of tokens: the transcriber split one name in two.
        ("ask abhi yan about it", ["Abhiyan"], "ask Abhiyan about it"),
        # Several entries, the nearest wins.
        ("kokoru and parakeet", ["Kokoro", "Parakeet"], "Kokoro and Parakeet"),
        # Punctuation and spacing around the token survive untouched.
        ("(abhiyen), please", ["Abhiyan"], "(Abhiyan), please"),
    ],
)
def test_vocabulary_words_are_spelled_the_users_way(raw, words, expected):
    assert apply_vocabulary(raw, words) == expected


@pytest.mark.parametrize(
    "raw, words",
    [
        # Under five characters an entry only ever fixes capitalization,
        # because every short word is one edit from half the language.
        ("a bowl of soup", ["Bol"]),
        ("the bot replied", ["Bol"]),
        # Common English words are never rewritten into someone's name.
        ("the code is done", ["Codex"]),
        ("run the tests in the cloud", ["Claude"]),
        ("that is the point", ["Poudel"]),
        # Nothing close enough.
        ("refactor the login module", ["Abhiyan", "Parakeet"]),
    ],
)
def test_vocabulary_never_invents_a_correction(raw, words):
    assert apply_vocabulary(raw, words) == raw


def test_an_empty_vocabulary_still_spells_tool_names():
    assert apply_vocabulary("open claude code", []) == "open Claude Code"
    assert apply_vocabulary("open claude code", None) == "open Claude Code"


@pytest.mark.asyncio
async def test_clean_transcript_applies_the_vocabulary_before_the_model():
    cleaner = FakeCleaner()
    out = await clean_transcript(
        FakeEngine(error=True),
        "um open cloud code and ask abhiyen",
        2.0,
        use_llm=False,
        cleaner=cleaner,
        vocabulary=["Abhiyan"],
    )
    # The model is handed already-spelled text, so it has nothing to undo.
    assert cleaner.calls == ["Open Claude Code and ask Abhiyan"]
    assert out == "Open Claude Code and ask Abhiyan [tuned]"


@pytest.mark.asyncio
async def test_vocabulary_applies_with_no_model_at_all():
    out = await clean_transcript(
        FakeEngine(error=True),
        "um push it to git hub please",
        2.0,
        use_llm=False,
        vocabulary=[],
    )
    assert out == "Push it to GitHub please"
