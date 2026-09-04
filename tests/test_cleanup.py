import pytest

from bol.cleanup import (
    SESSION_WORDS_MAX,
    apply_vocabulary,
    clean_transcript,
    deterministic_clean,
    paste_words,
    remember_pasted,
    session_words,
    title_words,
)


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


# ----------------------------------------------------- this session's own words


@pytest.mark.parametrize(
    "raw, expected",
    [
        # The four words a transcriber that has never seen "bol" reaches for.
        ("the bull model is fast", "the Bol model is fast"),
        ("the ball model is fast", "the Bol model is fast"),
        ("the bowl model is fast", "the Bol model is fast"),
        ("the bole model is fast", "the Bol model is fast"),
        # Capitalization, from a word that is already the right one.
        ("bol pastes it", "Bol pastes it"),
        # One word rather than two, and neither the same length nor the same
        # skeleton: the product's own name earns a spelling of its own.
        ("I was testing babel today", "I was testing Bol today"),
        # A different first letter, or a different skeleton, and it is a
        # different word: all three tests have to pass.
        ("pull the branch first", "pull the branch first"),
        ("boot the machine", "boot the machine"),
        ("the bugle sounded", "the bugle sounded"),
    ],
)
def test_a_word_that_sounds_like_this_sessions_word_is_spelled_it(raw, expected):
    assert apply_vocabulary(raw, session=["Bol"]) == expected


def test_a_three_letter_session_word_only_owns_its_listed_lookalikes():
    # The sound-alike rule needs four letters to be trusted; a short name
    # would otherwise own every short word starting with its letter. So in a
    # Bol session "bill" stays "bill", while the listed lookalikes become Bol.
    assert apply_vocabulary("the bill is paid", session=["Bol"]) == "the bill is paid"
    assert apply_vocabulary("I was testing bull today", session=["Bol"]) == "I was testing Bol today"
    assert apply_vocabulary("open bowl and run it", session=["Bol"]) == "open Bol and run it"
    # And the [vocabulary] pass never does any of this on its own.
    assert apply_vocabulary("the bull is paid", ["Bol"]) == "the bull is paid"


def test_the_session_word_has_to_be_in_the_set_for_its_own_spellings():
    # "babel" is Bol only in a session that has already seen "Bol" written
    # down. Somebody else's project is untouched.
    assert apply_vocabulary("run babel over it", session=["Parakeet"]) == (
        "run babel over it"
    )
    assert apply_vocabulary("run babel over it") == "run babel over it"


def test_a_session_word_overrides_the_stoplist_a_vocabulary_entry_obeys():
    # "call" is an English word, so the [vocabulary] pass will never touch it.
    # In a session whose own word is Cole, that word is a name.
    assert apply_vocabulary("call the parser", ["Cole"]) == "call the parser"
    assert apply_vocabulary("call the parser", session=["Cole"]) == "Cole the parser"


@pytest.mark.parametrize(
    "raw",
    [
        # Being talked about, not being used.
        'say "bowl" out loud',
        "the word 'bowl' means the crockery",
        "`bowl` is not the name",
        # A path, a file name and a flag are already spelled the way they are.
        "open bowl.py next",
        "look in src/bowl/main.py",
        "pass --bowl-first to it",
        "rename auth.bowl to something else",
    ],
)
def test_a_quoted_or_code_token_is_left_alone(raw):
    assert apply_vocabulary(raw, session=["Bol"]) == raw


@pytest.mark.parametrize(
    "title, expected",
    [
        # The project is the capitalized word; the shell, the agent and the
        # pane size are in every title whatever the project is.
        ("Bol - claude - 180x48", ["Bol"]),
        ("auth.py - Bol - Visual Studio Code", ["Bol"]),
        ("zsh - Bol - 120x40", ["Bol"]),
        ("bol - claude - main", []),          # lower case says nothing
        ("Bol - Parakeet - claude", ["Bol", "Parakeet"]),
        # No vowels, the way a tool's name is written.
        ("nvm - Bol", ["nvm", "Bol"]),
        ("", []),
        (None, []),
    ],
)
def test_the_front_window_title_gives_up_the_project_name(title, expected):
    assert title_words(title) == expected


def test_a_paste_teaches_the_names_in_it():
    assert paste_words("Refactor the Parakeet loader in stt/parakeet.py") == [
        # The capital comes first: written out is better evidence of how the
        # user spells it than the same word inside a path.
        "Parakeet",
        "parakeet.py",
    ]
    # A capital that only opens a sentence is punctuation, and a word the
    # language already owns is not a name.
    assert paste_words("Fix the login test. Then run it.") == []


def _name(index: int) -> str:
    """A distinct capitalized name per index, with no digits in it: a digit
    beside a word is how the pass recognises code."""
    return f"Zulu{chr(65 + index // 26)}{chr(97 + index % 26)}"


def test_learning_from_pastes_is_bounded_and_keeps_the_newest():
    learned: dict = {}
    for index in range(SESSION_WORDS_MAX + 5):
        remember_pasted(learned, f"ask {_name(index)} about it")

    assert len(learned) == SESSION_WORDS_MAX
    assert _name(0) not in learned                       # the oldest went
    assert _name(SESSION_WORDS_MAX + 4) in learned


def test_a_word_learned_from_a_paste_spells_the_next_dictation():
    learned: dict = {}
    remember_pasted(learned, "Refactor the Kokoro loader")
    words = session_words([], title_words("Bol - claude"), learned)

    assert words == ["Bol", "Kokoro"]
    assert apply_vocabulary("the kokora loader is slow", session=words) == (
        "the Kokoro loader is slow"
    )


def test_one_set_of_words_from_three_sources():
    # In the order they are trusted, spelled the way the first source wrote
    # it, and every word once.
    words = session_words(
        ["Kokoro"], title_words("Bol - claude"), {"kokoro": None, "auth.py": None}
    )
    assert words == ["Kokoro", "Bol", "auth.py"]


@pytest.mark.asyncio
async def test_clean_transcript_takes_the_session_words_too():
    cleaner = FakeCleaner()
    out = await clean_transcript(
        FakeEngine(error=True),
        "um the bowl daemon pastes it",
        2.0,
        use_llm=False,
        cleaner=cleaner,
        session=["Bol"],
    )
    # Spelled before the model sees it, so there is nothing for it to undo.
    assert cleaner.calls == ["The Bol daemon pastes it"]
    assert out == "The Bol daemon pastes it [tuned]"


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("I'm testing the hayball command", "I'm testing the hey Bol command"),
        ("it still says hey ball here", "it still says hey Bol here"),
        ("say hey bowl and talk", "say hey Bol and talk"),
        ("Hey bull, add a test", "hey Bol, add a test"),
        # One word on its own is a real word until the session says otherwise.
        ("put it in a bowl", "put it in a bowl"),
        ("the babel config", "the babel config"),
    ],
)
def test_the_wake_phrase_is_spelled_right_in_any_session(spoken, expected):
    assert apply_vocabulary(spoken) == expected
