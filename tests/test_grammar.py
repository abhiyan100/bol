from bol.grammar import Action, parse_transcript


def test_trailing_send_strips_command():
    p = parse_transcript("refactor the auth module send it")
    assert p.action is Action.SEND
    assert p.text == "refactor the auth module"


def test_trailing_send_with_punctuation():
    p = parse_transcript("Fix the login bug, send it.")
    assert p.action is Action.SEND
    assert p.text == "Fix the login bug"


def test_bare_send():
    p = parse_transcript("send")
    assert p.action is Action.SEND
    assert p.text == ""


def test_go_ahead_sends():
    p = parse_transcript("go ahead")
    assert p.action is Action.SEND
    assert p.text == ""


def test_type_prefix_no_submit():
    p = parse_transcript("type hello world")
    assert p.action is Action.TYPE
    assert p.text == "hello world"


def test_type_prefix_with_trailing_send_submits():
    p = parse_transcript("type git status and send it")
    assert p.action is Action.SEND
    assert p.text == "git status"


def test_close_discards():
    for phrase in ("close", "cancel", "scratch that", "never mind"):
        assert parse_transcript(phrase).action is Action.DISCARD


def test_sleep():
    assert parse_transcript("stop listening").action is Action.SLEEP


def test_interrupt():
    assert parse_transcript("stop claude").action is Action.INTERRUPT


def test_repeat():
    assert parse_transcript("say that again").action is Action.REPEAT


def test_plain_speech_is_dictation():
    p = parse_transcript("please look at the database schema")
    assert p.action is Action.DICTATE
    assert p.text == "please look at the database schema"


def test_send_inside_sentence_not_triggered():
    p = parse_transcript("add a send button to the form")
    assert p.action is Action.DICTATE


def test_empty():
    assert parse_transcript("   ").action is Action.DICTATE


def test_clean_and_send():
    p = parse_transcript("refactor the auth module clean it up and send it")
    assert p.action is Action.SEND
    assert p.text == "refactor the auth module"
    assert p.clean is True


def test_clean_without_send_dictates():
    p = parse_transcript("add a login test clean it up")
    assert p.action is Action.DICTATE
    assert p.text == "add a login test"
    assert p.clean is True


def test_send_without_clean_flag():
    p = parse_transcript("fix the bug send it")
    assert p.clean is False


def test_type_never_cleans():
    p = parse_transcript("type clean it up")
    assert p.action is Action.TYPE
    assert p.text == "clean it up"
    assert p.clean is False
