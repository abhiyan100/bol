"""Config tests: the shipped default file has to parse, and it has to say the
same thing the dataclasses do. A default that drifts from the commented TOML
is how people end up debugging a setting they never actually changed.
"""

import tomllib

import pytest

from bol.config import (
    DEFAULT_CONFIG_TOML,
    Config,
    load_config,
    removed_keys,
    validate_config,
)


def test_the_defaults():
    cfg = Config()
    assert cfg.hotkey.key == "alt_r"
    assert cfg.audio.pre_roll_ms == 300
    assert cfg.audio.warm_s == 120
    # One-way unless asked: nothing is spoken, and no summarizer, voice or
    # LLM server is loaded.
    assert cfg.talk_back is False


def test_the_default_file_parses():
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    assert data["talk_back"] is False
    assert data["hotkey"]["key"] == "alt_r"
    assert data["audio"]["pre_roll_ms"] == 300
    assert data["audio"]["warm_s"] == 120


def test_the_default_file_matches_the_dataclasses():
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    cfg = Config()
    assert data["talk_back"] == cfg.talk_back
    for key, value in data["hotkey"].items():
        assert getattr(cfg.hotkey, key) == value, key
    for key, value in data["audio"].items():
        assert getattr(cfg.audio, key) == value, key
    for key, value in data["ui"].items():
        assert getattr(cfg.ui, key) == value, key
    for key, value in data["stt"].items():
        assert getattr(cfg.stt, key) == value, key


def test_the_default_file_has_no_key_the_dataclasses_dropped(tmp_path):
    # The file is the documentation. A key nobody reads any more sitting in
    # it is how people end up configuring something that does nothing.
    path = tmp_path / "config.toml"
    path.write_text(DEFAULT_CONFIG_TOML)
    assert removed_keys(path) == []


def test_load_config_applies_the_new_fields(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "talk_back = true\n"
        '[hotkey]\nkey = "f13"\n'
        "[audio]\npre_roll_ms = 500\nwarm_s = 30\n"
    )

    cfg = load_config(path)

    assert cfg.talk_back is True
    assert cfg.hotkey.key == "f13"
    assert cfg.audio.pre_roll_ms == 500
    assert cfg.audio.warm_s == 30


def test_ui_defaults():
    cfg = Config()
    assert cfg.ui.pill is True
    assert cfg.ui.sounds is True
    assert cfg.ui.position == "top"


def test_an_old_config_file_still_loads(tmp_path):
    # Keys this Bol no longer reads are ignored, not fatal: an old file has
    # to keep starting Bol, and removed_keys() is what says what changed.
    path = tmp_path / "config.toml"
    path.write_text(
        "sound_cues = false\nhands_free = true\n"
        '[hotkey]\nmode = "auto"\nsubmit = "always"\n[ui]\npill = false\n'
    )

    cfg = load_config(path)

    assert cfg.ui.sounds is True   # the old alias decides nothing now
    assert cfg.ui.pill is False
    validate_config(cfg)


def test_load_config_reads_the_ui_section(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[ui]\npill = false\nposition = "bottom"\n')
    cfg = load_config(path)
    assert cfg.ui.pill is False
    assert cfg.ui.position == "bottom"


def test_load_config_without_a_file_is_all_defaults(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.hotkey.key == "alt_r"
    assert cfg.talk_back is False


# ------------------------------------------------------------------ validation


def test_validate_rejects_a_quoted_talk_back():
    # A non-empty string is truthy, so talk_back = "false" would load a
    # summarizer, a voice and an LLM server for someone who said not to.
    cfg = Config()
    cfg.talk_back = "false"

    with pytest.raises(ValueError) as err:
        validate_config(cfg)

    message = str(err.value)
    assert "talk_back" in message
    assert "true or false" in message


def test_validate_rejects_an_unknown_pill_position():
    cfg = Config()
    cfg.ui.position = "middle"

    with pytest.raises(ValueError) as err:
        validate_config(cfg)

    message = str(err.value)
    assert "middle" in message
    assert "top" in message and "bottom" in message


def test_the_daemon_validates_at_startup():
    # cmd_run prints a ValueError cleanly, so a typo costs one line instead of
    # a setting that silently means the opposite of what it says.
    from bol.daemon import Daemon

    cfg = Config()
    cfg.ui.position = "typo"
    with pytest.raises(ValueError):
        Daemon(cfg, text_mode=True)


def test_live_words_are_on_with_a_context_that_actually_streams():
    cfg = Config()
    assert cfg.stt.live is True
    # A right context longer than the recording, so no step ever finalizes and
    # the pill shows a re-decode instead of an accumulation of frames taken off
    # parakeet-mlx's window seam. See bol/stt/parakeet.py for the measurements.
    assert cfg.stt.stream_context == [256, 256]
    assert cfg.stt.stream_chunk_ms == 640


def test_load_config_applies_the_streaming_fields(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[stt]\nlive = false\nstream_context = [256, 8]\nstream_chunk_ms = 160\n"
    )
    cfg = load_config(path)
    assert cfg.stt.live is False
    assert cfg.stt.stream_context == [256, 8]
    assert cfg.stt.stream_chunk_ms == 160


# ------------------------------------------------------- v0.3 phase 3 fields


def test_silero_is_the_default_gate():
    cfg = Config()
    assert cfg.audio.vad == "silero"
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    assert data["audio"]["vad"] == cfg.audio.vad


def test_validate_accepts_both_gates():
    for mode in ("silero", "energy"):
        cfg = Config()
        cfg.audio.vad = mode
        validate_config(cfg)


def test_validate_rejects_an_unknown_gate():
    cfg = Config()
    cfg.audio.vad = "webrtc"

    with pytest.raises(ValueError) as err:
        validate_config(cfg)

    message = str(err.value)
    assert "webrtc" in message
    assert "silero" in message and "energy" in message


def test_the_vocabulary_starts_empty():
    cfg = Config()
    assert cfg.vocabulary.words == []
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    assert data["vocabulary"]["words"] == cfg.vocabulary.words


def test_load_config_reads_the_vocabulary(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[audio]\nvad = "energy"\n[vocabulary]\nwords = ["Abhiyan", "Bol"]\n')

    cfg = load_config(path)

    assert cfg.audio.vad == "energy"
    assert cfg.vocabulary.words == ["Abhiyan", "Bol"]


def test_two_configs_do_not_share_a_vocabulary_list():
    # A mutable default that leaks between Config() instances would put one
    # user's words into another run's transcripts.
    first, second = Config(), Config()
    first.vocabulary.words.append("Abhiyan")
    assert second.vocabulary.words == []


# ------------------------------------------------- keys that no longer exist


def _hint(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    hints = removed_keys(path)
    assert len(hints) == 1, hints
    return hints[0]


def test_the_old_hotkey_mode_gets_one_line(tmp_path):
    hint = _hint(tmp_path, '[hotkey]\nmode = "auto"\n')
    assert "[hotkey] mode" in hint
    assert "hold the key" in hint


def test_the_old_submit_setting_gets_one_line(tmp_path):
    hint = _hint(tmp_path, '[hotkey]\nsubmit = "always"\n')
    assert "[hotkey] submit" in hint
    assert "send it" in hint


def test_hands_free_gets_one_line(tmp_path):
    hint = _hint(tmp_path, "hands_free = true\n")
    assert "hands_free" in hint
    assert "awake" in hint


def test_the_old_bridge_mode_gets_one_line(tmp_path):
    hint = _hint(tmp_path, '[bridge]\nmode = "tmux"\n')
    assert "[bridge] mode" in hint


def test_sound_cues_gets_one_line(tmp_path):
    hint = _hint(tmp_path, "sound_cues = false\n")
    assert "sound_cues" in hint
    assert "[ui] sounds" in hint


def test_a_current_config_says_nothing(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('talk_back = true\n[hotkey]\nkey = "alt_r"\n[ui]\nsounds = false\n')
    assert removed_keys(path) == []


def test_every_removed_key_is_named_once(tmp_path):
    # One line per key that is actually in the file, and the path with it, so
    # the reader knows which config is being talked about.
    path = tmp_path / "config.toml"
    path.write_text(
        "hands_free = true\nsound_cues = false\n"
        '[hotkey]\nmode = "auto"\ntap_ms = 400\nsubmit = "auto"\n'
        "auto_send_min_words = 3\n"
        '[bridge]\nmode = "auto"\npane = "%3"\n'
    )

    hints = removed_keys(path)

    assert len(hints) == 8
    assert all(str(path) in hint for hint in hints)


def test_a_missing_or_broken_config_is_not_a_crash(tmp_path):
    assert removed_keys(tmp_path / "missing.toml") == []
    broken = tmp_path / "config.toml"
    broken.write_text("[hotkey\nmode = ")
    assert removed_keys(broken) == []


# ------------------------------------------------------- where Bol may paste


def test_dictation_goes_anywhere_by_default():
    cfg = Config()
    assert cfg.bridge.anywhere is True


def test_anywhere_is_in_the_default_file_and_agrees_with_the_dataclass():
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    cfg = Config()
    for key, value in data["bridge"].items():
        assert getattr(cfg.bridge, key) == value, key
    assert data["bridge"]["anywhere"] is True


def test_the_binding_window_is_in_the_default_file_and_agrees_with_the_dataclass():
    # Hooks are user-scoped, so which session Bol narrates is a setting
    # people will reach for. It has to be in the documented file, spelled
    # the same as the dataclass reads it.
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    cfg = Config()
    for key, value in data["server"].items():
        assert getattr(cfg.server, key) == value, key
    assert data["server"]["bind_window_s"] == cfg.server.bind_window_s == 20.0
    assert cfg.server.follow == "first"


def test_load_config_reads_the_binding_window(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[server]\nbind_window_s = 5.0\n")

    cfg = load_config(path)

    assert cfg.server.bind_window_s == 5.0


def test_talk_back_is_the_one_switch_between_the_two_modes(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("talk_back = true\n")

    cfg = load_config(path)

    assert cfg.talk_back is True
    validate_config(cfg)


def test_terminal_only_is_one_line(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[bridge]\nanywhere = false\n")

    cfg = load_config(path)

    assert cfg.bridge.anywhere is False
    validate_config(cfg)


def test_validate_rejects_a_quoted_anywhere():
    # A non-empty string is truthy, so "false" would mean the opposite of
    # what it says, and it would mean it about where Enter may land.
    cfg = Config()
    cfg.bridge.anywhere = "false"

    with pytest.raises(ValueError) as err:
        validate_config(cfg)

    message = str(err.value)
    assert "[bridge] anywhere" in message
    assert "true or false" in message


# ---------------------------------------------- what a fresh install now gets
#
# Three defaults changed together, and all three came from the same feedback:
# two seconds instead of three, a separate window for "start talking", and
# cleanup on every dictation because raw dictation is not professional.


def test_the_pause_is_two_and_a_half_seconds_and_the_speak_window_is_five():
    cfg = Config()
    assert cfg.wake.pause_ms == 2500
    assert cfg.wake.speak_window_ms == 5000
    # Off: with the awake window open, room noise kept reopening the mic.
    assert cfg.wake.awake_s == 0.0


def test_cleanup_runs_on_every_dictation_by_default():
    cfg = Config()
    assert cfg.cleanup.mode == "always"
    assert cfg.cleanup.model == "abhiyan10/bol-cleanup-350m-4bit"


def test_setup_has_not_been_answered_until_the_wizard_answers_it():
    # "auto" and [] are different answers: one is nobody having chosen, the
    # other is somebody choosing no coding agent at all.
    assert Config().setup.agents == "auto"


def test_the_default_file_matches_the_dataclasses_everywhere():
    # The one that catches a default changed in exactly one of the two places.
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    cfg = Config()
    for section in ("wake", "cleanup", "setup", "tts", "llm", "summarizer", "server"):
        for key, value in data.get(section, {}).items():
            assert getattr(getattr(cfg, section), key) == value, f"{section}.{key}"


def test_load_config_reads_the_setup_section(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[setup]\nagents = ["codex"]\n')
    assert load_config(path).setup.agents == ["codex"]

    path.write_text("[setup]\nagents = []\n")
    assert load_config(path).setup.agents == []


def test_validate_rejects_a_speak_window_of_zero():
    cfg = Config()
    cfg.wake.speak_window_ms = 0

    with pytest.raises(ValueError) as err:
        validate_config(cfg)

    assert "speak_window_ms" in str(err.value)
    assert "above 0" in str(err.value)


# ------------------------------------------ writing the setup wizard's answers


def test_the_template_keeps_its_comments_when_values_are_substituted():
    from bol.config import render_default_config

    out = render_default_config({
        (None, "talk_back"): True,
        ("cleanup", "mode"): "off",
        ("audio", "input_device"): "Shure MV7",
        ("setup", "agents"): ["claude", "codex"],
    })

    data = tomllib.loads(out)
    assert data["talk_back"] is True
    assert data["cleanup"]["mode"] == "off"
    assert data["audio"]["input_device"] == "Shure MV7"
    assert data["setup"]["agents"] == ["claude", "codex"]
    # A key the template ships commented out is uncommented, and every line
    # keeps the documentation that is the reason the file exists.
    assert "# mic name substring or index" in out
    assert "# the on-screen pill that shows what Bol is doing" in out
    assert removed_keys.__module__  # sanity: the module imported cleanly


def test_substituting_nothing_is_the_template_itself():
    from bol.config import render_default_config

    assert render_default_config({}) == DEFAULT_CONFIG_TOML


def test_writing_a_new_config_keeps_the_comments(tmp_path):
    from bol.config import write_config_values

    path, rewritten = write_config_values(
        {(None, "talk_back"): True}, tmp_path / "config.toml"
    )

    assert rewritten is False
    assert "# Bol configuration" in path.read_text()
    assert load_config(path).talk_back is True


def test_writing_over_an_existing_config_changes_only_those_keys(tmp_path):
    from bol.config import write_config_values

    path = tmp_path / "config.toml"
    path.write_text(
        '# a comment nobody promised to keep\n'
        'talk_back = false\n[hotkey]\nkey = "f13"\n'
        '[vocabulary]\nwords = ["Abhiyan"]\n'
    )

    written, rewritten = write_config_values(
        {(None, "talk_back"): True, ("setup", "agents"): []}, path
    )

    assert rewritten is True
    cfg = load_config(written)
    assert cfg.talk_back is True
    assert cfg.setup.agents == []
    assert cfg.hotkey.key == "f13"           # untouched
    assert cfg.vocabulary.words == ["Abhiyan"]
    validate_config(cfg)
