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
    validate_config,
)


def test_v03_defaults():
    cfg = Config()
    assert cfg.hotkey.mode == "auto"
    assert cfg.hotkey.tap_ms == 400
    assert cfg.hotkey.submit == "auto"
    assert cfg.hotkey.auto_send_min_words == 3
    assert cfg.audio.pre_roll_ms == 300
    assert cfg.audio.warm_s == 120
    # Reopening the mic unasked would send room noise now that dictation
    # submits itself.
    assert cfg.hands_free is False


def test_the_default_file_parses():
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    assert data["hands_free"] is False
    assert data["hotkey"]["mode"] == "auto"
    assert data["hotkey"]["tap_ms"] == 400
    assert data["hotkey"]["submit"] == "auto"
    assert data["hotkey"]["auto_send_min_words"] == 3
    assert data["audio"]["pre_roll_ms"] == 300
    assert data["audio"]["warm_s"] == 120


def test_the_default_file_matches_the_dataclasses():
    data = tomllib.loads(DEFAULT_CONFIG_TOML)
    cfg = Config()
    assert data["hands_free"] == cfg.hands_free
    for key, value in data["hotkey"].items():
        assert getattr(cfg.hotkey, key) == value, key
    for key, value in data["audio"].items():
        assert getattr(cfg.audio, key) == value, key
    for key, value in data["ui"].items():
        assert getattr(cfg.ui, key) == value, key
    for key, value in data["stt"].items():
        assert getattr(cfg.stt, key) == value, key


def test_load_config_applies_the_new_fields(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "hands_free = true\n"
        '[hotkey]\nmode = "push_to_talk"\ntap_ms = 250\nsubmit = "voice"\n'
        "auto_send_min_words = 5\n"
        "[audio]\npre_roll_ms = 500\nwarm_s = 30\n"
    )

    cfg = load_config(path)

    assert cfg.hands_free is True
    assert cfg.hotkey.mode == "push_to_talk"
    assert cfg.hotkey.tap_ms == 250
    assert cfg.hotkey.submit == "voice"
    assert cfg.hotkey.auto_send_min_words == 5
    assert cfg.audio.pre_roll_ms == 500
    assert cfg.audio.warm_s == 30


def test_ui_defaults():
    cfg = Config()
    assert cfg.ui.pill is True
    assert cfg.ui.sounds is True
    assert cfg.ui.position == "top"


def test_sound_cues_still_reads_and_writes_the_new_home():
    # Old code (and old muscle memory) says cfg.sound_cues; it has to keep
    # meaning exactly what [ui] sounds means.
    cfg = Config()
    assert cfg.sound_cues is True
    cfg.sound_cues = False
    assert cfg.ui.sounds is False
    cfg.ui.sounds = True
    assert cfg.sound_cues is True


def test_an_old_config_file_keeps_working(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("sound_cues = false\n")
    cfg = load_config(path)
    assert cfg.ui.sounds is False
    assert cfg.sound_cues is False


def test_the_ui_section_wins_over_the_old_alias(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('sound_cues = false\n[ui]\nsounds = true\npill = false\n')
    cfg = load_config(path)
    assert cfg.ui.sounds is True
    assert cfg.ui.pill is False


def test_load_config_reads_the_ui_section(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[ui]\npill = false\nposition = "bottom"\n')
    cfg = load_config(path)
    assert cfg.ui.pill is False
    assert cfg.ui.position == "bottom"


def test_load_config_without_a_file_is_all_defaults(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.hotkey.mode == "auto"
    assert cfg.hotkey.submit == "auto"


# ------------------------------------------------------------------ validation


def test_validate_accepts_every_documented_mode():
    for mode in ("auto", "push_to_talk", "toggle"):
        cfg = Config()
        cfg.hotkey.mode = mode
        validate_config(cfg)
    for submit in ("auto", "voice"):
        cfg = Config()
        cfg.hotkey.submit = submit
        validate_config(cfg)


def test_validate_rejects_an_unknown_mode():
    cfg = Config()
    cfg.hotkey.mode = "hold_to_talk"

    with pytest.raises(ValueError) as err:
        validate_config(cfg)

    message = str(err.value)
    assert "hold_to_talk" in message
    assert "auto" in message and "push_to_talk" in message and "toggle" in message


def test_validate_rejects_an_unknown_submit():
    cfg = Config()
    cfg.hotkey.submit = "always"

    with pytest.raises(ValueError) as err:
        validate_config(cfg)

    message = str(err.value)
    assert "always" in message
    assert "auto" in message and "voice" in message


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
    # a hotkey that silently never fires.
    from bol.daemon import Daemon

    cfg = Config()
    cfg.hotkey.mode = "typo"
    with pytest.raises(ValueError):
        Daemon(cfg, text_mode=True)


def test_live_words_are_on_with_a_context_that_actually_streams():
    cfg = Config()
    assert cfg.stt.live is True
    # parakeet-mlx's own default right context of 256 frames is 20 seconds of
    # held-back text. 16 frames commits a word after 1.3 s, which is a pill
    # you can read while you talk.
    assert cfg.stt.stream_context == [256, 16]
    assert cfg.stt.stream_chunk_ms == 320


def test_load_config_applies_the_streaming_fields(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[stt]\nlive = false\nstream_context = [256, 8]\nstream_chunk_ms = 160\n"
    )
    cfg = load_config(path)
    assert cfg.stt.live is False
    assert cfg.stt.stream_context == [256, 8]
    assert cfg.stt.stream_chunk_ms == 160
