import json
import stat

import pytest

from bol.hooks import installer

URL = "http://127.0.0.1:8770/hook"


def _settings(tmp_path):
    return json.loads((tmp_path / ".claude" / "settings.json").read_text())


def _bol_urls(settings, event):
    return [
        installer._hook_url(h)
        for entry in settings["hooks"][event]
        for h in entry["hooks"]
        if installer._hook_url(h)
    ]


def test_install_uninstall_roundtrip(tmp_path):
    path = installer.install(URL, scope="project", project_dir=tmp_path)
    assert path == tmp_path / ".claude" / "settings.json"
    settings = json.loads(path.read_text())
    for event in installer.EVENTS:
        assert URL in _bol_urls(settings, event)
    assert installer.installed(URL, scope="project", project_dir=tmp_path)

    installer.uninstall(URL, scope="project", project_dir=tmp_path)
    settings = json.loads(path.read_text())
    assert settings.get("hooks", {}) == {}


def test_install_is_idempotent(tmp_path):
    installer.install(URL, scope="project", project_dir=tmp_path)
    installer.install(URL, scope="project", project_dir=tmp_path)
    settings = json.loads(
        (tmp_path / ".claude" / "settings.json").read_text()
    )
    assert len(settings["hooks"]["Stop"]) == 1


def test_install_preserves_existing_hooks(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    existing = {
        "model": "opus",
        "hooks": {
            "Stop": [
                {"hooks": [{"type": "command", "command": "echo done"}]}
            ]
        },
    }
    (claude_dir / "settings.json").write_text(json.dumps(existing))

    installer.install(URL, scope="project", project_dir=tmp_path)
    settings = json.loads((claude_dir / "settings.json").read_text())
    assert settings["model"] == "opus"
    commands = [
        h.get("command")
        for entry in settings["hooks"]["Stop"]
        for h in entry["hooks"]
    ]
    assert "echo done" in commands

    installer.uninstall(URL, scope="project", project_dir=tmp_path)
    settings = json.loads((claude_dir / "settings.json").read_text())
    assert "echo done" in json.dumps(settings["hooks"]["Stop"])
    assert URL not in json.dumps(settings)


def test_corrupt_settings_gives_a_friendly_error(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    # A trailing comma: hand-edited settings.json files do this all the time.
    (claude_dir / "settings.json").write_text('{\n  "model": "opus",\n}\n')

    for call in (
        lambda: installer.install(URL, scope="project", project_dir=tmp_path),
        lambda: installer.installed(URL, scope="project", project_dir=tmp_path),
        lambda: installer.uninstall(URL, scope="project", project_dir=tmp_path),
    ):
        with pytest.raises(SystemExit) as excinfo:
            call()
        message = str(excinfo.value)
        assert "settings.json isn't valid JSON" in message
        assert "Fix it, then rerun." in message


def test_write_is_atomic_and_keeps_other_keys(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    path = claude_dir / "settings.json"
    path.write_text(json.dumps({"model": "opus", "permissions": {"allow": ["Bash"]}}))
    path.chmod(0o600)

    installer.install(URL, scope="project", project_dir=tmp_path)

    settings = json.loads(path.read_text())
    assert settings["model"] == "opus"
    assert settings["permissions"] == {"allow": ["Bash"]}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    leftovers = [p.name for p in claude_dir.iterdir() if p.name != "settings.json"]
    assert leftovers == ["settings.json" + installer.BACKUP_SUFFIX]


def test_uninstall_matches_any_token(tmp_path):
    installer.install(f"{URL}?token=deadbeef", scope="project", project_dir=tmp_path)

    # A regenerated token must still find (and remove) the old entries.
    installer.uninstall(f"{URL}?token=feedface", scope="project", project_dir=tmp_path)

    settings = _settings(tmp_path)
    assert "deadbeef" not in json.dumps(settings)
    assert settings.get("hooks", {}) == {}


def test_install_replaces_an_old_token_entry(tmp_path):
    installer.install(f"{URL}?token=old", scope="project", project_dir=tmp_path)
    installer.install(f"{URL}?token=new", scope="project", project_dir=tmp_path)

    settings = _settings(tmp_path)
    assert _bol_urls(settings, "Stop") == [f"{URL}?token=new"]


def test_hook_is_an_async_command_that_never_fails(tmp_path):
    installer.install(f"{URL}?token=abc", scope="project", project_dir=tmp_path)
    settings = _settings(tmp_path)
    (hook,) = settings["hooks"]["Stop"][0]["hooks"]
    # An http hook makes Claude Code print a connection error every time Bol
    # is not running; an async command with `|| true` is silent and never
    # blocks Claude.
    assert hook["type"] == "command"
    assert hook["async"] is True
    assert hook["command"].endswith("|| true")
    assert f"'{URL}?token=abc'" in hook["command"]
    assert "--data-binary @-" in hook["command"]
    assert "-m 3" in hook["command"]
    assert settings["hooks"]["PostToolUse"][0]["matcher"] == "*"


def test_old_http_entries_are_replaced_and_removed(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    old = {
        "hooks": {
            event: [{"hooks": [{"type": "http", "url": f"{URL}?token=old"}]}]
            for event in installer.EVENTS
        }
    }
    (claude_dir / "settings.json").write_text(json.dumps(old))
    assert not installer.installed(f"{URL}?token=old", scope="project", project_dir=tmp_path)

    installer.install(f"{URL}?token=new", scope="project", project_dir=tmp_path)
    settings = _settings(tmp_path)
    assert "old" not in json.dumps(settings)
    assert '"http"' not in json.dumps(settings)
    assert installer.installed(f"{URL}?token=new", scope="project", project_dir=tmp_path)

    installer.uninstall(f"{URL}?token=whatever", scope="project", project_dir=tmp_path)
    assert _settings(tmp_path).get("hooks", {}) == {}


def test_backup_is_taken_once(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    path = claude_dir / "settings.json"
    path.write_text(json.dumps({"model": "opus"}))
    backup = claude_dir / ("settings.json" + installer.BACKUP_SUFFIX)

    installer.install(f"{URL}?token=one", scope="project", project_dir=tmp_path)
    assert json.loads(backup.read_text()) == {"model": "opus"}

    # A second install must not overwrite the pre-Bol copy with a Bol-ised one.
    installer.install(f"{URL}?token=two", scope="project", project_dir=tmp_path)
    assert json.loads(backup.read_text()) == {"model": "opus"}
