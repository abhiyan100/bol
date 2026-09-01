import json

from bol.hooks import installer

URL = "http://127.0.0.1:8770/hook"


def test_install_uninstall_roundtrip(tmp_path):
    path = installer.install(URL, scope="project", project_dir=tmp_path)
    assert path == tmp_path / ".claude" / "settings.json"
    settings = json.loads(path.read_text())
    for event in installer.EVENTS:
        urls = [
            h["url"]
            for entry in settings["hooks"][event]
            for h in entry["hooks"]
        ]
        assert URL in urls
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
