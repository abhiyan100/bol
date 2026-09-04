"""Codex CLI talk-back: hooks file, payload mapping, and the name Bol says.

Bol narrates one agent at a time and must never call one the other, so every
test here is about a Codex payload arriving at the same daemon that serves
Claude Code, and coming back out as Codex.
"""

import json
from pathlib import Path

import pytest

import bol.cli as cli
from bol.config import Config
from bol.hooks import installer
from bol.hooks.events import StopEvent, TurnTracker, agent_of, display_name
from bol.daemon import Daemon
from bol.summarize.template import TemplateSummarizer

URL = "http://127.0.0.1:8770/hook"


def _hooks(tmp_path):
    return json.loads((tmp_path / ".codex" / "hooks.json").read_text())


def _codex(**kw):
    """A Codex install into a temp home-shaped directory."""
    return dict(scope="project", project_dir=kw.pop("tmp_path"), agent="codex", **kw)


# --------------------------------------------------------------------- installer


def test_codex_hooks_go_to_hooks_json_with_codex_events(tmp_path):
    path = installer.install(URL, **_codex(tmp_path=tmp_path))
    assert path == tmp_path / ".codex" / "hooks.json"
    settings = _hooks(tmp_path)
    assert sorted(settings["hooks"]) == ["PermissionRequest", "PostToolUse", "Stop"]
    # Claude's Notification has no Codex counterpart and must not be written.
    assert "Notification" not in settings["hooks"]


def test_codex_stop_hook_echoes_json_and_never_fails(tmp_path):
    installer.install(f"{URL}?token=abc", **_codex(tmp_path=tmp_path))
    settings = _hooks(tmp_path)
    for event in installer.CODEX_EVENTS:
        (hook,) = settings["hooks"][event][0]["hooks"]
        assert hook["type"] == "command"
        assert hook["async"] is True
        # Codex rejects a Stop hook that exits 0 without JSON on stdout.
        assert hook["command"].endswith("; echo '{}'")
        assert "|| true" in hook["command"]
        assert f"'{URL}?token=abc'" in hook["command"]
    # No matcher: Codex's schema has no such key, and a trust review reads it.
    assert "matcher" not in settings["hooks"]["PostToolUse"][0]


def test_codex_install_is_idempotent_and_replaces_an_old_token(tmp_path):
    installer.install(f"{URL}?token=old", **_codex(tmp_path=tmp_path))
    installer.install(f"{URL}?token=new", **_codex(tmp_path=tmp_path))
    settings = _hooks(tmp_path)
    assert len(settings["hooks"]["Stop"]) == 1
    assert "old" not in json.dumps(settings)
    assert "new" in json.dumps(settings)


def test_codex_installed_and_uninstall_roundtrip(tmp_path):
    assert not installer.installed(URL, **_codex(tmp_path=tmp_path))
    installer.install(URL, **_codex(tmp_path=tmp_path))
    assert installer.installed(URL, **_codex(tmp_path=tmp_path))
    # And the Claude answer is independent: a different file entirely.
    assert not installer.installed(URL, scope="project", project_dir=tmp_path)

    installer.uninstall(f"{URL}?token=whatever", **_codex(tmp_path=tmp_path))
    assert _hooks(tmp_path).get("hooks", {}) == {}


def test_codex_install_preserves_other_hooks_and_backs_up_once(tmp_path):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    path = codex_dir / "hooks.json"
    path.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [{"command": "echo hi"}]}]}}))

    installer.install(f"{URL}?token=one", **_codex(tmp_path=tmp_path))
    installer.install(f"{URL}?token=two", **_codex(tmp_path=tmp_path))

    assert "echo hi" in json.dumps(_hooks(tmp_path))
    backup = codex_dir / ("hooks.json" + installer.BACKUP_SUFFIX)
    assert "echo hi" in backup.read_text()
    assert "token=one" not in backup.read_text()


def test_an_unknown_agent_is_refused():
    with pytest.raises(ValueError):
        installer.install(URL, agent="nothing")


def test_the_claude_command_is_untouched_by_the_codex_one():
    assert not installer.hook_command(URL).endswith("; echo '{}'")
    assert installer.bol_hook(URL) == installer.bol_hook(URL, "claude")


# ------------------------------------------------------------------------ events


def _codex_stop(session="c1", message="Refactored the parser."):
    return {
        "hook_event_name": "Stop",
        "session_id": session,
        "turn_id": f"t-{session}",
        "model": "gpt-5-codex",
        "cwd": "/work/repo",
        "transcript_path": "/tmp/t.jsonl",
        "last_assistant_message": message,
    }


def test_codex_stop_becomes_a_stop_event():
    event = TurnTracker().finish_turn(_codex_stop())
    assert isinstance(event, StopEvent)
    assert event.agent == "codex"
    assert event.session_id == "c1"
    # turn_id stands in for Claude's prompt_id, so the turn still has a key.
    assert event.prompt_id == "t-c1"
    assert event.last_assistant_message == "Refactored the parser."
    assert event.cwd == "/work/repo"


def test_codex_tools_reach_their_stop_through_the_session_fallback():
    # Codex's PostToolUse carries no turn_id, so it lands under session_id
    # while the Stop that closes the turn is keyed by turn_id.
    tracker = TurnTracker()
    tracker.record_tool(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "c1",
            "tool_use_id": "u1",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"},
            "tool_response": {},
        },
        "codex",
    )
    event = tracker.finish_turn(_codex_stop(), "codex")
    assert [(t.tool_name, t.detail, t.agent) for t in event.tools] == [
        ("Bash", "pytest -q", "codex")
    ]


def test_apply_patch_reports_the_file_not_the_diff():
    tracker = TurnTracker()
    patch = (
        "*** Begin Patch\n*** Update File: bol/daemon.py\n"
        "@@\n-old\n+new\n*** End Patch\n"
    )
    use = tracker.record_tool(
        {
            "session_id": "c1",
            "tool_name": "apply_patch",
            "tool_input": {"command": patch},
            "tool_response": {},
        },
        "codex",
    )
    assert use.detail == "bol/daemon.py"


def test_an_mcp_tool_with_nothing_to_show_keeps_its_name():
    use = TurnTracker().record_tool(
        {
            "session_id": "c1",
            "tool_name": "mcp__github__create_issue",
            "tool_input": {"title": "bug"},
            "tool_response": {},
        },
        "codex",
    )
    assert use.tool_name == "mcp__github__create_issue"
    assert use.detail == ""


def test_permission_request_names_the_command():
    note = TurnTracker.permission_request(
        {
            "hook_event_name": "PermissionRequest",
            "session_id": "c1",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf build"},
        }
    )
    assert note.agent == "codex"
    assert note.notification_type == "permission_prompt"
    assert note.message == "Codex wants to run: rm -rf build"
    assert note.session_id == "c1"


def test_permission_request_without_a_command_names_the_tool():
    note = TurnTracker.permission_request(
        {
            "hook_event_name": "PermissionRequest",
            "session_id": "c1",
            "tool_name": "mcp__fetch__get",
            "tool_input": {},
        }
    )
    assert note.message == "Codex asks: mcp__fetch__get"


@pytest.mark.parametrize(
    "payload, default, expected",
    [
        ({"hook_event_name": "PermissionRequest"}, "claude", "codex"),
        ({"turn_id": "t1"}, "claude", "codex"),
        ({"prompt_id": "p1"}, "codex", "claude"),
        ({"hook_event_name": "Notification"}, "claude", "claude"),
        # No signal either way: whatever the caller is already narrating.
        ({"session_id": "s"}, "codex", "codex"),
    ],
)
def test_agent_inference(payload, default, expected):
    assert agent_of(payload, default) == expected


def test_display_name_falls_back_to_claude():
    assert display_name("codex") == "Codex"
    assert display_name("claude") == "Claude"
    assert display_name("") == "Claude"


# ------------------------------------------------------------------- summarizer


@pytest.mark.asyncio
async def test_the_template_summarizer_names_the_agent():
    cfg = Config()
    event = StopEvent(
        session_id="c1",
        prompt_id="t1",
        last_assistant_message="",
        transcript_path="",
        cwd="",
        agent="codex",
    )
    text = await TemplateSummarizer(cfg).summarize(event)
    assert "Codex is done" in text
    assert "Claude" not in text


@pytest.mark.asyncio
async def test_the_template_summarizer_still_says_claude_by_default():
    event = StopEvent("s", "p", "", "", "")
    text = await TemplateSummarizer(Config()).summarize(event)
    assert "Claude is done" in text


def test_the_persona_prompt_names_the_agent_it_describes():
    from bol.summarize.llm import LlmSummarizer

    summarizer = LlmSummarizer(Config(), engine=None)
    assert "Claude" in summarizer.system_prompt
    summarizer.agent = "codex"
    assert "Codex" in summarizer.system_prompt
    assert "Claude" not in summarizer.system_prompt


# ----------------------------------------------------------------------- daemon


class FakeBridge:
    def __init__(self):
        self.injected = []
        self.keys = []
        self.explicit = []

    async def inject(self, text, submit, explicit=False):
        self.explicit.append(explicit)
        self.injected.append((text, submit))

    async def inject_keys(self, *keys, explicit=False):
        self.explicit.append(explicit)
        self.keys.append(keys)

    async def interrupt(self):
        self.keys.append(("Escape",))


class FakeSpeaker:
    def __init__(self):
        self.spoken = []

    async def speak(self, text):
        self.spoken.append(text)

    async def stop(self):
        pass


class FakeHud:
    def __init__(self):
        self.calls = []

    async def start(self):
        pass

    async def stop(self):
        pass

    def set(self, state, text="", detail="", hold=0.0):
        self.calls.append((state, text, detail))


def _daemon(talk_back=True):
    cfg = Config()
    cfg.ui.sounds = False
    # Cleanup is "always" by default; these tests are about hooks and would
    # otherwise load a 195 MB model on every utterance.
    cfg.cleanup.mode = "off"
    cfg.cleanup.model = ""
    cfg.talk_back = talk_back
    d = Daemon(cfg, text_mode=True, clock=lambda: 0.0)
    d.bridge = FakeBridge()
    d.speaker = FakeSpeaker() if talk_back else None
    d.hud = FakeHud()
    d.summarizer = TemplateSummarizer(cfg) if talk_back else None
    return d


@pytest.mark.asyncio
async def test_a_codex_stop_is_spoken_as_codex():
    d = _daemon()
    await d._on_stop(_codex_stop(message="Renamed the config keys and fixed a test."))
    (said,) = d.speaker.spoken
    assert "Codex" in said
    assert "Claude" not in said
    assert d.agent_name == "Codex"


@pytest.mark.asyncio
async def test_a_claude_stop_is_still_spoken_as_claude():
    d = _daemon()
    await d._on_stop(
        {
            "hook_event_name": "Stop",
            "session_id": "s1",
            "prompt_id": "p1",
            "cwd": "/work/repo",
            "last_assistant_message": "Fixed the flaky test and reran the suite.",
        }
    )
    (said,) = d.speaker.spoken
    assert "Claude" in said
    assert "Codex" not in said


@pytest.mark.asyncio
async def test_a_codex_permission_request_arms_go_ahead():
    d = _daemon()
    await d._on_permission_request(
        {
            "hook_event_name": "PermissionRequest",
            "session_id": "c1",
            "cwd": "/work/repo",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf build"},
        }
    )
    assert d._permission_session == "c1"
    assert d.hud.calls[-1] == ("permission", "Codex wants to run: rm -rf build", "")
    assert "Codex wants to run: rm -rf build" in d.speaker.spoken[-1]

    reopen = await d._handle_utterance("go ahead")
    assert reopen is False
    assert d.bridge.keys == [("Enter",)]
    assert d.bridge.explicit == [True]


@pytest.mark.asyncio
async def test_one_way_ignores_a_codex_permission_request():
    # Pure dictation says nothing about a coding agent, Codex included: no
    # pill, nothing armed, and in a real run no hook server to hear it on.
    d = _daemon(talk_back=False)
    await d._on_permission_request(
        {
            "hook_event_name": "PermissionRequest",
            "session_id": "c1",
            "tool_name": "apply_patch",
            "tool_input": {"command": "*** Begin Patch\n*** Add File: a.py\n"},
        }
    )
    assert d._permission_session is None
    assert d.hud.calls == []


@pytest.mark.asyncio
async def test_the_mode_line_follows_the_agent_being_narrated():
    # One-way never learns an agent's name (it hears no hooks at all), so the
    # line it prints is the fixed invitation. Two-way is where the name comes
    # from, and the mode line follows whoever spoke.
    assert (
        _daemon(talk_back=False).mode_line()
        == "bol: one-way (dictation). Add --talk-back to hear what Claude did."
    )
    d = _daemon()
    await d._on_stop(_codex_stop())
    assert d.agent_name == "Codex"


@pytest.mark.asyncio
async def test_the_first_session_wins_whichever_agent_it_is():
    d = _daemon()
    await d._on_stop(_codex_stop(session="c1", message="Codex went first here."))
    await d._on_stop(
        {
            "hook_event_name": "Stop",
            "session_id": "s2",
            "prompt_id": "p2",
            "last_assistant_message": "Claude arrived second and is ignored.",
        }
    )
    assert d._bound_session == "c1"
    assert len(d.speaker.spoken) == 1
    assert d.agent_name == "Codex"


@pytest.mark.asyncio
async def test_follow_all_narrates_both_agents():
    d = _daemon()
    d.cfg.server.follow = "all"
    await d._on_stop(_codex_stop(session="c1", message="Codex finished its turn."))
    await d._on_stop(
        {
            "hook_event_name": "Stop",
            "session_id": "s2",
            "prompt_id": "p2",
            "last_assistant_message": "Claude finished its turn as well.",
        }
    )
    assert len(d.speaker.spoken) == 2
    assert "Codex" in d.speaker.spoken[0]
    assert "Claude" in d.speaker.spoken[1]


@pytest.mark.asyncio
async def test_bridge_errors_name_the_agent_being_narrated():
    from bol.bridge import BridgeError

    d = _daemon()
    await d._on_stop(_codex_stop())

    class Broken(FakeBridge):
        async def inject(self, text, submit, explicit=False):
            raise BridgeError("no pane")

    d.bridge = Broken()
    await d._handle_utterance("add a login test")
    assert "Couldn't reach Codex" in d.speaker.spoken[-1]


# -------------------------------------------------------------------------- cli


@pytest.fixture
def wired(monkeypatch):
    """cmd_hook / _setup_hooks with no token file and no real writes."""
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: Config())
    monkeypatch.setattr(cli, "_url", lambda cfg: f"{URL}?token=t")
    calls = []

    def fake_install(url, scope="user", project_dir=None, agent="claude"):
        calls.append((agent, scope))
        return Path(f"/fake/{agent}")

    monkeypatch.setattr(cli.installer, "install", fake_install)
    monkeypatch.setattr(
        cli.installer, "uninstall", lambda *a, **k: Path("/fake/removed")
    )
    return calls


def _on_path(monkeypatch, *names):
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: f"/usr/bin/{name}" if name in names else None
    )


def _two_way():
    cfg = Config()
    cfg.talk_back = True
    return cfg


def test_setup_wires_every_agent_on_path(wired, monkeypatch, capsys):
    _on_path(monkeypatch, "claude", "codex")
    cli._setup_hooks(_two_way())
    assert [agent for agent, _scope in wired] == ["claude", "codex"]
    out = capsys.readouterr().out
    assert installer.CODEX_TRUST_NOTE in out
    assert "hooks installed for Claude Code and Codex CLI" in out
    assert "; echo '{}'" in out


def test_setup_wires_only_the_agent_that_is_installed(wired, monkeypatch, capsys):
    _on_path(monkeypatch, "codex")
    cli._setup_hooks(_two_way())
    assert [agent for agent, _scope in wired] == ["codex"]
    out = capsys.readouterr().out
    assert "hooks installed for Codex CLI." in out
    assert "Claude Code" not in out


def test_setup_writes_no_hooks_when_there_is_no_agent_to_wire(
    wired, monkeypatch, capsys
):
    # Nothing on PATH is the same answer as "anything, no coding agent": Bol
    # dictates, and nobody's settings file is opened.
    _on_path(monkeypatch)
    cli._setup_hooks(_two_way())
    assert wired == []
    assert cli.NO_HOOKS_LINE in capsys.readouterr().out


def test_setup_writes_no_hooks_for_dictation_only(wired, monkeypatch, capsys):
    # Both agents installed, talk-back off: still no settings file touched.
    _on_path(monkeypatch, "claude", "codex")
    cli._setup_hooks(Config())
    assert wired == []
    out = capsys.readouterr().out
    assert cli.NO_HOOKS_LINE in out
    assert "bol hook install" in out


def test_hook_install_agent_codex_prints_the_trust_note(wired, capsys):
    assert cli.main(["hook", "install", "--agent", "codex"]) == 0
    assert wired == [("codex", "user")]
    out = capsys.readouterr().out
    assert installer.CODEX_TRUST_NOTE in out
    assert "restart running Codex CLI sessions" in out


def test_hook_install_defaults_to_claude(wired, capsys):
    assert cli.main(["hook", "install"]) == 0
    assert wired == [("claude", "user")]
    assert installer.CODEX_TRUST_NOTE not in capsys.readouterr().out


def test_codex_project_scope_falls_back_to_the_one_file_codex_reads(wired, capsys):
    assert cli.main(["hook", "install", "--agent", "codex", "--scope", "project"]) == 0
    assert wired == [("codex", "user")]
    assert "~/.codex/hooks.json only" in capsys.readouterr().out


def test_hook_uninstall_takes_an_agent(wired, monkeypatch, capsys):
    removed = []
    monkeypatch.setattr(
        cli.installer,
        "uninstall",
        lambda url, scope="user", project_dir=None, agent="claude": removed.append(agent)
        or Path("/fake/removed"),
    )
    assert cli.main(["hook", "uninstall", "--agent", "codex"]) == 0
    assert removed == ["codex"]


def test_doctor_reports_hooks_per_agent(monkeypatch):
    monkeypatch.setattr(cli, "_url", lambda cfg: f"{URL}?token=t")
    monkeypatch.setattr(cli.installer, "installed", lambda url, agent="claude": True)
    _on_path(monkeypatch, "claude", "codex")
    labels = [label for _status, label, _hint in cli.probe_wiring(_two_way())]
    assert "hooks installed for Claude Code (user scope)" in labels
    assert "hooks installed for Codex CLI (user scope)" in labels
