import pytest

from bol.bridge import BridgeError, FocusedBridge, TmuxBridge, TmuxError, build_bridge
from bol.bridge.base import AutoBridge
from bol.bridge.focused import SubmitBlocked, frontmost_bundle_id
from bol.config import Config

TERMINAL = "com.apple.Terminal"
CURSOR = "com.todesktop.230313mzl4w4u92"


def _cfg(mode):
    cfg = Config()
    cfg.bridge.mode = mode
    return cfg


def test_factory_modes():
    assert isinstance(build_bridge(_cfg("tmux")), TmuxBridge)
    assert isinstance(build_bridge(_cfg("focused")), FocusedBridge)
    assert isinstance(build_bridge(_cfg("auto")), AutoBridge)


def test_tmux_error_is_bridge_error():
    assert issubclass(TmuxError, BridgeError)


@pytest.mark.asyncio
async def test_focused_guard_blocks_non_terminal(monkeypatch):
    bridge = FocusedBridge(None, 0.2)

    async def fake_front():
        return "com.tinyspeck.slackmacgap"

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    with pytest.raises(BridgeError, match="isn't a terminal"):
        await bridge.inject("hello", submit=False)


@pytest.mark.asyncio
async def test_focused_guard_allows_terminal(monkeypatch):
    bridge = FocusedBridge([TERMINAL], 0.0)
    calls = []

    async def fake_front():
        return TERMINAL

    async def fake_run(cmd, stdin=None):
        calls.append((tuple(cmd), stdin))
        return b""

    async def fake_osascript(script):
        calls.append(("osascript", script))
        return ""

    async def fake_title():
        return "claude - myproject"

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    monkeypatch.setattr(bridge, "_front_window_title", fake_title)
    monkeypatch.setattr("bol.bridge.focused._run", fake_run)
    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)

    await bridge.inject("git status", submit=True)
    scripts = [c[1] for c in calls if c[0] == "osascript"]
    assert any('keystroke "v"' in s for s in scripts)  # paste happened
    assert any("key code 36" in s for s in scripts)    # Enter happened
    copied = [c for c in calls if c[0] == ("pbcopy",)]
    assert copied[0][1] == b"git status"               # our text on pasteboard
    assert len(copied) == 2                            # and restored after


@pytest.mark.asyncio
async def test_focused_submit_blocked_in_plain_shell(monkeypatch):
    bridge = FocusedBridge([TERMINAL], 0.0)
    typed = []

    async def fake_front():
        return TERMINAL

    async def fake_title():
        return "zsh - ~/dotfiles"

    async def fake_run(cmd, stdin=None):
        typed.append((tuple(cmd), stdin))
        return b""

    async def fake_osascript(script):
        typed.append(("osascript", script))
        return ""

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    monkeypatch.setattr(bridge, "_front_window_title", fake_title)
    monkeypatch.setattr("bol.bridge.focused._run", fake_run)
    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)

    with pytest.raises(SubmitBlocked):
        await bridge.inject("rm -rf build send it payload", submit=True)
    scripts = [t[1] for t in typed if t[0] == "osascript"]
    assert any('keystroke "v"' in s for s in scripts)   # text still typed
    assert not any("key code 36" in s for s in scripts)  # Enter withheld


# --- clipboard handling (C1) ------------------------------------------------


@pytest.mark.asyncio
async def test_clipboard_restored_when_osascript_fails(monkeypatch):
    """Automation revoked mid-paste must not leave dictation on the clipboard."""
    bridge = FocusedBridge([TERMINAL], 0.0)
    calls = []

    async def fake_front():
        return TERMINAL

    async def fake_run(cmd, stdin=None):
        calls.append((tuple(cmd), stdin))
        return b"user clipboard" if cmd[0] == "pbpaste" else b""

    async def fake_osascript(script):
        raise BridgeError("osascript failed: not authorized to send keystrokes")

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    monkeypatch.setattr("bol.bridge.focused._run", fake_run)
    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)

    with pytest.raises(BridgeError):
        await bridge.inject("dictated text", submit=False)
    copied = [c[1] for c in calls if c[0] == ("pbcopy",)]
    assert copied == [b"dictated text", b"user clipboard"]


@pytest.mark.asyncio
async def test_non_text_clipboard_is_left_alone(monkeypatch):
    """An image on the pasteboard reads back empty; restoring would destroy it."""
    bridge = FocusedBridge([TERMINAL], 0.0)
    calls = []

    async def fake_front():
        return TERMINAL

    async def fake_run(cmd, stdin=None):
        calls.append((tuple(cmd), stdin))
        return b""

    async def fake_osascript(script):
        calls.append(("osascript", script))
        if "clipboard info" in script:
            return "{«class PNGf», 65536}"
        return ""

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    monkeypatch.setattr("bol.bridge.focused._run", fake_run)
    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)

    await bridge.inject("dictated text", submit=False)
    copied = [c[1] for c in calls if c[0] == ("pbcopy",)]
    assert copied == [b"dictated text"]  # no empty restore
    scripts = [c[1] for c in calls if c[0] == "osascript"]
    assert any('keystroke "v"' in s for s in scripts)


# --- focus race (C2) --------------------------------------------------------


@pytest.mark.asyncio
async def test_focus_change_mid_paste_aborts(monkeypatch):
    bridge = FocusedBridge([TERMINAL], 0.0)
    calls = []
    fronts = [TERMINAL, "com.tinyspeck.slackmacgap"]

    async def fake_front():
        return fronts.pop(0) if fronts else "com.tinyspeck.slackmacgap"

    async def fake_run(cmd, stdin=None):
        calls.append((tuple(cmd), stdin))
        return b"user clipboard" if cmd[0] == "pbpaste" else b""

    async def fake_osascript(script):
        calls.append(("osascript", script))
        return ""

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    monkeypatch.setattr("bol.bridge.focused._run", fake_run)
    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)

    with pytest.raises(BridgeError, match="focus moved away"):
        await bridge.inject("secret plans", submit=False)
    scripts = [c[1] for c in calls if c[0] == "osascript"]
    assert not any('keystroke "v"' in s for s in scripts)  # nothing was typed
    copied = [c[1] for c in calls if c[0] == ("pbcopy",)]
    assert copied[-1] == b"user clipboard"                 # clipboard restored


# --- submit gate (C3) -------------------------------------------------------

CLAUDE_IN_IDE = [
    (4242, 1, "/Applications/Cursor.app/Contents/MacOS/Cursor"),
    (5000, 4242, "/bin/zsh -zsh"),
    (5001, 5000, "/opt/homebrew/bin/node /Users/me/.npm/@anthropic-ai/claude-code/cli.js"),
]
NO_CLAUDE_IN_IDE = [
    (4242, 1, "/Applications/Cursor.app/Contents/MacOS/Cursor"),
    (5000, 4242, "/bin/zsh -zsh"),
    (5001, 5000, "/usr/bin/vim notes.md"),
]


async def _gate(monkeypatch, front, title=None, tree=None):
    bridge = FocusedBridge(None, 0.0)

    async def fake_front():
        return front

    async def fake_title():
        return title

    async def fake_pid():
        return 4242

    async def fake_tree():
        return list(tree or [])

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    monkeypatch.setattr(bridge, "_front_window_title", fake_title)
    monkeypatch.setattr(bridge, "_frontmost_pid", fake_pid)
    monkeypatch.setattr(bridge, "_process_tree", fake_tree)
    return await bridge._submit_allowed()


@pytest.mark.asyncio
async def test_gate_blocks_plain_shell_title(monkeypatch):
    allowed, reason = await _gate(monkeypatch, TERMINAL, title="zsh - ~/dotfiles")
    assert allowed is False
    assert "doesn't look like Claude" in reason


@pytest.mark.asyncio
async def test_gate_allows_claude_title(monkeypatch):
    allowed, _ = await _gate(monkeypatch, TERMINAL, title="claude - ~/Desktop/Bol")
    assert allowed is True


@pytest.mark.asyncio
async def test_gate_blocks_claude_slug_project_title(monkeypatch):
    """"claude-tools" is a directory name, not a session."""
    allowed, _ = await _gate(monkeypatch, TERMINAL, title="claude-tools - zsh")
    assert allowed is False


@pytest.mark.asyncio
async def test_gate_blocks_unreadable_title(monkeypatch):
    allowed, reason = await _gate(monkeypatch, TERMINAL, title=None)
    assert allowed is False
    assert "can't be read" in reason


@pytest.mark.asyncio
async def test_gate_allows_ide_with_claude_descendant(monkeypatch):
    allowed, _ = await _gate(
        monkeypatch, CURSOR, title="cleanup.py - Bol - Cursor", tree=CLAUDE_IN_IDE
    )
    assert allowed is True


@pytest.mark.asyncio
async def test_gate_blocks_ide_without_claude_descendant(monkeypatch):
    allowed, reason = await _gate(
        monkeypatch, CURSOR, title="cleanup.py - Bol - Cursor", tree=NO_CLAUDE_IN_IDE
    )
    assert allowed is False
    assert "no Claude Code session" in reason


@pytest.mark.asyncio
async def test_submit_blocked_carries_reason(monkeypatch):
    bridge = FocusedBridge([TERMINAL], 0.0)

    async def fake_front():
        return TERMINAL

    async def fake_title():
        return "zsh - ~/dotfiles"

    async def fake_run(cmd, stdin=None):
        return b""

    async def fake_osascript(script):
        return ""

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    monkeypatch.setattr(bridge, "_front_window_title", fake_title)
    monkeypatch.setattr("bol.bridge.focused._run", fake_run)
    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)

    with pytest.raises(SubmitBlocked) as exc:
        await bridge.inject("hello", submit=True)
    assert "doesn't look like Claude" in exc.value.reason


# --- doctor helper (C4) -----------------------------------------------------


@pytest.mark.asyncio
async def test_frontmost_bundle_id_is_quiet_on_failure(monkeypatch):
    async def boom(script):
        raise BridgeError("osascript failed: not authorized")

    monkeypatch.setattr("bol.bridge.focused._osascript", boom)
    assert await frontmost_bundle_id() == ""


@pytest.mark.asyncio
async def test_frontmost_bundle_id_reads_front_app(monkeypatch):
    async def fake_osascript(script):
        return "com.mitchellh.ghostty"

    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)
    assert await frontmost_bundle_id() == "com.mitchellh.ghostty"


def test_default_allowlist_covers_terminals_and_ides():
    from bol.bridge import DEFAULT_ALLOWED_APPS, IDE_APPS, TERMINAL_APPS

    assert "com.github.wez.wezterm" in TERMINAL_APPS
    assert "dev.warp.Warp-Preview" in TERMINAL_APPS
    assert "dev.zed.Zed" in IDE_APPS
    assert "com.jetbrains.pycharm" in IDE_APPS
    assert not TERMINAL_APPS & IDE_APPS
    assert set(DEFAULT_ALLOWED_APPS) == TERMINAL_APPS | IDE_APPS
