import pytest

from bol.bridge import BridgeError, FocusedBridge, build_bridge
from bol.bridge.focused import DISCARD_LINES, SubmitBlocked, frontmost_bundle_id
from bol.config import Config

TERMINAL = "com.apple.Terminal"
CURSOR = "com.todesktop.230313mzl4w4u92"
SLACK = "com.tinyspeck.slackmacgap"
NOTES = "com.apple.Notes"


def _wire(monkeypatch, front, title=None, tree=None, anywhere=True, allowed=None):
    """A FocusedBridge whose every keystroke is recorded and none of them run."""
    bridge = FocusedBridge(allowed, 0.0, anywhere=anywhere)
    calls = []

    async def fake_front():
        return front

    async def fake_title():
        return title

    async def fake_pid():
        return 4242

    async def fake_tree():
        return list(tree or [])

    async def fake_run(cmd, stdin=None):
        calls.append((tuple(cmd), stdin))
        return b""

    async def fake_osascript(script):
        calls.append(("osascript", script))
        return ""

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    monkeypatch.setattr(bridge, "_front_window_title", fake_title)
    monkeypatch.setattr(bridge, "_frontmost_pid", fake_pid)
    monkeypatch.setattr(bridge, "_process_tree", fake_tree)
    monkeypatch.setattr("bol.bridge.focused._run", fake_run)
    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)
    return bridge, calls


def _scripts(calls):
    return [c[1] for c in calls if c[0] == "osascript"]


def _pasted(calls):
    return any('keystroke "v"' in s for s in _scripts(calls))


def _entered(calls):
    return any("key code 36" in s for s in _scripts(calls))


def test_the_factory_builds_the_one_bridge_there_is():
    # tmux injection lived in v0.4 and earlier; typing into the front app made
    # it redundant, so there is nothing left to choose between.
    assert isinstance(build_bridge(Config()), FocusedBridge)


@pytest.mark.asyncio
async def test_focused_guard_blocks_non_terminal_when_anywhere_is_off(monkeypatch):
    bridge, calls = _wire(monkeypatch, SLACK, anywhere=False)
    with pytest.raises(BridgeError, match="isn't a terminal"):
        await bridge.inject("hello", submit=False)
    assert not _pasted(calls)


@pytest.mark.asyncio
async def test_dictation_lands_in_any_app_by_default(monkeypatch):
    """The paste is characters in the field the user is looking at."""
    bridge, calls = _wire(monkeypatch, NOTES)
    await bridge.inject("dinner with sam at seven", submit=False)
    assert _pasted(calls)
    assert not _entered(calls)
    copied = [c for c in calls if c[0] == ("pbcopy",)]
    assert copied[0][1] == b"dinner with sam at seven"


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


# --- where Enter may land (anywhere x explicit) ------------------------------

_APPS = {
    "terminal-claude": (TERMINAL, "claude - ~/Desktop/Bol", None),
    "terminal-shell": (TERMINAL, "zsh - ~/dotfiles", None),
    "ide-claude": (CURSOR, "cleanup.py - Bol - Cursor", CLAUDE_IN_IDE),
    "other-app": (SLACK, None, None),
}

# (anywhere, explicit, app, outcome), where outcome is what happens to Enter:
#   "enter"   pressed;
#   "blocked" text typed, Enter withheld (SubmitBlocked, the daemon says so);
#   "refused" not even typed, because the allowlist is in front of everything.
#
# The whole point of the feature is one cell: anywhere + explicit + other-app.
# Everything Bol decides for itself keeps the gate it always had.
GATE_MATRIX = [
    (True, True, "terminal-claude", "enter"),
    (True, True, "terminal-shell", "enter"),
    (True, True, "ide-claude", "enter"),
    (True, True, "other-app", "enter"),
    (True, False, "terminal-claude", "enter"),
    (True, False, "terminal-shell", "blocked"),
    (True, False, "ide-claude", "enter"),
    (True, False, "other-app", "blocked"),
    (False, True, "terminal-claude", "enter"),
    (False, True, "terminal-shell", "blocked"),
    (False, True, "ide-claude", "enter"),
    (False, True, "other-app", "refused"),
    (False, False, "terminal-claude", "enter"),
    (False, False, "terminal-shell", "blocked"),
    (False, False, "ide-claude", "enter"),
    (False, False, "other-app", "refused"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("anywhere,explicit,app,outcome", GATE_MATRIX)
async def test_submit_gate_matrix(monkeypatch, anywhere, explicit, app, outcome):
    front, title, tree = _APPS[app]
    bridge, calls = _wire(monkeypatch, front, title=title, tree=tree, anywhere=anywhere)
    inject = bridge.inject("run the tests", submit=True, explicit=explicit)
    if outcome == "enter":
        await inject
        assert _pasted(calls)
        assert _entered(calls)
        return
    with pytest.raises(BridgeError) as exc:
        await inject
    if outcome == "blocked":
        # Typed but unsent is a pending paste, not a failure: saying "send it"
        # in front of Claude finishes it.
        assert isinstance(exc.value, SubmitBlocked)
        assert _pasted(calls)
    else:
        assert not isinstance(exc.value, SubmitBlocked)
        assert not _pasted(calls)
    assert not _entered(calls)


@pytest.mark.asyncio
async def test_explicit_enter_alone_goes_anywhere(monkeypatch):
    """"Send it" on a paste already sitting in a Slack message box."""
    bridge, calls = _wire(monkeypatch, SLACK)
    await bridge.inject_keys("Enter", explicit=True)
    assert _entered(calls)


@pytest.mark.asyncio
async def test_an_automatic_key_still_needs_a_terminal(monkeypatch):
    bridge, calls = _wire(monkeypatch, SLACK)
    with pytest.raises(BridgeError, match="isn't a terminal"):
        await bridge.inject_keys("Enter")
    assert not _entered(calls)


@pytest.mark.asyncio
async def test_paste_without_enter_is_refused_when_anywhere_is_off(monkeypatch):
    bridge, calls = _wire(monkeypatch, NOTES, anywhere=False)
    with pytest.raises(BridgeError, match="isn't a terminal"):
        await bridge.inject("dinner with sam", submit=False)
    assert not _pasted(calls)


# --- discard and interrupt outside a terminal --------------------------------


def _keystrokes(calls):
    return [s for s in _scripts(calls) if "keystroke" in s or "key code" in s]


def _ctrl(letter: str) -> str:
    return (
        f'tell application "System Events" to keystroke "{letter}" '
        "using control down"
    )


@pytest.mark.asyncio
async def test_discard_in_a_terminal_clears_the_whole_box(monkeypatch):
    """"Scratch that" means the box, not the line the cursor sits on.

    Ctrl+E to the end of the logical line, because Ctrl+U only kills what is
    behind the cursor, then Ctrl+U once per line of a wrapped dictation. The
    presses past the top are free: Ctrl+U on an empty input does nothing.
    """
    bridge, calls = _wire(monkeypatch, TERMINAL)
    await bridge.inject_keys("C-u", explicit=True)
    assert _keystrokes(calls) == [_ctrl("e")] + [_ctrl("u")] * DISCARD_LINES
    assert DISCARD_LINES == 4
    assert not any("using command down" in s for s in _keystrokes(calls))


@pytest.mark.asyncio
async def test_discard_in_an_ide_clears_the_whole_box_too(monkeypatch):
    bridge, calls = _wire(monkeypatch, CURSOR)
    await bridge.inject_keys("C-u", explicit=True)
    assert _keystrokes(calls) == [_ctrl("e")] + [_ctrl("u")] * DISCARD_LINES


@pytest.mark.asyncio
async def test_discard_in_another_app_undoes_the_paste(monkeypatch):
    """Control-U means nothing in Notes; one Cmd+Z takes the paste back."""
    bridge, calls = _wire(monkeypatch, NOTES)
    await bridge.inject_keys("C-u", explicit=True)
    keys = _keystrokes(calls)
    assert keys == ['tell application "System Events" to keystroke "z" using command down']


@pytest.mark.asyncio
async def test_discard_with_an_unreadable_front_app_does_nothing(monkeypatch):
    """An undo sent into a window nobody can name edits someone else's work."""
    bridge, calls = _wire(monkeypatch, "")
    await bridge.inject_keys("C-u", explicit=True)
    assert _keystrokes(calls) == []


@pytest.mark.asyncio
async def test_interrupt_is_explicit_so_it_goes_anywhere(monkeypatch):
    bridge, calls = _wire(monkeypatch, SLACK)
    await bridge.interrupt()
    assert any("key code 53" in s for s in _keystrokes(calls))


# --- the front window title, for the words this session spells its way -------


def _title_bridge(monkeypatch, title, boom=False):
    """A bridge whose only osascript is the window-title read, counted."""
    bridge = FocusedBridge(None, 0.0)
    reads = []

    async def fake_osascript(script):
        reads.append(script)
        if boom:
            raise BridgeError("not authorized")
        return title

    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)
    return bridge, reads


@pytest.mark.asyncio
async def test_the_front_window_title_is_read_once_and_reused(monkeypatch):
    # The submit gate reads this string for its own reasons and a paste is one
    # gesture, so the cleanup pass must not pay for a second osascript.
    bridge, reads = _title_bridge(monkeypatch, "Bol - claude - 180x48")

    assert await bridge.front_title() == "Bol - claude - 180x48"
    assert await bridge.front_title() == "Bol - claude - 180x48"

    assert len(reads) == 1


@pytest.mark.asyncio
async def test_a_title_that_cannot_be_read_is_empty_not_an_error(monkeypatch):
    # A title is a spelling hint; a hint may never cost the paste.
    bridge, _reads = _title_bridge(monkeypatch, "", boom=True)

    assert await bridge.front_title() == ""


@pytest.mark.asyncio
async def test_the_submit_gate_fills_the_title_cache(monkeypatch):
    bridge, reads = _title_bridge(monkeypatch, "bol - claude")

    allowed, _reason = await bridge._submit_allowed(TERMINAL)

    assert allowed is True
    assert await bridge.front_title() == "bol - claude"
    assert len(reads) == 1  # the gate's read, reused


# --- what the config reaches -------------------------------------------------


def test_anywhere_reaches_the_focused_bridge_from_config():
    cfg = Config()
    cfg.bridge.anywhere = False
    assert build_bridge(cfg)._anywhere is False
    assert build_bridge(Config())._anywhere is True
