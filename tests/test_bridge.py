import pytest

from bol.bridge import BridgeError, FocusedBridge, TmuxBridge, TmuxError, build_bridge
from bol.bridge.base import AutoBridge
from bol.config import Config


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
    bridge = FocusedBridge(["com.apple.Terminal"], 0.0)
    calls = []

    async def fake_front():
        return "com.apple.Terminal"

    async def fake_run(cmd, stdin=None):
        calls.append((tuple(cmd), stdin))
        return b""

    async def fake_osascript(script):
        calls.append(("osascript", script))
        return ""

    monkeypatch.setattr(bridge, "_frontmost", fake_front)
    monkeypatch.setattr("bol.bridge.focused._run", fake_run)
    monkeypatch.setattr("bol.bridge.focused._osascript", fake_osascript)

    await bridge.inject("git status", submit=True)
    scripts = [c[1] for c in calls if c[0] == "osascript"]
    assert any('keystroke "v"' in s for s in scripts)  # paste happened
    assert any("key code 36" in s for s in scripts)    # Enter happened
    copied = [c for c in calls if c[0] == ("pbcopy",)]
    assert copied[0][1] == b"git status"               # our text on pasteboard
    assert len(copied) == 2                            # and restored after
