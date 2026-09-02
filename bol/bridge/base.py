"""Bridge protocol: how text and keys reach Claude Code.

One implementation, FocusedBridge: paste into the app the user is looking at,
guarded by an allowlist and, for Enter, by whether that window is running
Claude. Zero setup, and it works wherever the user types. (A tmux bridge that
injected into a pinned pane lived here until v0.5; typing into the front app
made it redundant.)

Completion detection never depends on the bridge; hooks cover any session.
"""

from __future__ import annotations

from typing import Protocol


class BridgeError(RuntimeError):
    pass


class Bridge(Protocol):
    async def attach(self) -> str:
        """Prepare the bridge; returns a human-readable target description."""
        ...

    async def inject(self, text: str, submit: bool, *, explicit: bool = False) -> None:
        """explicit means the user asked for this in words ("send it", "go
        ahead"), which is what lets an Enter land outside a Claude window."""
        ...

    async def inject_keys(self, *keys: str, explicit: bool = False) -> None:
        """Key names: "Enter", "Escape", "C-u"."""
        ...

    async def interrupt(self) -> None: ...


def build_bridge(cfg) -> "Bridge":
    from .focused import FocusedBridge

    return FocusedBridge(
        cfg.bridge.allowed_apps, cfg.bridge.enter_delay_s, cfg.bridge.anywhere
    )
