"""Bridge protocol: how text and keys reach Claude Code.

Two implementations:
  - FocusedBridge (default): paste into the frontmost app, FluidVoice-style.
    Zero setup; guarded by a terminal-app allowlist.
  - TmuxBridge (power mode): focus-independent injection into a pinned pane.

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

    async def inject(self, text: str, submit: bool) -> None: ...

    async def inject_keys(self, *keys: str) -> None:
        """Key names: "Enter", "Escape", "C-u"."""
        ...

    async def interrupt(self) -> None: ...


def build_bridge(cfg) -> "Bridge":
    from .focused import FocusedBridge
    from .tmux import TmuxBridge

    mode = cfg.bridge.mode
    if mode == "tmux":
        return TmuxBridge(cfg.bridge.pane, cfg.bridge.enter_delay_s)
    if mode == "focused":
        return FocusedBridge(cfg.bridge.allowed_apps, cfg.bridge.enter_delay_s)
    return AutoBridge(cfg)


class AutoBridge:
    """Picks tmux when a Claude pane exists, else focused-app injection.
    Decided once at attach time."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._inner: Bridge | None = None

    async def attach(self) -> str:
        from .focused import FocusedBridge
        from .tmux import TmuxBridge, TmuxError

        tmux = TmuxBridge(self._cfg.bridge.pane, self._cfg.bridge.enter_delay_s)
        try:
            desc = await tmux.attach()
            self._inner = tmux
            return desc
        except TmuxError:
            focused = FocusedBridge(
                self._cfg.bridge.allowed_apps, self._cfg.bridge.enter_delay_s
            )
            desc = await focused.attach()
            self._inner = focused
            return desc

    @property
    def inner(self) -> Bridge:
        if self._inner is None:
            raise BridgeError("bridge not attached")
        return self._inner

    async def inject(self, text: str, submit: bool) -> None:
        await self.inner.inject(text, submit)

    async def inject_keys(self, *keys: str) -> None:
        await self.inner.inject_keys(*keys)

    async def interrupt(self) -> None:
        await self.inner.interrupt()
