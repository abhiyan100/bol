"""Focused-app bridge: paste into whatever terminal the user is looking at.

FluidVoice-style insertion with zero extra dependencies:
  pbcopy swaps the pasteboard -> System Events presses Cmd+V -> pasteboard
  restored. Enter/Escape/Ctrl-U go through System Events key codes.

Needs the Automation/Accessibility permission macOS prompts for on first use
(granted to the terminal running Bol). A frontmost-app allowlist keeps speech
from landing in Slack: if the front app isn't a terminal, injection is
refused with a spoken warning instead.
"""

from __future__ import annotations

import asyncio
import logging

from .base import BridgeError

log = logging.getLogger("bol.bridge")

DEFAULT_ALLOWED_APPS = [
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "com.mitchellh.ghostty",
    "dev.warp.Warp-Stable",
    "net.kovidgoyal.kitty",
    "org.alacritty",
    "com.microsoft.VSCode",
    "com.todesktop.230313mzl4w4u92",  # Cursor
    "com.jetbrains.intellij",
]

_KEYCODES = {"Enter": 36, "Escape": 53}


async def _run(cmd: list[str], stdin: bytes | None = None) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(stdin)
    if proc.returncode != 0:
        raise BridgeError(f"{cmd[0]} failed: {err.decode().strip()[:200]}")
    return out


async def _osascript(script: str) -> str:
    out = await _run(["osascript", "-e", script])
    return out.decode().strip()


class SubmitBlocked(BridgeError):
    """Text was typed, but auto-Enter was withheld."""


class FocusedBridge:
    def __init__(self, allowed_apps: list[str] | None, enter_delay_s: float) -> None:
        self._allowed = allowed_apps if allowed_apps else DEFAULT_ALLOWED_APPS
        self._enter_delay = enter_delay_s

    async def attach(self) -> str:
        # Nothing to pin; surface the permission need early with a probe.
        try:
            front = await self._frontmost()
        except BridgeError as exc:
            raise BridgeError(
                "System Events probe failed. Grant Automation/Accessibility "
                f"to your terminal in System Settings. ({exc})"
            )
        return f"focused app (currently {front or 'unknown'})"

    async def _frontmost(self) -> str:
        return await _osascript(
            'tell application "System Events" to get bundle identifier of '
            "first application process whose frontmost is true"
        )

    async def _guard(self) -> None:
        front = await self._frontmost()
        if front not in self._allowed:
            raise BridgeError(
                f"front app {front or 'unknown'} isn't a terminal; "
                "click into your Claude Code window first"
            )

    async def _front_window_title(self) -> str | None:
        """Best-effort front window title; None when unreadable."""
        try:
            return await _osascript(
                'tell application "System Events" to get name of front window '
                "of (first application process whose frontmost is true)"
            )
        except BridgeError:
            return None

    async def _submit_allowed(self) -> bool:
        """Auto-Enter only when the front window looks like a Claude session.
        A terminal tab title without "claude" is likely a plain shell, where
        pasted speech plus Enter would EXECUTE as a command. If the title
        can't be read at all, allow (the app allowlist already passed)."""
        title = await self._front_window_title()
        if title is None:
            return True
        return "claude" in title.lower()

    async def inject(self, text: str, submit: bool) -> None:
        await self._guard()
        if submit and not await self._submit_allowed():
            if text:
                await self.inject(text, submit=False)
            raise SubmitBlocked(
                "typed it, but that window doesn't look like Claude, "
                "so press Enter yourself"
            )
        if text:
            # Snapshot pasteboard, paste ours, restore.
            try:
                previous = await _run(["pbpaste"])
            except BridgeError:
                previous = b""
            await _run(["pbcopy"], stdin=text.encode())
            await _osascript(
                'tell application "System Events" to keystroke "v" using command down'
            )
            await asyncio.sleep(0.15)
            await _run(["pbcopy"], stdin=previous)
        if submit:
            await asyncio.sleep(self._enter_delay)
            await _osascript('tell application "System Events" to key code 36')

    async def inject_keys(self, *keys: str) -> None:
        await self._guard()
        if "Enter" in keys and not await self._submit_allowed():
            raise SubmitBlocked(
                "that window doesn't look like Claude, so press Enter yourself"
            )
        for key in keys:
            if key in _KEYCODES:
                await _osascript(
                    f'tell application "System Events" to key code {_KEYCODES[key]}'
                )
            elif key == "C-u":
                await _osascript(
                    'tell application "System Events" to keystroke "u" using control down'
                )
            else:
                raise BridgeError(f"unsupported key: {key}")

    async def interrupt(self) -> None:
        await self.inject_keys("Escape")
