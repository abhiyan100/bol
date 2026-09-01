"""tmux bridge: find the Claude Code pane, inject text, press Enter.

Injection contract (empirically verified against Claude Code 2.1.x):
  - Text goes in via load-buffer + paste-buffer, NEVER send-keys: send-keys
    submits on embedded newlines and mangles '#', '!', '$'.
  - Enter is a separate send-keys call after a short delay, otherwise the TUI
    can swallow it as part of the paste.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


from .base import BridgeError


class TmuxError(BridgeError):
    pass


@dataclass
class ClaudePane:
    pane_id: str        # immutable id, e.g. "%3"
    target: str         # human-readable session:window.pane
    command: str        # pane_current_command


async def _tmux(*args: str) -> str:
    if shutil.which("tmux") is None:
        raise TmuxError("tmux is not installed (brew install tmux)")
    proc = await asyncio.create_subprocess_exec(
        "tmux", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise TmuxError(f"tmux {' '.join(args)} failed: {err.decode().strip()}")
    return out.decode()


# Claude Code retitles its process to its bare version ("2.1.252"), so
# pane_current_command can't be trusted. Ground truth is the pane TTY's
# process table.
_VERSIONISH = re.compile(r"^\d+\.\d+\.\d+")


async def _looks_like_claude(command: str, pane_tty: str) -> bool:
    if command and command.lower() not in {"claude", "node", "bun"} and not _VERSIONISH.match(command):
        return False
    if not pane_tty:
        return bool(_VERSIONISH.match(command)) or command.lower() == "claude"
    proc = await asyncio.create_subprocess_exec(
        "ps", "-o", "comm=", "-t", pane_tty.removeprefix("/dev/"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    names = {
        line.strip().rsplit("/", 1)[-1].lower()
        for line in out.decode().splitlines()
    }
    return "claude" in names or bool(_VERSIONISH.match(command))


class TmuxBridge:
    def __init__(self, pane: str = "", enter_delay_s: float = 0.2) -> None:
        self._pinned = pane
        self._enter_delay = enter_delay_s
        self._pane: ClaudePane | None = None

    async def discover(self) -> list[ClaudePane]:
        """All panes that look like they're running Claude Code."""
        out = await _tmux(
            "list-panes", "-a", "-F",
            "#{pane_id}\t#{session_name}:#{window_index}.#{pane_index}\t#{pane_current_command}\t#{pane_tty}",
        )
        panes: list[ClaudePane] = []
        for line in out.splitlines():
            pane_id, target, command, tty = (line.split("\t") + [""] * 4)[:4]
            if await _looks_like_claude(command, tty):
                panes.append(ClaudePane(pane_id, target, command))
        return panes

    async def attach(self) -> str:
        """Pin the target pane. Prefers explicit config, else sole discovery.
        Returns a human-readable description of the target."""
        pane = await self._attach_pane()
        return f"tmux pane {pane.pane_id} ({pane.target})"

    async def _attach_pane(self) -> ClaudePane:
        if self._pinned:
            out = await _tmux(
                "display-message", "-p", "-t", self._pinned,
                "#{pane_id}\t#{session_name}:#{window_index}.#{pane_index}\t#{pane_current_command}",
            )
            pane_id, target, command = (out.strip().split("\t") + ["", "", ""])[:3]
            self._pane = ClaudePane(pane_id, target, command)
            return self._pane

        panes = await self.discover()
        if not panes:
            raise TmuxError(
                "no tmux pane running Claude Code found. Start one with "
                "`bol launch` or run `claude` inside tmux"
            )
        if len(panes) > 1:
            listing = ", ".join(f"{p.pane_id} ({p.target})" for p in panes)
            raise TmuxError(
                f"multiple Claude panes found: {listing}. "
                f"Pin one in config: [bridge] pane = \"%N\""
            )
        self._pane = panes[0]
        return self._pane

    @property
    def pane(self) -> ClaudePane:
        if self._pane is None:
            raise TmuxError("bridge not attached")
        return self._pane

    async def verify(self) -> bool:
        """Re-check the pinned pane still exists and runs Claude."""
        try:
            out = await _tmux(
                "display-message", "-p", "-t", self.pane.pane_id,
                "#{pane_current_command}\t#{pane_tty}",
            )
        except TmuxError:
            return False
        command, tty = (out.strip().split("\t") + ["", ""])[:2]
        return await _looks_like_claude(command, tty)

    async def inject(self, text: str, submit: bool) -> None:
        """Paste text into the pane; optionally press Enter to send."""
        if not await self.verify():
            raise TmuxError(
                f"pane {self.pane.pane_id} is gone or no longer running Claude"
            )
        if text:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(text)
                path = Path(f.name)
            try:
                await _tmux("load-buffer", "-b", "bol", str(path))
                await _tmux("paste-buffer", "-b", "bol", "-t", self.pane.pane_id, "-d")
            finally:
                path.unlink(missing_ok=True)
        if submit:
            await asyncio.sleep(self._enter_delay)
            await _tmux("send-keys", "-t", self.pane.pane_id, "Enter")

    async def interrupt(self) -> None:
        await _tmux("send-keys", "-t", self.pane.pane_id, "Escape")

    async def inject_keys(self, *keys: str) -> None:
        """Send raw tmux key names (e.g. "C-u", "Escape", "Enter")."""
        await _tmux("send-keys", "-t", self.pane.pane_id, *keys)

    @staticmethod
    async def launch(session: str = "bol", cwd: str | None = None) -> ClaudePane:
        """Create a detached tmux session running claude; caller may attach."""
        args = ["new-session", "-d", "-s", session]
        if cwd:
            args += ["-c", cwd]
        args += ["claude"]
        await _tmux(*args)
        out = await _tmux(
            "display-message", "-p", "-t", session,
            "#{pane_id}\t#{session_name}:#{window_index}.#{pane_index}\t#{pane_current_command}",
        )
        pane_id, target, command = (out.strip().split("\t") + ["", "", ""])[:3]
        return ClaudePane(pane_id, target, command)
