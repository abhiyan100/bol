"""Focused-app bridge: paste into whatever terminal the user is looking at.

FluidVoice-style insertion with zero extra dependencies:
  pbcopy swaps the pasteboard -> System Events presses Cmd+V -> pasteboard
  restored. Enter/Escape/Ctrl-U go through System Events key codes.

Needs the Automation/Accessibility permission macOS prompts for on first use
(granted to the terminal running Bol). A frontmost-app allowlist keeps speech
from landing in Slack: if the front app isn't a terminal or an editor, the
injection is refused with a spoken warning instead.

Two guards sit on top of the allowlist:
  - the frontmost app is re-read immediately before the Cmd+V keystroke, so a
    Cmd-Tab mid-paste aborts instead of dictating into the new app;
  - auto-Enter is withheld unless the front window is positively identified as
    a Claude Code session (see _submit_allowed), because a stray Enter in a
    plain shell executes the pasted speech.
"""

from __future__ import annotations

import asyncio
import logging
import re

from .base import BridgeError

log = logging.getLogger("bol.bridge")

# Terminal emulators: the front window title is the only thing to go on, so
# submit is gated on the title naming a Claude session.
TERMINAL_APPS = frozenset({
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "com.mitchellh.ghostty",
    "dev.warp.Warp-Stable",
    "dev.warp.Warp-Preview",
    "net.kovidgoyal.kitty",
    "org.alacritty",
    "com.github.wez.wezterm",
    "co.zeit.hyper",
    "org.tabby",
})

# Editors with a built-in terminal. Their window titles are "<file> - <root> -
# Visual Studio Code" and can never name Claude, so submit is gated on the
# process tree instead.
IDE_APPS = frozenset({
    "com.microsoft.VSCode",
    "com.microsoft.VSCodeInsiders",
    "com.vscodium",
    "com.todesktop.230313mzl4w4u92",  # Cursor
    "com.exafunction.windsurf",
    "dev.zed.Zed",
    "com.jetbrains.intellij",
    "com.jetbrains.pycharm",
    "com.jetbrains.WebStorm",
    "com.jetbrains.goland",
    "com.jetbrains.rustrover",
    "com.jetbrains.CLion",
})

DEFAULT_ALLOWED_APPS = sorted(TERMINAL_APPS | IDE_APPS)

_KEYCODES = {"Enter": 36, "Escape": 53}

# Submit gate, title branch. "claude" has to be a whole word: the substring
# rule used to allow a plain shell tab titled "claude-tools". A hyphenated
# project name still contains the word, so a title that looks like a
# "claude-<slug>" directory is treated as a shell, not a session.
_CLAUDE_WORD = re.compile(r"\bclaude\b", re.IGNORECASE)
_CLAUDE_SLUG = re.compile(r"claude-[a-z0-9]", re.IGNORECASE)

# Submit gate, process branch. Matches "claude" or "claude-code" as a whole
# path component or argument: "/usr/local/bin/claude",
# "node .../@anthropic-ai/claude-code/cli.js". Deliberately does not match
# "~/.claude/settings.json" (preceded by a dot) or "claude-notes.md".
_CLAUDE_PROC = re.compile(r"(?:^|[/\s])claude(?:-code)?(?:[/\s@]|$)", re.IGNORECASE)

_FRONT_APP_SCRIPT = (
    'tell application "System Events" to get bundle identifier of '
    "first application process whose frontmost is true"
)
_FRONT_PID_SCRIPT = (
    'tell application "System Events" to get unix id of '
    "first application process whose frontmost is true"
)
_FRONT_TITLE_SCRIPT = (
    'tell application "System Events" to get name of front window '
    "of (first application process whose frontmost is true)"
)


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


async def frontmost_bundle_id() -> str:
    """Bundle id of the frontmost app, for the doctor and the bridge.

    Returns "" when it can't be read (usually a missing Automation
    permission); this never raises, so callers can print a hint instead of a
    traceback."""
    try:
        return await _osascript(_FRONT_APP_SCRIPT)
    except BridgeError as exc:
        log.debug("frontmost app unreadable (%s)", exc)
        return ""


class SubmitBlocked(BridgeError):
    """Text was typed, but auto-Enter was withheld."""

    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason or message


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
        return await _osascript(_FRONT_APP_SCRIPT)

    async def _frontmost_pid(self) -> int:
        out = await _osascript(_FRONT_PID_SCRIPT)
        try:
            return int(out.strip())
        except ValueError:
            raise BridgeError(f"unreadable front app pid: {out[:40]!r}")

    async def _guard(self) -> str:
        """Check the front app is on the allowlist; returns its bundle id."""
        front = await self._frontmost()
        if front not in self._allowed:
            raise BridgeError(
                f"front app {front or 'unknown'} isn't a terminal; "
                "click into your Claude Code window first"
            )
        return front

    async def _front_window_title(self) -> str | None:
        """Best-effort front window title; None when unreadable."""
        try:
            return await _osascript(_FRONT_TITLE_SCRIPT)
        except BridgeError:
            return None

    async def _process_tree(self) -> list[tuple[int, int, str]]:
        """(pid, ppid, command line) for every process, for the submit gate.

        comm can itself contain spaces on macOS ("Visual Studio Code.app/..."),
        so comm and args stay joined as one searchable string."""
        out = await _run(["ps", "-axo", "pid,ppid,comm,args"])
        rows: list[tuple[int, int, str]] = []
        for line in out.decode(errors="replace").splitlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                rows.append((int(parts[0]), int(parts[1]), parts[2]))
            except ValueError:
                continue
        return rows

    async def _has_claude_descendant(self, root_pid: int) -> bool:
        """True when a Claude Code process runs under the frontmost app."""
        try:
            rows = await self._process_tree()
        except BridgeError as exc:
            log.debug("process tree unreadable (%s)", exc)
            return False
        children: dict[int, list[tuple[int, str]]] = {}
        for pid, ppid, cmdline in rows:
            children.setdefault(ppid, []).append((pid, cmdline))
        seen: set[int] = set()
        stack = [root_pid]
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            for child_pid, cmdline in children.get(pid, ()):
                if _CLAUDE_PROC.search(cmdline):
                    return True
                stack.append(child_pid)
        return False

    async def _submit_allowed(self, front: str | None = None) -> tuple[bool, str]:
        """Auto-Enter only when the front window is a Claude Code session.

        Returns (allowed, reason). Editors get a process-tree check because
        their titles can't carry the session name; everything else has to name
        Claude in the front window title. Unreadable means blocked: a stray
        Enter in a plain shell executes the pasted speech."""
        if front is None:
            front = await self._frontmost()
        if front in IDE_APPS:
            try:
                pid = await self._frontmost_pid()
            except BridgeError:
                return False, "that editor's process list can't be read"
            if await self._has_claude_descendant(pid):
                return True, ""
            return False, "that editor has no Claude Code session running in it"
        title = await self._front_window_title()
        if title is None:
            return False, "that window's title can't be read"
        if not _CLAUDE_WORD.search(title) or _CLAUDE_SLUG.search(title):
            return False, f"the window titled {title!r} doesn't look like Claude"
        return True, ""

    async def _paste(self, text: str, front: str) -> None:
        """Put text on the pasteboard, press Cmd+V, restore the pasteboard.

        The frontmost app is re-read right before the keystroke: the allowlist
        check happened 150 to 300ms earlier, and a Cmd-Tab in that window
        would otherwise drop the dictation into whatever app is now in front."""
        previous = await self._clipboard_snapshot()
        try:
            await _run(["pbcopy"], stdin=text.encode())
            if await self._frontmost() != front:
                raise BridgeError("focus moved away mid-paste; nothing was typed")
            await _osascript(
                'tell application "System Events" to keystroke "v" using command down'
            )
            await asyncio.sleep(0.15)
        finally:
            if previous is not None:
                try:
                    await _run(["pbcopy"], stdin=previous)
                except BridgeError as exc:
                    log.debug("clipboard restore failed (%s)", exc)

    async def _clipboard_snapshot(self) -> bytes | None:
        """Pasteboard contents to restore afterwards, or None to leave it be.

        pbpaste only round-trips plain text. An empty read from a non-empty
        pasteboard means it holds an image, a file, or rich text, and writing
        an empty string back would destroy it."""
        try:
            previous = await _run(["pbpaste"])
        except BridgeError:
            previous = b""
        if previous:
            return previous
        try:
            info = await _osascript("clipboard info")
        except BridgeError:
            info = ""
        if info.strip():
            log.info("left your clipboard alone (non-text content)")
            return None
        return previous

    async def inject(self, text: str, submit: bool) -> None:
        front = await self._guard()
        if submit:
            allowed, reason = await self._submit_allowed(front)
            if not allowed:
                if text:
                    await self._paste(text, front)
                raise SubmitBlocked(
                    f"typed it, but {reason}, so press Enter yourself", reason
                )
        if text:
            await self._paste(text, front)
        if submit:
            await asyncio.sleep(self._enter_delay)
            await _osascript('tell application "System Events" to key code 36')

    async def inject_keys(self, *keys: str) -> None:
        front = await self._guard()
        if "Enter" in keys:
            allowed, reason = await self._submit_allowed(front)
            if not allowed:
                raise SubmitBlocked(
                    f"{reason}, so press Enter yourself", reason
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
