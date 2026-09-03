"""Focused-app bridge: paste into whatever terminal the user is looking at.

FluidVoice-style insertion with zero extra dependencies:
  pbcopy swaps the pasteboard -> System Events presses Cmd+V -> pasteboard
  restored. Enter/Escape/Ctrl-U go through System Events key codes.

Needs the Automation/Accessibility permission macOS prompts for on first use
(granted to the terminal running Bol).

Text and Enter are guarded separately, because they are different mistakes.
Pasting characters into the wrong window is a typo the user can see and undo;
pressing Enter there runs a command, sends a message, or answers a prompt. So
with [bridge] anywhere = true (the default) dictation lands wherever the
cursor is, Notes and Slack included, while Enter still has to be earned:
  - an explicit Enter is the user's own words ("send it", "go ahead") and goes
    wherever they are looking;
  - any other Enter is Bol's own doing, and is withheld unless the front app
    is an allowlisted terminal or IDE AND is positively identified as a
    Claude Code session (see _submit_allowed) -- a stray Enter in a plain
    shell executes the pasted speech. Nothing in Bol presses Enter by itself
    any more, so this gate is the backstop, not the daily path.
anywhere = false puts the allowlist back in front of everything, pastes and
keys alike, which is what Bol did before.

One guard sits under both: the frontmost app is re-read immediately before the
Cmd+V keystroke, so a Cmd-Tab mid-paste aborts instead of dictating into the
new app.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

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

# "Scratch that" has to clear the whole box, not the line the cursor is on.
# Claude Code's own docs say Ctrl+U deletes to the start of the line and to
# "repeat to clear across lines in multiline input", so a pasted paragraph
# takes several presses. Four is what a dictated prompt is worth, and the
# presses past the end are free: Ctrl+U on an empty input does nothing.
# Cursor to the end first (Ctrl+E), because Ctrl+U only kills what is behind it.
DISCARD_LINES = 4

# Between keystrokes. System Events delivers them faster than a TUI redraws,
# and a Ctrl+U that lands mid-redraw is a Ctrl+U the app never saw.
DISCARD_GAP_S = 0.04

# How long a title read is worth reusing. It is a spelling hint for the cleanup
# pass, so a couple of seconds old is the same window and the same project.
TITLE_CACHE_S = 2.0

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
    # Every keystroke arrives with explicit=: True when the user asked for it
    # in words, False when it is Bol's own doing. The gate below is the whole
    # reason the daemon says which.

    def __init__(
        self,
        allowed_apps: list[str] | None,
        enter_delay_s: float,
        anywhere: bool = True,
    ) -> None:
        self._allowed = allowed_apps if allowed_apps else DEFAULT_ALLOWED_APPS
        self._enter_delay = enter_delay_s
        self._anywhere = anywhere
        # The last front window title read, for whoever wants it next: the
        # submit gate reads it anyway, and the cleanup pass wants the project
        # name out of it without paying for a second osascript.
        self._title = ""
        self._title_at = 0.0

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

    async def _guard(self, anywhere_ok: bool) -> str:
        """Check the front app may receive this at all; returns its bundle id.

        anywhere_ok covers the things that are not Bol acting on its own: a
        paste, which only ever puts characters in the field the user is
        looking at, and a key the user asked for by name. Those go anywhere
        under [bridge] anywhere = true. Everything else, and every call when
        anywhere = false, still has to be a terminal or an IDE."""
        front = await self._frontmost()
        if anywhere_ok and self._anywhere:
            return front
        if front not in self._allowed:
            raise BridgeError(
                f"front app {front or 'unknown'} isn't a terminal; "
                "click into your Claude Code window first"
            )
        return front

    async def _front_window_title(self) -> str | None:
        """Best-effort front window title; None when unreadable."""
        try:
            title = await _osascript(_FRONT_TITLE_SCRIPT)
        except BridgeError:
            return None
        self._title, self._title_at = title, time.monotonic()
        return title

    async def front_title(self) -> str:
        """The front window's title, for the words this session spells its way.

        Cached for TITLE_CACHE_S, because the submit gate reads the same string
        for its own reasons and a paste is one gesture. "" when it cannot be
        read: a title is a spelling hint, and a hint may never cost a paste.
        """
        if self._title and time.monotonic() - self._title_at < TITLE_CACHE_S:
            return self._title
        return await self._front_window_title() or ""

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

    async def _enter_allowed(self, front: str, explicit: bool) -> tuple[bool, str]:
        """Whether Enter may be pressed in the front app right now.

        The split is intent. An explicit Enter came from the user's own words,
        so it means what it does in the app they are looking at: a newline in
        Notes, a sent message in Slack, a submitted prompt in Claude. Bol has
        no business second-guessing that, so anywhere mode lets it through.
        An automatic Enter is Bol's guess that an utterance was a finished
        instruction, and a guess is only worth acting on where a wrong one is
        cheap: the old gate, an allowlisted terminal or IDE with positive
        evidence that Claude is running in it."""
        if explicit and self._anywhere:
            return True, ""
        if front not in self._allowed:
            return False, f"{front or 'that window'} isn't a terminal"
        return await self._submit_allowed(front)

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

    async def inject(self, text: str, submit: bool, *, explicit: bool = False) -> None:
        """Paste text, and optionally press Enter after it.

        explicit says the Enter came from the user's own words rather than
        from Bol. The paste is guarded as a paste either way, so a
        withheld Enter still leaves the words in the box for a later
        "send it" -- that is what SubmitBlocked means to the daemon."""
        front = await self._guard(anywhere_ok=True)
        if submit:
            allowed, reason = await self._enter_allowed(front, explicit)
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

    async def inject_keys(self, *keys: str, explicit: bool = False) -> None:
        front = await self._guard(anywhere_ok=explicit)
        if "Enter" in keys:
            allowed, reason = await self._enter_allowed(front, explicit)
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
                await self._discard_line(front)
            else:
                raise BridgeError(f"unsupported key: {key}")

    async def _discard_line(self, front: str) -> None:
        """Take back what was just pasted, the way the front app understands.

        "Scratch that" means the box, not the line the cursor happens to be on:
        a dictated prompt wraps, and clearing one line of it left the rest
        sitting there. In a shell and in Claude Code's TUI that is Ctrl+E to
        the end of the logical line and then Ctrl+U, DISCARD_LINES times,
        because Ctrl+U kills back to the start of a line and has to be repeated
        to cross the lines of a multiline input. The presses past the top cost
        nothing: Ctrl+U on an empty input does nothing.

        In Notes or Slack neither key means anything, so the paste would sit
        there after the user said "scratch that". There the paste is the last
        edit, so one Cmd+Z takes exactly it back and leaves whatever was in the
        field before it alone -- which Cmd+A then Delete would not. An app that
        cannot even be named gets nothing: an undo sent into an unknown window
        is a destructive edit on someone else's work."""
        if not front:
            log.info("front app unreadable, so nothing was discarded")
            return
        if front in TERMINAL_APPS or front in IDE_APPS:
            await _osascript(
                'tell application "System Events" to keystroke "e" using control down'
            )
            for _ in range(DISCARD_LINES):
                await asyncio.sleep(DISCARD_GAP_S)
                await _osascript(
                    'tell application "System Events" to keystroke "u" using control down'
                )
            return
        await _osascript(
            'tell application "System Events" to keystroke "z" using command down'
        )

    async def interrupt(self) -> None:
        # Escape is only ever said out loud ("interrupt", "stop claude"), so
        # it goes where the user is looking, like the other spoken keys.
        await self.inject_keys("Escape", explicit=True)
