"""The pill: a small always-on-top capsule that says what Bol is doing.

The window lives in a child process on purpose. AppKit in the daemon would
drag a run loop and an app identity into a process whose whole job is to stay
out of the way: the daemon must never become the frontmost app, or the paste
lands in the wrong window. So the daemon owns a pipe, not a window.

The client below is the daemon's half. It never raises and never blocks the
event loop: a dead child costs one skipped line, not a wedged hotkey.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from ..config import CONFIG_DIR
from .render import (
    DEFAULTS,
    DOT_COUNT,
    DOTS,
    HOLD_S,
    STATES,
    Dots,
    Update,
    animated,
    clamp_level,
    dot_alphas,
    dots_for,
    draft_span,
    hold_for,
    label_for,
    listening_dots,
    parse_line,
    render,
    tool_line,
    truncate_middle,
)

__all__ = [
    "Hud",
    "Dots",
    "Update",
    "DEFAULTS",
    "DOTS",
    "DOT_COUNT",
    "HOLD_S",
    "STATES",
    "animated",
    "clamp_level",
    "dot_alphas",
    "dots_for",
    "draft_span",
    "hold_for",
    "label_for",
    "listening_dots",
    "parse_line",
    "render",
    "tool_line",
    "truncate_middle",
]

log = logging.getLogger("bol.hud")

# A child that keeps dying gets retried at most this often. The pill is
# decoration; a respawn loop competing with the mic is not.
RESPAWN_S = 60.0
POSITIONS = ("top", "bottom")
# How often the microphone level is allowed down the pipe. Fast enough that
# the meter looks like a voice and not a progress bar, slow enough that a
# decoration never becomes the busiest thing on the event loop.
LEVEL_HZ = 15.0


class Hud:
    """Client for the pill child process.

    `set()` is synchronous by design: it is called from the hotkey callback,
    where the pill has to appear on the keystroke rather than after whatever
    the loop gets to next.
    """

    def __init__(
        self,
        enabled: bool = True,
        position: str = "top",
        text: bool = False,
        log_path: Path | None = None,
        spawn=None,
        clock=time.monotonic,
    ) -> None:
        self.enabled = bool(enabled)
        self.position = position if position in POSITIONS else "top"
        # [ui] text: whether the child draws the current line beside the dots.
        self.text = bool(text)
        self.log_path = Path(log_path) if log_path else CONFIG_DIR / "hud.log"
        # Injected in tests; None means "really start a child process".
        self._spawn = spawn
        self._clock = clock
        self._proc = None
        self._log_file = None
        self._started = False
        self._stopped = False
        self._last_respawn: float | None = None
        self._warned = False
        self._tasks: set = set()
        # The last line sent, so a level can refresh the meter without
        # wiping the words the live decoder put on the same line.
        self._state = ""
        self._text = ""
        self._detail = ""
        self._level_at: float | None = None
        # How long a child gets to exit after its stdin closes, before it is
        # killed. Shutdown must not hang on a stuck window.
        self._close_timeout = 2.0

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if not self.enabled or self._stopped or self._started:
            return
        self._started = True
        await self._launch()

    async def stop(self) -> None:
        self._stopped = True
        await self.idle()
        await self._close_proc()
        self._close_log()

    async def idle(self) -> None:
        """Wait for queued drains and respawns to finish."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------------ send

    def set(
        self,
        state: str,
        text: str = "",
        detail: str = "",
        hold: float = 0.0,
        level: float = 0.0,
    ) -> None:
        """Put one line on the pill. Never raises, never blocks.

        hold overrides how long a transient state stays up, for this line
        only: a sentence the user is meant to read needs longer than the one
        word the state table sizes its default for.
        """
        if not self.enabled or self._stopped or state not in STATES:
            return
        self._state, self._text, self._detail = state, text, detail
        # A new line restarts the meter's clock, so the first level after the
        # microphone opens reaches the pill at once rather than a frame late.
        self._level_at = None
        self._send(state, text, detail, hold, level)

    def level(self, value: float) -> None:
        """Refresh the listening meter. Throttled to LEVEL_HZ; never raises.

        This is a meter, not a message: it only ever refreshes a pill that is
        already listening. A level measured while Bol is thinking is dropped
        rather than dragging the pill back to a state it has left, which is
        what makes it safe to wire straight to the recorder.
        """
        if not self.enabled or self._stopped or self._state != "listening":
            return
        now = self._clock()
        if self._level_at is not None and now - self._level_at < 1.0 / LEVEL_HZ:
            return
        self._level_at = now
        self._send("listening", self._text, self._detail, 0.0, value)

    def _send(
        self, state: str, text: str, detail: str, hold: float, level: float
    ) -> None:
        payload = {"state": state, "text": text, "detail": detail}
        if hold:
            # Sent only when there is something to override, so the ordinary
            # line down the pipe stays exactly what it always was.
            payload["hold"] = float(hold)
        meter = clamp_level(level)
        if meter:
            payload["level"] = round(meter, 3)
        line = json.dumps(payload) + "\n"
        if not self._write(line):
            self._respawn(line)

    def _write(self, line: str) -> bool:
        stdin = getattr(self._proc, "stdin", None)
        if stdin is None:
            return False
        try:
            stdin.write(line.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError, RuntimeError):
            return False
        self._track(self._drain(stdin))
        return True

    async def _drain(self, stdin) -> None:
        drain = getattr(stdin, "drain", None)
        if drain is None:
            return
        try:
            await drain()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError, RuntimeError):
            # The child is gone. The next set() notices on its own write and
            # decides whether a respawn is due.
            pass

    # --------------------------------------------------------------- process

    async def _launch(self) -> None:
        try:
            self._proc = await self._open()
        except Exception as exc:  # a missing python, a read-only log dir, ...
            self._proc = None
            log.debug("could not start the pill: %s", exc)
            self._unavailable()

    async def _open(self):
        if self._spawn is not None:
            return await self._spawn()
        argv = [sys.executable, "-m", "bol.hud.app", "--position", self.position]
        if self.text:
            argv.append("--text")
        return await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=self._stderr(),
        )

    def _stderr(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "a", buffering=1)
        except OSError:
            return asyncio.subprocess.DEVNULL
        return self._log_file

    def _close_log(self) -> None:
        handle, self._log_file = self._log_file, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def _respawn(self, line: str) -> None:
        """Bring the child back, at most once a minute, and resend the line."""
        if not self._started or self._stopped:
            return
        now = self._clock()
        if self._last_respawn is not None and now - self._last_respawn < RESPAWN_S:
            self._unavailable()
            return
        self._last_respawn = now
        loop = self._loop()
        if loop is None:
            return
        self._track(self._respawn_and_write(line))

    async def _respawn_and_write(self, line: str) -> None:
        await self._close_proc()
        await self._launch()
        self._write(line)

    async def _close_proc(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        stdin = getattr(proc, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except (BrokenPipeError, ConnectionResetError, OSError, ValueError, RuntimeError):
                pass
        wait = getattr(proc, "wait", None)
        if wait is None:
            return
        try:
            await asyncio.wait_for(wait(), timeout=self._close_timeout)
        except Exception:
            kill = getattr(proc, "kill", None)
            if kill is not None:
                try:
                    kill()
                except (OSError, ProcessLookupError):
                    pass

    # ----------------------------------------------------------------- misc

    def _loop(self):
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def _track(self, coro) -> None:
        """Run a coroutine in the background, keeping a reference to it."""
        loop = self._loop()
        if loop is None:
            coro.close()
            return
        task = loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _unavailable(self) -> None:
        if self._warned:
            return
        self._warned = True
        log.info("pill unavailable, continuing without it")
