"""Opt-in "hey Bol": a wake phrase that starts a recording without the hotkey.

Off by default, and here is the honest version of what turning it on costs.
Wake mode keeps the microphone open and runs a small keyword model on your
Mac; nothing is recorded or sent anywhere; expect the occasional false wake
from TV or conversation; a false wake costs a Listening pill and nothing is
sent unless you say three words. Turn the wifi off and it still works.

This module is the daemon's half: it owns the child process, feeds it the
blocks the microphone is already producing, and turns a line of its stdout
into a callback. Nothing here imports sherpa-onnx or onnxruntime, so a daemon
built without the wake extra pays nothing for this file existing.

Two more pieces live here because they are wake's rules and the daemon just
applies them: which spellings of a phrase count as the same wake (the speech
model hears "hey bowl" and "hey ball" at least as often as "hey bol"), and how
the phrase is taken back off the front of the transcript before the grammar
sees it.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from importlib.util import find_spec
from math import inf
from pathlib import Path

import numpy as np

from .. import install_hint
from .listener import FRAME_BYTES, FRAME_DTYPE, FRAME_SAMPLES, READY
from .model import (
    DISK_BYTES,
    DOWNLOAD_BYTES,
    MODEL_NAME,
    MODEL_URL,
    download_model,
    human_size,
    missing_files,
    model_dir,
    model_files,
    model_present,
)

log = logging.getLogger("bol.wake")

__all__ = [
    "WakeListener",
    "DISK_BYTES",
    "DOWNLOAD_BYTES",
    "FRAME_BYTES",
    "FRAME_DTYPE",
    "FRAME_SAMPLES",
    "MODEL_NAME",
    "MODEL_URL",
    "SPELLINGS",
    "all_spellings",
    "UNMUTE_DELAY_S",
    "download_model",
    "human_size",
    "missing_files",
    "model_dir",
    "model_files",
    "model_present",
    "parse_wake_line",
    "spellings",
    "strip_wake_phrase",
    "wake_available",
]

# Bol speaks, and 500 ms later the wake listener is allowed to hear again.
# Without the tail the room's reverb of Bol's own last word is still arriving
# when the frames start flowing.
UNMUTE_DELAY_S = 0.5

# Roughly a second of audio. Past that the child is not merely behind, it is
# gone, and a wake a second late is worse than no wake: it lands on whatever
# the user is saying now.
QUEUE_FRAMES = 32

# How long the child gets to load the keyword model and say READY. The int8
# model is about 5 MB, so this is generous by a factor of ten and only ever
# spent when something is wrong.
READY_TIMEOUT_S = 20.0

# Spellings of one wake phrase. A speech model that has never seen "bol"
# spells it out of the words it does know, so all three have to be keywords
# and all three have to come back off the front of the transcript. Anything
# not in here is used exactly as the user wrote it.
SPELLINGS = {"hey bol": ("hey bol", "hey bowl", "hey ball")}


def spellings(phrase: str) -> tuple[str, ...]:
    """Every spelling that counts as this wake phrase."""
    key = " ".join(phrase.lower().split())
    return SPELLINGS.get(key, (key,) if key else ())


def all_spellings(phrases) -> list[str]:
    """Every spelling of every configured phrase, in the order a human would
    read them out: the phrase as written, then the ways it gets misheard."""
    out: list[str] = []
    for phrase in phrases or ():
        for spelling in spellings(phrase):
            if spelling not in out:
                out.append(spelling)
    return out


def wake_available() -> bool:
    """Is the wake extra importable? Checked without importing it.

    find_spec does not execute the module, which is the point: onnxruntime
    must never be loaded into the daemon process.
    """
    return find_spec("sherpa_onnx") is not None and find_spec("sentencepiece") is not None


def parse_wake_line(line: str) -> float | None:
    """Read one line of the child's stdout, or None if it is not a detection.

    The child is another process writing down a pipe; a partial line, a
    library's stray print, or a version that grew a new field all have to be
    ignored rather than raise on the reader task.
    """
    if not isinstance(line, str):
        return None
    parts = line.strip().split()
    if len(parts) < 2 or parts[0] != "wake":
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def strip_wake_phrase(text: str, phrases=("hey bol",)) -> str:
    """Take the wake phrase back off the front of a transcript.

    Only off the front, and only once: "hey bol, remind me to say hey bol"
    keeps its second one, because that one is what the user dictated. The
    comma is optional on both sides of the phrase, since the transcriber
    punctuates however it feels.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    # Longest first: the alternation matches in order, so with "hey bol"
    # ahead of "hey bowl" the transcript "hey bowl run it" would lose its
    # "bol" and keep a stray "l".
    words = sorted(all_spellings(phrases), key=len, reverse=True)
    if not words:
        return text.strip()
    alternatives = "|".join(
        r"[\s,]+".join(re.escape(word) for word in spelling.split())
        for spelling in words
    )
    pattern = re.compile(rf"^\s*(?:{alternatives})(?!\w)[\s,.!?:;-]*", re.IGNORECASE)
    return pattern.sub("", text, count=1).strip()


def _frame(block) -> bytes:
    """One audio block as the exact bytes the child expects.

    PortAudio hands back full blocks, but a stream stopped mid-block can give
    a short one, and the child reads fixed-size frames.
    """
    data = np.asarray(block, dtype=FRAME_DTYPE).reshape(-1)
    if data.size != FRAME_SAMPLES:
        fitted = np.zeros(FRAME_SAMPLES, dtype=FRAME_DTYPE)
        fitted[: min(data.size, FRAME_SAMPLES)] = data[:FRAME_SAMPLES]
        data = fitted
    return data.tobytes()


class WakeListener:
    """The daemon's handle on the keyword listener child.

    Every method is safe to call whether or not the child is running: wake is
    a convenience on top of a hotkey that already works, and a listener that
    failed to start must cost one log line, not a dead daemon.
    """

    def __init__(
        self,
        cfg,
        on_wake,
        *,
        model_root: Path | None = None,
        spawn=None,
        clock=time.monotonic,
    ) -> None:
        self.cfg = cfg
        self._on_wake = on_wake
        self._root = Path(model_root) if model_root is not None else model_dir()
        # Injected in tests; None means "really start a child process".
        self._spawn = spawn
        self._clock = clock
        self._proc = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frames: asyncio.Queue | None = None
        self._tasks: list[asyncio.Task] = []
        self._mute_until = 0.0
        self._stopping = False
        self._last_error = ""

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> bool:
        """Bring the child up. False (with a reason logged) if it cannot."""
        if self._spawn is None:
            if not wake_available():
                log.warning(
                    "[wake] enabled = true, but the wake extra is not installed. "
                    "Install it with: %s",
                    install_hint("stt,llm,wake"),
                )
                return False
            if not model_present(self._root):
                log.warning(
                    "[wake] enabled = true, but the keyword model is missing "
                    "from %s (%s). Run `bol setup`.",
                    self._root,
                    ", ".join(missing_files(self._root)),
                )
                return False
        try:
            self._proc = await self._open()
        except Exception as exc:  # noqa: BLE001 - a missing python, a bad path
            log.warning("could not start the wake listener: %s", exc)
            return False
        self._loop = asyncio.get_running_loop()
        self._frames = asyncio.Queue()
        self._spawn_task(self._read_errors())
        if not await self._await_ready():
            await self.stop()
            return False
        self._spawn_task(self._pump_frames())
        self._spawn_task(self._read_wakes())
        log.info(
            "wake phrase armed: %s (threshold %.2f)",
            ", ".join(all_spellings(self.cfg.phrases)) or "none",
            float(self.cfg.threshold),
        )
        return True

    def _spawn_task(self, coro) -> None:
        """Run one of the listener's own loops, keeping a handle to cancel."""
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.append(task)
        task.add_done_callback(self._finished)

    def _finished(self, task) -> None:
        """Collect a loop's exception here rather than let asyncio report it
        at shutdown, long after wake went quietly deaf."""
        if task in self._tasks:
            self._tasks.remove(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.warning("the wake listener stopped: %s", exc)

    async def _open(self):
        if self._spawn is not None:
            return await self._spawn()
        argv = [
            sys.executable,
            "-m",
            "bol.wake",
            "--model-dir",
            str(self._root),
            "--threshold",
            str(float(self.cfg.threshold)),
        ]
        for phrase in all_spellings(self.cfg.phrases):
            argv += ["--phrase", phrase]
        return await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _await_ready(self) -> bool:
        stdout = getattr(self._proc, "stdout", None)
        if stdout is None:
            return False
        try:
            line = await asyncio.wait_for(stdout.readline(), timeout=READY_TIMEOUT_S)
        except (asyncio.TimeoutError, OSError, ValueError) as exc:
            log.warning("the wake listener did not start (%s)", exc)
            return False
        if line.decode("utf-8", "replace").strip() != READY:
            reason = self._last_error or "it exited before it was ready"
            log.warning("the wake listener did not start: %s", reason)
            return False
        return True

    async def stop(self) -> None:
        self._stopping = True
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        proc, self._proc = self._proc, None
        self._frames = None
        if proc is None:
            return
        stdin = getattr(proc, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
                pass
        wait = getattr(proc, "wait", None)
        if wait is None:
            return
        try:
            await asyncio.wait_for(wait(), timeout=2.0)
        except Exception:  # noqa: BLE001 - a wedged child is killed, not awaited
            kill = getattr(proc, "kill", None)
            if kill is not None:
                try:
                    kill()
                except (OSError, ProcessLookupError):
                    pass

    @property
    def running(self) -> bool:
        return self._proc is not None

    # ------------------------------------------------------------------ mute

    def mute(self) -> None:
        """Stop forwarding audio. Bol is about to speak, or is recording."""
        self._mute_until = inf

    def unmute(self, delay: float = UNMUTE_DELAY_S) -> None:
        """Hear again, after a tail long enough to miss Bol's own last word."""
        self._mute_until = self._clock() + max(0.0, delay)

    @property
    def muted(self) -> bool:
        return self._clock() < self._mute_until

    # ----------------------------------------------------------------- audio

    def feed(self, block) -> None:
        """One block from the audio callback. PortAudio's thread, not the loop.

        Never raises and never blocks: this runs inside the microphone
        callback, where a slow line is a dropout in the recording the user
        actually asked for.
        """
        loop, frames = self._loop, self._frames
        if loop is None or frames is None or self.muted:
            return
        if frames.qsize() >= QUEUE_FRAMES:
            return  # the child is behind; a stale wake is worse than no wake
        try:
            loop.call_soon_threadsafe(frames.put_nowait, _frame(block))
        except RuntimeError:
            pass  # the loop is closing

    async def _pump_frames(self) -> None:
        # Held locally: stop() drops the queue, and this task is cancelled
        # rather than left to trip over the None on its next lap.
        frames = self._frames
        while frames is not None:
            frame = await frames.get()
            stdin = getattr(self._proc, "stdin", None)
            if stdin is None:
                return
            try:
                stdin.write(frame)
                drain = getattr(stdin, "drain", None)
                if drain is not None:
                    await drain()
            except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
                return  # the child is gone; _read_errors reports why

    async def _read_wakes(self) -> None:
        stdout = getattr(self._proc, "stdout", None)
        while stdout is not None:
            line = await stdout.readline()
            if not line:
                return
            score = parse_wake_line(line.decode("utf-8", "replace"))
            if score is None:
                continue
            if self.muted:
                # Bol started speaking while this line was in the pipe.
                continue
            try:
                self._on_wake(score)
            except Exception as exc:  # noqa: BLE001 - a callback must not end the reader
                log.warning("wake callback failed: %s", exc)

    async def _read_errors(self) -> None:
        stderr = getattr(self._proc, "stderr", None)
        while stderr is not None:
            line = await stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            self._last_error = text
            # sherpa-onnx narrates its whole config on startup, so this is
            # debug; the one line that matters is reported by whoever noticed
            # the child was not ready.
            log.debug("wake listener: %s", text)
