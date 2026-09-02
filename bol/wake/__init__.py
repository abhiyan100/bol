"""The trigger words: "type", "send it", "hey Bol", and the ones that stop it.

On by default, and here is the honest version of what that costs. Wake mode
keeps the microphone open and runs a small keyword model on your Mac; nothing
is recorded or sent anywhere; expect the occasional false trigger from TV or
conversation, and "type" in particular also fires inside "what type of file"
and "the prototype". A false trigger costs a Listening pill and a paste that
waits; nothing reaches Claude until a send phrase presses Enter. Turn the wifi
off and it still works.

Four kinds of trigger word, and the daemon does something different with each:

  WAKE   "hey bol"        a recording with the usual auto-send rules
  TYPE   "type"           dictation: pasted after a pause, never submitted
  SEND   "send it"        Enter on what is already pasted; no recording
  CANCEL "scratch that"   wipe what is already pasted; no recording
  SLEEP  "stop listening" stop hearing trigger words until the next keypress

The child spotter knows none of that. It is handed spellings and reports which
one matched; the kind is this module's table and the daemon's to act on.

This module is the daemon's half: it owns the child process, feeds it the
blocks the microphone is already producing, and turns a line of its stdout
into a callback. Nothing here imports sherpa-onnx or onnxruntime, so the
daemon process never loads the keyword model or its runtime.

Two more pieces live here because they are wake's rules and the daemon just
applies them: which spellings of a phrase count as the same trigger (the
speech model hears "hey bowl" and "hey ball" at least as often as "hey bol"),
and how the trigger word is taken back off the front of the transcript before
the grammar sees it.
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
    "CANCEL",
    "DISK_BYTES",
    "DOWNLOAD_BYTES",
    "FRAME_BYTES",
    "FRAME_DTYPE",
    "FRAME_SAMPLES",
    "KINDS",
    "MODEL_NAME",
    "MODEL_URL",
    "SEND",
    "SLEEP",
    "SPELLINGS",
    "TYPE",
    "WAKE",
    "all_spellings",
    "UNMUTE_DELAY_S",
    "download_model",
    "human_size",
    "keyword_map",
    "lead_phrases",
    "missing_files",
    "model_dir",
    "model_files",
    "model_present",
    "parse_wake_line",
    "spellings",
    "strip_wake_phrase",
    "trigger_phrases",
    "wake_available",
]

# What a trigger word means. The child reports a spelling; the daemon looks it
# up here and does one of five very different things with it.
WAKE, TYPE, SEND, CANCEL, SLEEP = "wake", "type", "send", "cancel", "sleep"

# Kind -> the WakeConfig field holding its phrases, in the order a spelling
# claimed by two kinds is resolved: whoever comes first keeps it. WAKE leads
# because "hey bol" is the phrase people configure; SLEEP trails because it is
# the one whose default a [commands] list is most likely to widen.
KINDS = (
    (WAKE, "phrases", None),
    (TYPE, "type_phrases", None),
    (SEND, "send_phrases", "send"),
    (CANCEL, "cancel_phrases", None),
    (SLEEP, "sleep_phrases", "sleep"),
)

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


# ------------------------------------------------------------------ triggers


def _clean(value) -> list[str]:
    """A configured phrase list, lower-cased, de-duplicated, blanks dropped.

    Everything that reads config comes through here, so a stray "  Type  " in
    a TOML file is the same keyword as "type" rather than a second one the
    daemon's phrase table would never match.
    """
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        phrase = " ".join(str(item).lower().split())
        if phrase and phrase not in out:
            out.append(phrase)
    return out


def _untouched(field: str, phrases: list[str]) -> bool:
    """Is this list still the default nobody edited?

    Imported here rather than at module scope: bol.config is cheap, but this
    file is imported by `bol doctor` on machines with no wake model and no
    reason to build a Config.
    """
    from ..config import WakeConfig

    return phrases == _clean(getattr(WakeConfig(), field, None))


def trigger_phrases(cfg, commands=None) -> dict[str, list[str]]:
    """kind -> the phrases to listen for, as a person would say them.

    A [commands] list wins over a wake default the user never touched, so
    remapping send to "ship it" in the grammar remaps the trigger word with
    it and the two can never disagree about what sending is called. A wake
    list the user did edit is theirs, and is left exactly as written.
    """
    out: dict[str, list[str]] = {}
    for kind, field, command in KINDS:
        phrases = _clean(getattr(cfg, field, None))
        if command and _untouched(field, phrases):
            override = _clean((commands or {}).get(command))
            if override:
                phrases = override
        out[kind] = phrases
    return out


def keyword_map(cfg, commands=None) -> dict[str, str]:
    """spelling -> kind, in the order the child is given the keywords.

    One spelling is one keyword however many kinds claim it, and the first
    kind in KINDS keeps it: the child reports a spelling and nothing else, so
    a spelling that meant two things would be a coin toss at runtime.
    """
    out: dict[str, str] = {}
    for kind, phrases in trigger_phrases(cfg, commands).items():
        for spelling in all_spellings(phrases):
            out.setdefault(spelling, kind)
    return out


def lead_phrases(cfg, commands=None) -> list[str]:
    """One phrase per kind, as written: what `bol doctor` tells you to say.

    Not every spelling. "hey bowl" is in the keyword file because the decoder
    produces it, not because anyone should be told to say it.
    """
    phrases = trigger_phrases(cfg, commands)
    return [phrases[kind][0] for kind, _f, _c in KINDS if phrases.get(kind)]


def keyword_args(cfg, commands=None) -> list[str]:
    """The --phrase arguments for the child, per-phrase threshold and all.

    Only "type" can carry its own threshold, and only when someone set one:
    measured against `say`, raising it costs the real "type add a login test"
    before it costs the "type" inside "prototype", so the default leaves every
    keyword on the shared threshold.
    """
    try:
        type_threshold = float(getattr(cfg, "type_threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        type_threshold = 0.0
    args: list[str] = []
    for spelling, kind in keyword_map(cfg, commands).items():
        if kind == TYPE and type_threshold > 0.0:
            args += ["--phrase", f"{spelling}={type_threshold:g}"]
        else:
            args += ["--phrase", spelling]
    return args


def wake_available() -> bool:
    """Is the wake extra importable? Checked without importing it.

    find_spec does not execute the module, which is the point: onnxruntime
    must never be loaded into the daemon process.
    """
    return find_spec("sherpa_onnx") is not None and find_spec("sentencepiece") is not None


def parse_wake_line(line: str) -> tuple[float, str] | None:
    """One line of the child's stdout as (threshold, phrase), or None.

    The phrase is what the daemon acts on: it is the spelling that matched,
    and the only thing that says whether this was "type", "send it" or "hey
    bol". A line with no phrase on it is still a detection, and is treated as
    a plain wake, because that is what the format used to mean.

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
        score = float(parts[1])
    except ValueError:
        return None
    return score, " ".join(parts[2:]).strip().lower()


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
        commands=None,
        model_root: Path | None = None,
        spawn=None,
        clock=time.monotonic,
    ) -> None:
        self.cfg = cfg
        self._commands = commands or {}
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
                    "[wake] enabled = true, but sherpa-onnx is not installed. "
                    "Reinstall Bol with: %s",
                    install_hint("stt,llm"),
                )
                return False
            if not model_present(self._root):
                # Info, not a warning, and one line: wake is on by default
                # now, so the very first `bol run` on a machine that has not
                # been set up would otherwise open with a scolding about a
                # feature the user never asked for. The hotkey works.
                log.info(
                    "wake: keyword model not downloaded, run `bol setup`; "
                    "hotkey still works"
                )
                log.debug("missing from %s: %s", self._root, ", ".join(missing_files(self._root)))
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
            "listening for: %s (threshold %.2f)",
            ", ".join(lead_phrases(self.cfg, self._commands)) or "none",
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
        ] + keyword_args(self.cfg, self._commands)
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
            hit = parse_wake_line(line.decode("utf-8", "replace"))
            if hit is None:
                continue
            if self.muted:
                # Bol started speaking while this line was in the pipe.
                continue
            score, phrase = hit
            try:
                self._on_wake(score, phrase)
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
