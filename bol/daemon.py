"""Bol daemon: the loop.

hold the key / a trigger word → record → transcribe → parse command → paste
                                                                        ↓
        pill (and, with talk_back, a spoken summary) ← Stop hook ← Claude

Dictated text is pasted where the cursor is as soon as the recording ends, so
the user watches it appear exactly like typing, and the pill goes away the
moment the words land. Nothing is ever submitted by itself: every Enter is the
user saying so, either as a send phrase ("send it", in the recording or as a
trigger word) or as the answer to a permission prompt.

There is also no key at all, by default. The keyword listener is up from the
moment Bol starts, and four trigger words drive the whole loop without one:
"type" starts dictation that is pasted after a two second pause, "send it"
presses Enter on what is already pasted, "scratch that" wipes it, and
"hey Bol" starts the conversation flow. "Stop listening" mutes the trigger
words until the next keypress. See bol/wake for what that costs and how it is
kept off the daemon's own process.

Three things, and only three, put the pill on screen: a trigger word, the
hotkey, and a question Bol asked out loud. Speech alone never does. The awake
window ([wake] awake_s) is off by default for exactly that reason: with it
open, room noise kept reopening the microphone and the pill kept coming back.

talk_back decides whether anything comes back out loud, and it decides more
than that. Off (the default) Bol is pure dictation: no speaker, no summarizer,
no LLM server, no hook server, no hooks, and nothing on screen or in the
terminal about a coding agent at all. On, the reply is spoken, the agent's
hooks drive the pill, and "hey Bol" has someone to talk to.

While the user is still talking the pill shows the words as they are decoded.
That path is display only and stops at the pill: the text that reaches Claude
is always the one full-buffer decode taken after the recording ends.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
import time
from pathlib import Path

from .audio import Recorder
from .audio.capture import CANCELLED
from .bridge import BridgeError, build_bridge, frontmost_bundle_id
from .bridge.focused import SubmitBlocked
from .cleanup import CLEANUP_SYSTEM, build_cleaner, clean_transcript
from .config import Config, hook_token, validate_config
from .grammar import Action, Grammar
from .hooks import HookServer, TurnTracker, display_name
from .hotkey import HotkeyListener, MouseListener
from .hud import Hud, tool_line
from .llm import LLMEngine
from .speak import build_speaker, play_cue
from .stt import STREAM_END, build_transcriber
from .summarize import build_summarizer
from .wake import (
    CANCEL,
    SEND,
    SLEEP,
    TYPE,
    WAKE,
    WakeListener,
    keyword_map,
    lead_phrases,
    strip_wake_phrase,
    trigger_phrases,
)

log = logging.getLogger("bol")


def _drain(task: asyncio.Task) -> None:
    """Collect a fire-and-forget task's exception.

    Without this a failure surfaces only as asyncio's "Task exception was
    never retrieved" on shutdown, long after the feature went quietly dead.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning("background task failed: %s", exc)


# What one recording left behind, and therefore whether the microphone stays.
#   CHAIN: words were handled and the conversation is still Bol's to continue.
#   QUIET: nobody said anything worth keeping. The mic may reopen, but only
#          because the awake window says so.
#   STOP:  the turn left Bol. Claude has it, or Bol went to sleep, or the
#          capture failed and reopening would be a spin loop.
CHAIN, QUIET, STOP = "chain", "quiet", "stop"


# How often the pill may be redrawn with streamed words. Past about four a
# second a line of text reads as flicker, and every redraw is a write down a
# pipe to another process.
LIVE_PILL_HZ = 4.0

# A pill line the user is meant to read rather than a receipt, and how long it
# stays: long enough for a sentence, short enough to be gone before the next
# thing said. A paste no longer gets one. The words appearing in the box is
# the whole message, and a blue capsule sitting there afterwards was the thing
# people asked us to take away.
HINT_S = 2.5

# The pause is the one state with no way out except the key. Saying so is the
# difference between a paused Bol and a broken one.
SLEEP_HINT = "Paused. Press the key to resume"

# How often a recording a trigger word started checks that the user is still
# looking at the same app. Every check is an osascript, and the thing being
# caught (walking away from a microphone that opened itself) does not need
# to be caught in under half a second.
FRONTMOST_POLL_S = 0.5


class _LiveWords:
    """Streamed partials on their way to the pill.

    The decoder reports after every step, and after a stall it reports several
    in a row, so the updates are rate limited with the last one always
    delivered. Everything after close() is dropped: that recording is over and
    the next line on the pill belongs to Finalizing.
    """

    def __init__(self, hud, loop, interval: float = 1.0 / LIVE_PILL_HZ) -> None:
        self.hud = hud
        self.blocks: queue.Queue | None = None
        self._loop = loop
        self._interval = interval
        self._last: float | None = None
        self._pending: tuple[str, str] | None = None
        self._timer = None
        self._closed = False

    def emit(self, committed: str, draft: str) -> None:
        """Called from the decoder, already hopped onto the event loop."""
        if self._closed:
            return
        now = self._loop.time()
        if self._last is not None and now - self._last < self._interval:
            self._pending = (committed, draft)
            if self._timer is None:
                self._timer = self._loop.call_later(
                    self._interval - (now - self._last), self._flush
                )
            return
        self._show(committed, draft)

    def _flush(self) -> None:
        self._timer = None
        pending, self._pending = self._pending, None
        if pending is not None and not self._closed:
            self._show(*pending)

    def _show(self, committed: str, draft: str) -> None:
        self._last = self._loop.time()
        if not committed and not draft:
            return  # nothing decoded yet, so leave "Listening" where it is
        self.hud.set("listening", committed, draft)

    def close(self) -> None:
        """End the decode and stop writing to the pill. Safe to call twice."""
        self._closed = True
        self._pending = None
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        blocks, self.blocks = self.blocks, None
        if blocks is not None:
            blocks.put_nowait(STREAM_END)


class Daemon:
    def __init__(self, cfg: Config, text_mode: bool = False, clock=time.monotonic) -> None:
        # Fail here, with the list of valid values, rather than starting a
        # daemon on a setting nothing reads the way the file meant it.
        validate_config(cfg)
        self.cfg = cfg
        self.text_mode = text_mode
        # One-way (the default) or two-way. Everything that says something
        # back is built only in two-way mode, so one-way loads no voice, no
        # summarizer and no LLM server at all: they are not switched off at
        # the point of use, they are not there.
        self.talk_back = bool(cfg.talk_back)
        self.bridge = build_bridge(cfg)
        self.grammar = Grammar(cfg.commands)
        self.tracker = TurnTracker()
        # The hook server exists only to hear from a coding agent, and in
        # one-way mode Bol has nothing to say about one. Not started, not
        # bound, not even constructed: no port, no token file, no /health,
        # and no way for a hook to put an agent's name on the pill.
        self.server = (
            HookServer(
                cfg.server.host,
                cfg.server.port,
                hook_token(),
                allow_remote=cfg.server.allow_remote,
            )
            if self.talk_back
            else None
        )
        self.speaker = build_speaker(cfg) if self.talk_back else None
        self.engine = LLMEngine(cfg) if self.talk_back else None
        # Both modes: the cleanup model is part of getting the dictation
        # right, not part of talking back.
        self.cleaner = build_cleaner(cfg)
        self.summarizer = build_summarizer(cfg, self.engine) if self.talk_back else None
        self.recorder = Recorder(cfg.audio)
        self.transcriber = None if text_mode else build_transcriber(cfg)
        self.hotkey: HotkeyListener | None = None
        # Watches for the click that means "not that". Optional, like wake.
        self.mouse: MouseListener | None = None
        # The recording a trigger word started, while it is running: the one
        # a click or an app switch is allowed to cancel.
        self._wake_session = None
        # Why it was cancelled, so the loop can print one line about it
        # instead of a log line and then a second, vaguer print.
        self._cancel_reason = ""
        # Wake mode, when [wake] enabled = true and the extra is installed.
        # None the rest of the time, and every wake path checks for it.
        self.wake: WakeListener | None = None
        # Monotonic seconds; injected so the awake window can be tested
        # without a test that sleeps for a minute.
        self._clock = clock
        self._awake_until = 0.0
        # spelling -> kind for every trigger word, built here rather than when
        # the listener starts so the daemon can answer "what does this phrase
        # mean" whether or not a child process ever came up.
        self._wake_kinds = keyword_map(cfg.wake, cfg.commands)
        # Every spelling that has to come back off the front of a transcript a
        # "type" started: the trigger word is not part of what was dictated.
        self._type_phrases = list(cfg.wake.phrases or ()) + trigger_phrases(
            cfg.wake, cfg.commands
        ).get(TYPE, [])
        # Text Bol pasted and did not submit. A send phrase presses Enter on
        # it; without one there is nothing to send, and "send it" said at the
        # television must not press Enter on whatever the user typed by hand.
        self._pending_paste = False
        # The on-screen pill. Inert until start(), and every call on it is a
        # no-op when the child is missing.
        self.hud = Hud(
            enabled=cfg.ui.pill, position=cfg.ui.position, text=cfg.ui.text
        )

        self._listen_lock = asyncio.Lock()
        # Hook handlers run as independent tasks; without this a Stop and a
        # Notification landing together cut each other off mid-sentence.
        self._speak_lock = asyncio.Lock()
        self._pending_listen = False
        self._active_session = None
        # Whether the recording in flight is one that opened itself (a trigger
        # word, the awake window). Those yield to the key; a hold does not.
        self._active_auto = False
        self._ptt_session = None
        self._asleep = False
        self._last_reply = ""
        # Agent hooks are user-scoped, so every session on this machine posts
        # here. Bol latches onto one (see [server] follow).
        self._bound_session: str | None = None
        self._bound_cwd = ""
        self._warned_other_session = False
        # Which session raised the permission prompt we're waiting on, so a
        # spoken "yes" can never approve a different session's prompt.
        self._permission_session: str | None = None
        # Which agent Bol is narrating, and therefore the name every sentence
        # built at runtime uses. Claude Code until a hook event says Codex;
        # the session it follows is the one that decides.
        self._agent = "claude"

    # ---------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        target = await self.bridge.attach()
        print(f"bol: injecting into {target}")

        if self.server is not None:
            self.server.on("Stop", self._on_stop)
            self.server.on("PostToolUse", self._on_tool)
            self.server.on("Notification", self._on_notification)
            # Codex's stand-in for Notification. One endpoint, one daemon:
            # both agents can be wired at once and the payload says which
            # spoke.
            self.server.on("PermissionRequest", self._on_permission_request)
            await self.server.start()
        loop = asyncio.get_running_loop()
        # Two-way only, and in the background; template/raw fallbacks cover
        # the gap (and the first run's model download).
        if self.engine is not None:
            loop.create_task(self.engine.start()).add_done_callback(_drain)
        # Both modes, because dictation is cleaned in both: cold, the first
        # pass spends its whole deadline loading weights and hands back the
        # text unchanged.
        warmup = getattr(self.cleaner, "warmup", None)
        if callable(warmup):
            loop.create_task(warmup()).add_done_callback(_drain)
        if self.server is not None:
            print(
                "bol: hook server on "
                f"http://{self.cfg.server.host}:{self.cfg.server.port}/hook"
            )
        print(self.mode_line())

        if self.transcriber is not None and not self.text_mode:
            # Started before the warmup rather than after it. Loading the
            # speech model is the longest silence in a Bol startup, and an
            # unexplained wait is what the pill exists to end. Either way it
            # is up before the key is armed, so the first press has a pill to
            # appear on.
            await self.hud.start()
        await self._warm_speech_model()

        if self.text_mode or self.transcriber is None:
            print("bol: text mode. Type what you'd say ('send it', 'type ...', 'close').")
            await self._text_console()
            return

        # The pill's level meter, straight off the recorder's own callback.
        # Hud.level throttles it and drops anything measured while the pill
        # is showing something other than listening.
        self.recorder.on_level = self.hud.level
        await self._open_microphone()
        await self._start_wake()

        self.hotkey = HotkeyListener(
            self.cfg.hotkey, self._hotkey_pressed, self._hotkey_released
        )
        self.hotkey.start()
        if self.wake is not None:
            # Started after the hotkey, and only when a trigger word can open
            # the microphone on its own: it exists to cancel those recordings
            # and has nothing to watch otherwise. Same permission, so it adds
            # no prompt, and it never raises.
            mouse = MouseListener(self._clicked)
            if mouse.start():
                self.mouse = mouse
        key = self.cfg.hotkey.key
        phrase = self._wake_phrase()
        print(f"bol: hold {key}{phrase} to talk. Ctrl+C to quit.")
        try:
            await asyncio.Event().wait()
        finally:
            self.hotkey.stop()
            if self.mouse is not None:
                self.mouse.stop()
            await self._stop_wake()
            await self.hud.stop()
            await self.recorder.close()
            if self.server is not None:
                await self.server.stop()
            if self.engine is not None:
                await self.engine.stop()

    @property
    def agent_name(self) -> str:
        """What Bol calls the agent it is narrating, out loud and on screen."""
        return display_name(self._agent)

    def _bind_agent(self, agent: str) -> None:
        """Remember which agent this event came from, for the words Bol uses."""
        if agent and agent != self._agent:
            log.debug("narrating %s", display_name(agent))
            self._agent = agent

    def mode_line(self) -> str:
        """The one line that says which half of Bol is running."""
        if self.talk_back:
            return "bol: two-way (talk-back on)."
        return (
            "bol: one-way (dictation). Add --talk-back to hear what "
            f"{self.agent_name} did."
        )

    async def _warm_speech_model(self) -> None:
        """Load the speech model, with the pill saying so.

        Cold, this is seconds of nothing: no window, no sound, and a hotkey
        that answers late. The pill says what the wait is for and clears
        itself however the load ends, so a failed warmup does not leave
        "Loading speech model" on screen forever.
        """
        if self.transcriber is None:
            return
        print("bol: warming up speech model ...")
        self.hud.set("thinking", "Loading speech model")
        try:
            await self.transcriber.warmup()
        finally:
            self.hud.set("idle")

    async def _open_microphone(self) -> None:
        """Build the input stream before the key is armed.

        Construction is the expensive half of opening a mic and it must not
        land on the press. When the configured device is gone (the headset is
        off, another app took it) Bol says which one and tries the system
        default once, because a built-in mic that works beats a named one
        that does not.
        """
        try:
            await self.recorder.open()
            return
        except Exception as exc:
            name = self.recorder.device_label
            log.warning("could not prepare the microphone (%s): %s", name, exc)
            self.hud.set("error", f"Mic lost: {name}")
        if not self.cfg.audio.input_device.strip():
            return  # already the default device; there is nothing to fall back to
        log.debug("retrying with the system default input device")
        self.recorder.use_default_device()
        try:
            await self.recorder.open()
        except Exception as exc:
            log.warning("the default input device did not open either: %s", exc)
        else:
            print("bol: using the system default microphone instead.")

    # --------------------------------------------------------------------- wake

    async def _start_wake(self) -> None:
        """Arm the trigger words, if they are switched on and able to run.

        Every way this can fail is one line and a daemon that carries on with
        the hotkey: a missing package, a model nobody has downloaded yet, a
        child that will not start. Trigger words are a second way in, and the
        first one has to keep working when they are not there.
        """
        if not self.cfg.wake.enabled or self.text_mode or self.transcriber is None:
            return
        listener = WakeListener(
            self.cfg.wake,
            self._wake_detected,
            commands=self.cfg.commands,
            clock=self._clock,
        )
        if not await listener.start():
            return
        try:
            # One microphone owner: the listener reads the blocks the daemon's
            # own stream is already producing, and that stream now stays up.
            await self.recorder.monitor(listener.feed)
        except Exception as exc:
            # No microphone to share, which _open_microphone has already said
            # its piece about. The hotkey is still worth arming.
            log.warning("wake mode needs the microphone and could not have it: %s", exc)
            await listener.stop()
            return
        self.wake = listener

    async def _stop_wake(self) -> None:
        listener, self.wake = self.wake, None
        if listener is None:
            return
        await self.recorder.monitor(None)
        await listener.stop()

    def _wake_phrase(self) -> str:
        """What to add to the startup line when the trigger words are armed."""
        if self.wake is None:
            return ""
        said = lead_phrases(self.cfg.wake, self.cfg.commands)[:2]
        if not said:
            return ""
        return ", or say " + " or ".join(f'"{phrase}"' for phrase in said) + ","

    def _wake_detected(self, score: float, phrase: str = "") -> None:
        """The child heard a trigger word. What happens next depends on which.

        "send it" and "scratch that" act on text that is already in Claude's
        box, so they start no recording at all: there is nothing left to say.
        "stop listening" stops the trigger words until the next keypress.
        "type" and "hey bol" both open the microphone; a type is dictation,
        with the longer pause of someone composing a prompt. Both are ignored
        while a recording is already running, because that recording is the
        answer to whatever is being said now.
        """
        if self.wake is None or self._asleep:
            return
        kind = self._wake_kinds.get(phrase, WAKE)
        if kind in (SEND, CANCEL, SLEEP):
            self._wake_command(kind, phrase)
            return
        if self._listen_lock.locked() or self._pending_listen:
            return
        # Debug, not info: a trigger word is a thing the user just said, and
        # narrating it back at them fills the terminal with lines nobody asked
        # for. "you: ..." is the line that says Bol heard them.
        log.debug("heard %r (%.2f)", phrase or "a trigger word", score)
        self.hud.set("listening", "Listening")
        self._touch_awake()
        self._prewarm()
        session = self._begin(kind)
        self._hush()
        listen = asyncio.get_running_loop().create_task(
            self._listen_session(session, until_silence=True, trigger=kind)
        )
        listen.add_done_callback(_drain)

    def _hush(self) -> None:
        """Barge-in: cut Bol off mid-sentence. Inert in one-way mode, where
        there is no voice to interrupt."""
        if self.speaker is None:
            return
        task = asyncio.get_running_loop().create_task(self.speaker.stop())
        task.add_done_callback(_drain)

    def _begin(self, trigger: str = ""):
        """A recording session, with the timings its trigger word implies.

        Both overrides only ever apply to a recording nobody pressed a key
        for. A hotkey recording waits the full listen_window_s for someone to
        start, because the user's hand on the key is already the statement
        that they mean to say something; a microphone that opened itself on a
        word heard across the room has to give up much sooner than that, and
        speak_window_ms is the number the user picked for exactly that.
        pause_ms is the other end: how long a pause in a dictation lasts
        before the words are pasted.
        """
        session = self.recorder.begin()
        if trigger:
            session.window_ms = int(self.cfg.wake.speak_window_ms)
        if trigger == TYPE:
            session.silence_ms = int(self.cfg.wake.pause_ms)
        return session

    def _wake_command(self, kind: str, phrase: str = "") -> None:
        """A trigger word that acts on text rather than starting a recording."""
        if kind == SLEEP:
            log.debug("heard %r, pausing", phrase or kind)
            self._go_to_sleep()
            return
        if not self._pending_paste:
            # Nothing of Bol's is sitting in the input box, so there is
            # nothing to send or to wipe. Silence is the right answer: the
            # alternative is pressing Enter on something the user typed.
            log.debug("heard %r with nothing pasted, ignored", phrase or kind)
            return
        task = asyncio.get_running_loop().create_task(self._run_wake_command(kind))
        task.add_done_callback(_drain)

    async def _run_wake_command(self, kind: str) -> None:
        try:
            if kind == SEND:
                self._permission_session = None
                # A trigger word is the user saying it, so this Enter goes
                # wherever they are looking, Notes and Slack included.
                await self._keys("Enter", explicit=True)
                self._pending_paste = False
                # Nothing on screen: the text left the box, which the user can
                # see. The chime is the whole receipt.
                self._idle_pill()
                self._cue("done")
                print("bol: sent.")
            else:
                await self._keys("C-u", explicit=True)
                self._pending_paste = False
                self._idle_pill()
                self._cue("discard")
                print("bol: scratched that.")
        except SubmitBlocked as exc:
            # The paste is still there and still unsent, so the flag stays up
            # and saying it again once the agent is in front will work.
            log.debug("submit withheld: %s", exc)
            await self._speak(
                f"That window doesn't look like {self.agent_name}, "
                "so I didn't press Enter."
            )
        except BridgeError as exc:
            await self._speak(
                f"Couldn't reach {self.agent_name}: {exc}", state="error"
            )

    def _go_to_sleep(self) -> None:
        """Stop hearing trigger words until the next hotkey press.

        The mute is what makes this a pause rather than a label: _asleep alone
        would leave the keyword model listening and burning a core for
        detections the daemon then throws away. _unmute_wake is a no-op while
        this flag is up, so the recording and the reply that were in flight
        when it was set cannot quietly switch the ear back on.
        """
        self._asleep = True
        # Or the next pause would reopen the mic Bol was just told to leave.
        self._awake_until = 0.0
        self._mute_wake()
        self.hud.set("sending", SLEEP_HINT, hold=HINT_S)
        print("bol: paused. Press the key to resume.")

    # ------------------------------------------------------------ cancelling

    def _clicked(self) -> None:
        """A mouse button went down somewhere. On the event loop.

        Only ever cancels a recording a trigger word started. A hotkey
        recording is one the user is holding a key for, and clicking while
        dictating (to put the cursor somewhere, to bring a window forward) is
        a thing people do on purpose.
        """
        self._cancel_wake_recording("click")

    def _cancel_wake_recording(self, why: str, detail: str = "") -> None:
        """Stop a recording nobody asked for, and remember the short reason.

        The reason is kept rather than logged: the capture loop prints exactly
        one line about a cancelled recording, and two lines saying the same
        thing in different words is what this used to be.
        """
        session = self._wake_session
        if session is None:
            return
        self._wake_session = None
        self._cancel_reason = why
        log.debug("recording cancelled: %s", detail or why)
        session.request_stop(CANCELLED)

    async def _watch_frontmost(self, session) -> None:
        """Cancel this recording if the user goes somewhere else.

        Polled rather than subscribed: the notification would be an
        NSWorkspace observer and a second event source to own, where this
        runs for the few seconds of one wake recording and stops. Reading it
        costs an osascript, so the interval is 500 ms and not less.

        A blank answer is "could not read it" (usually a missing Automation
        permission), never "no app is frontmost", so it is ignored: a
        permission Bol does not have must not cancel every recording.
        """
        try:
            was = await frontmost_bundle_id()
            while not session.stopped:
                await asyncio.sleep(FRONTMOST_POLL_S)
                if session.stopped:
                    return
                now = await frontmost_bundle_id()
                if not now or not was:
                    was = now or was
                    continue
                if now != was:
                    self._cancel_wake_recording(
                        "you switched apps", f"{was} gave way to {now}"
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a watcher must not end a recording
            log.debug("could not watch the frontmost app: %s", exc)

    def _touch_awake(self) -> None:
        """Start (or extend) the window in which no wake phrase is needed."""
        if self.wake is None:
            return
        self._awake_until = self._clock() + float(self.cfg.wake.awake_s)

    def _awake(self) -> bool:
        return self.wake is not None and self._clock() < self._awake_until

    def _idle_pill(self) -> None:
        """Take the pill off the screen.

        There is no "awake" pill any more. It stood for the window in which
        speech needs no trigger word, and a capsule that sits there between
        sentences is one nobody stops seeing; with awake_s off by default it
        would also have been lying most of the time.
        """
        self.hud.set("idle")

    def _mute_wake(self) -> None:
        if self.wake is not None:
            self.wake.mute()

    def _unmute_wake(self) -> None:
        # Never while paused. Both callers are "whatever I was doing is over",
        # and a pause set during a recording or a reply would otherwise be
        # undone by the finally clause of the thing it interrupted.
        if self.wake is not None and not self._asleep:
            self.wake.unmute()

    # ---------------------------------------------------------------- listening

    def _prewarm(self) -> None:
        """Warm the KV cache for the next LLM call while the user speaks:
        api mode cleans the transcript first; local mode's next call is the
        persona summary. One-way mode has no engine and nothing to warm."""
        if self.engine is None or self.summarizer is None:
            return
        if self.cfg.llm.provider == "api":
            self.engine.prewarm(CLEANUP_SYSTEM)
            return
        system = getattr(self.summarizer, "system_prompt", None)
        if system:
            self.engine.prewarm(system)

    def _hotkey_pressed(self) -> None:
        # First line on purpose: the pill has to appear on the keystroke, not
        # after the mic, the prewarm, or anything else this press starts.
        self.hud.set("listening", "Listening")
        if self._asleep:
            # The key is the only way back from a pause, and it says so on the
            # pill when the pause starts. Both halves come back together: the
            # loop, and the ear that was muted with it.
            self._asleep = False
            self._unmute_wake()
            print("bol: listening again.")
        # Holding the key opens the awake window too: having reached for it
        # once, the user should not have to reach for it again to say the
        # next thing.
        self._touch_awake()
        self._prewarm()
        # The session token is minted HERE, synchronously, so a release that
        # lands before recording starts still stops exactly this session.
        session = self.recorder.begin()
        self._ptt_session = session
        self._hush()  # barge-in over TTS
        # Barge-in over a recording that opened itself: it yields to the key,
        # and the press waits for the mic instead of being dropped.
        if self._active_session is not None and self._active_auto:
            self._active_session.request_stop()
        listen = asyncio.get_running_loop().create_task(
            self._listen_session(session, until_silence=False)
        )
        listen.add_done_callback(_drain)

    def _hotkey_released(self) -> None:
        """The key came up, so the recording it started ends now.

        Stops only its own session, and is inert if that press never got the
        mic. The words go on to be pasted, never sent: a release is "I have
        finished saying it", not "send it".
        """
        if self._ptt_session is not None:
            self._ptt_session.request_stop("release")
            self._ptt_session = None

    def _cue(self, name: str) -> None:
        """Play a cue without waiting for it.

        play_cue awaits afplay, so awaiting it here put the whole chime in
        front of the microphone (or in front of the paste). Cues are
        decoration; they never gate the thing the user asked for.
        """
        if not self.cfg.ui.sounds:
            return
        task = asyncio.get_running_loop().create_task(play_cue(name))
        task.add_done_callback(_drain)

    async def _follow_up_listen(self, question: str = "") -> None:
        """Open the microphone once, after Bol has spoken. Two-way only.

        Bol just read a summary or asked a question, so the next thing said is
        the answer and should not need a trigger word. Exactly one window of
        speak_window_ms, and then the microphone closes whatever happened:
        the rolling awake window is what had the pill coming back again and
        again while nobody was talking.
        """
        if not self.talk_back or self.text_mode or self.transcriber is None:
            return
        if self._asleep or self._listen_lock.locked() or self._pending_listen:
            return
        self.hud.set("listening", "Listening")
        session = self._begin(WAKE)
        await self._listen_session(
            session, until_silence=True, trigger=WAKE, once=True
        )
        # Nobody answered, and the question is still waiting: put it back on
        # screen rather than leaving an unanswered prompt with a blank pill.
        if question and self._permission_session:
            self.hud.set("permission", question)

    async def _listen_session(
        self, session, until_silence: bool, trigger: str = "", once: bool = False
    ) -> None:
        """Own the mic for one recording, then keep it open while the awake
        window lasts (the reopen loop: chaining must happen here, not via a
        nested call that would deadlock on our own lock).

        trigger is the kind of trigger word that started this, or "" for the
        hotkey, and it rides the whole loop rather than the first recording
        alone. A "type" that opened the awake window means the next sentence
        is dictation too, and having to say the trigger word again after every
        pause is exactly what the awake window exists to spare people.
        """
        if self._pending_listen:
            return  # one queued press is enough; drop extras
        self._pending_listen = True
        try:
            async with self._listen_lock:
                self._pending_listen = False
                while True:
                    outcome = await self._capture_and_handle(
                        session, until_silence, trigger
                    )
                    if once or not self._reopens(outcome):
                        break
                    session = self._begin(trigger)
                    until_silence = True
        finally:
            self._pending_listen = False

    def _reopens(self, outcome: str) -> bool:
        """Whether the mic goes straight back up after this recording.

        One reason, and it is the awake window, which is off by default: with
        it open, "hey Bol" (or a hold, or a dictation) is said once and the
        pauses in the minute that follows are pauses, not the end of the
        conversation. Off, every listen is one listen, and room noise has
        nothing to reopen. Never after a STOP either way, so a turn handed to
        the agent stays handed over and a dead microphone is not retried in a
        tight loop.
        """
        if self._asleep or self.transcriber is None or outcome == STOP:
            return False
        return self._awake()

    def _start_live(self, session) -> _LiveWords | None:
        """Put words in the pill while this recording runs, if we can.

        Returns the handle to close when the recording ends, or None when
        there is nothing to stream: text mode, [stt] engine = "none", live
        turned off, or a transcriber with no streaming decoder.
        """
        if not self.cfg.stt.live or self.text_mode:
            return None
        stream = getattr(self.transcriber, "stream", None)
        if stream is None:
            return None
        loop = asyncio.get_running_loop()
        live = _LiveWords(self.hud, loop, 1.0 / LIVE_PILL_HZ)
        # A thread-safe queue, not an asyncio one: the audio callback fills it
        # from PortAudio's thread and the decoder drains it on the MLX thread.
        live.blocks = queue.Queue()
        session.tap = live.blocks
        loop.create_task(self._run_live(stream, live)).add_done_callback(_drain)
        return live

    async def _run_live(self, stream, live: _LiveWords) -> None:
        blocks = live.blocks
        if blocks is None:
            return
        try:
            await stream(
                blocks,
                live.emit,
                context_size=self.cfg.stt.stream_context,
                chunk_ms=self.cfg.stt.stream_chunk_ms,
                sample_rate=self.cfg.audio.sample_rate,
            )
        except Exception as exc:
            # Debug, once per recording, and that is all: these words are a
            # courtesy, and the decode that decides what Claude gets still
            # runs on the finished recording either way.
            log.debug("live words failed: %s", exc)

    async def _capture_and_handle(
        self, session, until_silence: bool, trigger: str = ""
    ) -> str:
        typing = trigger == TYPE
        self._active_session = session
        self._active_auto = until_silence
        # The wake listener and this recording are the same microphone. Bol
        # is already listening properly, so the keyword model has nothing to
        # add and every chance to hear the dictation as a wake.
        self._mute_wake()
        watcher = None
        if trigger:
            # Only for a recording that started itself: this is the one the
            # user may not have meant, so it is the one that watches for them
            # doing something else instead.
            self._wake_session = session
            watcher = asyncio.get_running_loop().create_task(
                self._watch_frontmost(session)
            )
        try:
            live = self._start_live(session)
            try:
                try:
                    # Fired, not awaited, and before record() so the chime and
                    # the mic open together, not one after the other.
                    self._cue("listen")
                    audio = await self.recorder.record(
                        session, until_silence=until_silence
                    )
                finally:
                    # However the recording ended, the live decoder stops
                    # here, before anything else claims the pill: nothing
                    # streamed may land on it after this line.
                    if live is not None:
                        live.close()
            except Exception as exc:
                # A dead input device (headphones unplugged, another app
                # grabbing the mic) used to surface only as an unretrieved
                # task exception, leaving the hotkey silently dead forever.
                log.warning("capture failed: %s", exc)
                await self._speak(
                    "Lost the microphone. Check your input device.", state="error"
                )
                return STOP
            if session.end_reason == CANCELLED:
                # A click, or the user is in another app now. Whatever was
                # said belongs to what they went to do, so it is not
                # transcribed, not pasted, and the window that would have
                # reopened the microphone is shut with it.
                self._awake_until = 0.0
                self.hud.set("idle")
                why, self._cancel_reason = self._cancel_reason, ""
                print(f"bol: cancelled ({why})." if why else "bol: cancelled.")
                return STOP
            if audio is None:
                log.debug("no speech captured")
                self._idle_pill()
                return QUIET
            # The mic is closed and the words are on their way to the model:
            # say so, because this is the part with a wait in it.
            self.hud.set("finalizing", "Finalizing")
            assert self.transcriber is not None
            try:
                text = await self.transcriber.transcribe(audio, self.cfg.audio.sample_rate)
            except Exception as exc:
                log.warning("transcription failed: %s", exc)
                await self._speak("Couldn't transcribe that one. Try again.", state="error")
                return STOP
            if self.cfg.wake.enabled:
                # The trigger word started the recording; it is not part of
                # what was dictated, and the grammar must never see it. Only
                # a dictation strips "type", because in a conversation
                # "type hello world" is the grammar's own command.
                text = strip_wake_phrase(
                    text, self._type_phrases if typing else self.cfg.wake.phrases
                )
            if not text:
                self._idle_pill()
                return QUIET
            print(f"you: {text}")
            # Words heard is the evidence the conversation is still going, so
            # the awake window is measured from the last thing said, not from
            # the wake (or the hold) that opened it.
            self._touch_awake()
            handled = await self._handle_utterance(text)
            return CHAIN if handled else STOP
        finally:
            if watcher is not None:
                watcher.cancel()
            if self._wake_session is session:
                self._wake_session = None
            self._active_session = None
            # This recording is over however it ended, so no stale session
            # can swallow the next press.
            if self._ptt_session is session:
                self._ptt_session = None
            # However this ended, the keyword model gets its ear back, after
            # the same tail that follows Bol speaking.
            self._unmute_wake()

    # ---------------------------------------------------------------- actions

    _YES = {"yes", "yeah", "yep", "approve", "go ahead", "do it"}
    _NO = {"no", "nope", "deny", "don't", "dont"}

    # Every keystroke Bol sends is either the user's own words or Bol's doing,
    # and the bridge treats the two differently: an explicit Enter goes
    # wherever the cursor is, anything else only into a Claude window. These
    # two wrappers are where the daemon says which.
    async def _inject(
        self, text: str, submit: bool = False, explicit: bool = False
    ) -> None:
        await self.bridge.inject(text, submit, explicit=explicit)

    async def _keys(self, *keys: str, explicit: bool = False) -> None:
        await self.bridge.inject_keys(*keys, explicit=explicit)

    async def _handle_utterance(self, text: str) -> bool:
        """Act on one utterance. Returns True if the mic may reopen (the awake
        window decides), False if the turn passed to Claude or the loop should
        go quiet.
        """
        try:
            if self._permission_session is not None:
                norm = text.strip().lower().rstrip(".!,")
                if norm in self._YES:
                    return await self._answer_permission(approve=True)
                if norm in self._NO:
                    return await self._answer_permission(approve=False)
            return await self._apply(self.grammar.parse(text))
        except SubmitBlocked as exc:
            # The text WAS typed; only the Enter was withheld. Which is
            # exactly a pending paste, and saying "send it" once the agent is
            # in front is now the way to finish it.
            log.debug("submit withheld: %s", exc)
            self._pending_paste = True
            await self._speak(
                f"Typed it, but that window doesn't look like {self.agent_name}, "
                "so I didn't press Enter."
            )
            return True
        except BridgeError as exc:
            await self._speak(
                f"Couldn't reach {self.agent_name}: {exc}", state="error"
            )
            return True

    async def _answer_permission(self, approve: bool) -> bool:
        """Answer the permission prompt Bol actually announced.

        Enter goes to whichever terminal is frontmost, so approving a prompt
        raised by a session Bol isn't narrating would approve the wrong thing.
        """
        session = self._permission_session or ""
        self._permission_session = None
        if not self._follows(session):
            await self._speak(
                f"That prompt came from another {self.agent_name} session, "
                "so I left it alone."
            )
            return True
        if approve:
            # "Go ahead" is the user answering a question they were just read,
            # which is as explicit as an Enter gets.
            await self._keys("Enter", explicit=True)
            self._pending_paste = False
            self.hud.set("sending", "Approved")
            print("bol: approved.")
            return False
        await self._keys("Escape", explicit=True)
        self._pending_paste = False
        self.hud.set("sending", "Denied")
        print("bol: denied.")
        return True

    async def _apply(self, parsed) -> bool:
        action, text = parsed.action, parsed.text
        mode = self.cfg.cleanup.mode
        # "always" means every dictation there is, whichever way it started:
        # the "type" flow, a hold, an awake follow-up, and the spoken
        # "type ..." command too. Raw dictation is not what anybody wants to
        # hand a coding agent, and a mode called always that skipped one of
        # the four was the bug people were actually reporting.
        wants_clean = (parsed.clean and mode != "off") or (
            mode == "always"
            and action in (Action.DICTATE, Action.TYPE, Action.SEND)
        )
        if wants_clean and text:
            cleaned = await clean_transcript(
                self.engine,
                text,
                self.cfg.cleanup.deadline_s,
                # One-way mode has no engine, so the tuned local cleaner is
                # the whole cleanup pass there, api provider or not.
                use_llm=self.engine is not None and self.cfg.llm.provider == "api",
                cleaner=self.cleaner,
                vocabulary=self.cfg.vocabulary.words,
            )
            if cleaned != text:
                print(f"bol: cleaned -> {cleaned}")
                text = cleaned
        if action in (Action.DICTATE, Action.TYPE):
            # A bare "clean it up" parses as DICTATE with no text; injecting
            # a lone space would litter Claude's input box.
            if not text:
                return True
            await self._inject(
                text + " " if action is Action.DICTATE else text, submit=False
            )
            # Bol put text in the box and did not submit it, which is the
            # whole precondition for a "send it" that presses Enter on it.
            # The pill goes away here rather than saying so: the words are on
            # screen, in the box, where the user is already looking, and a
            # blue capsule repeating that after every single paste is what
            # people asked us to take away.
            self._pending_paste = True
            self._idle_pill()
            print("bol: pasted.")
            return True
        if action is Action.SEND:
            self._permission_session = None
            # A SEND is the user saying "send it", so this Enter goes
            # wherever they are looking, Notes and Slack included.
            await self._inject(text, submit=True, explicit=True)
            self._pending_paste = False
            # Same as a spoken "send it": the pill stays off and the chime is
            # the receipt.
            self._idle_pill()
            self._cue("done")
            print("bol: sent.")
            return False
        if action is Action.DISCARD:
            # C-u wipes the agent's input line. The user said so, so it is
            # allowed outside a terminal too, where the bridge turns it into
            # the one Cmd+Z that undoes the paste.
            await self._keys("C-u", explicit=True)
            self._pending_paste = False
            self._idle_pill()
            self._cue("discard")
            return True
        if action is Action.INTERRUPT:
            await self.bridge.interrupt()
            self._pending_paste = False
            await self._speak("Interrupted.")
            return True
        if action is Action.SLEEP:
            self._go_to_sleep()
            return False
        if action is Action.REPEAT:
            if self._last_reply:
                await self._speak(self._last_reply)
            return True
        return False

    async def _speak(self, text: str, state: str = "speaking", pill: str = "") -> None:
        # Serialized so two hook handlers can't cut each other off; barge-in
        # still works because the hotkey calls speaker.stop() directly.
        async with self._speak_lock:
            self._last_reply = text
            print(f"bol: {text}")
            self.hud.set(state, pill or text)
            if self.speaker is None:
                # One-way: the terminal line and the pill are the whole
                # reply. An error or a permission question stays up for the
                # same reasons it does below; everything else clears.
                if state == "speaking":
                    self._idle_pill()
                return
            # Bol's own voice is the loudest thing this microphone will hear
            # all day, and "hey Bol" is a phrase Bol says. Deaf while
            # speaking, and for the tail after it.
            self._mute_wake()
            try:
                await self.speaker.speak(text)
            finally:
                self._unmute_wake()
                # An error and a permission question stay on screen: one is a
                # remedy to read, the other a question still waiting for its
                # answer. Both leave on their own, or on the next state.
                if state == "speaking":
                    self._idle_pill()

    # ---------------------------------------------------------------- sessions

    def _follows(self, session_id: str, cwd: str = "") -> bool:
        """Whether this hook event belongs to the session Bol is narrating.

        Agent hooks are user-scoped, so a second `claude` (or a `codex`) in
        another terminal posts here too. Left unfiltered, each session's Stop
        cuts the previous summary off mid-sentence and reopens the mic, and a
        "yes" meant for one prompt approves whichever terminal is frontmost.
        Bol latches onto the first session it hears from, whichever agent that
        is; [server] follow = "all" opts back into narrating every session.
        """
        if self.cfg.server.follow == "all" or not session_id:
            return True
        if self._bound_session is None:
            self._bound_session = session_id
            self._bound_cwd = cwd
            print(f"bol: narrating {self._session_label()}.")
            return True
        if session_id == self._bound_session:
            if cwd and not self._bound_cwd:
                self._bound_cwd = cwd
            return True
        if not self._warned_other_session:
            self._warned_other_session = True
            print(
                "bol: another agent session is running; Bol is only "
                f"narrating {self._session_label()}."
            )
        log.debug("ignoring hook event from session %s", session_id)
        return False

    def _shows(self, session_id: str) -> bool:
        """Whether the pill may show this event. Read-only on purpose.

        _follows() latches Bol onto a session, and that decision belongs to
        the Stop handler. A tool event arriving first must never be what
        binds Bol to a session it was never asked to narrate.
        """
        if self.cfg.server.follow == "all" or not session_id:
            return True
        return self._bound_session in (None, session_id)

    def _session_label(self) -> str:
        if self._bound_cwd:
            return Path(self._bound_cwd).name or self._bound_cwd
        return (self._bound_session or "")[:8] or "this session"

    # ---------------------------------------------------------------- hooks

    # In one-way mode none of these are registered, because there is no hook
    # server to register them on. They check anyway: "one-way says nothing
    # about a coding agent" is a promise about the daemon, not about one
    # call site, and a promise that only holds when nobody calls the method
    # directly is not one.

    async def _on_tool(self, payload: dict) -> None:
        if not self.talk_back:
            return
        # Recorded for every session: the tracker is bounded, and the Stop
        # handler is where the session filter decides who gets narrated.
        # A Codex PostToolUse carries no agent marker of its own, so the
        # agent already being narrated is what it is attributed to.
        tool = self.tracker.record_tool(payload, self._agent)
        if self._shows(payload.get("session_id", "")):
            self.hud.set("thinking", "Thinking", tool_line(tool.tool_name, tool.detail))

    async def _on_stop(self, payload: dict) -> None:
        if not self.talk_back:
            return
        # Always finish the turn, even for a session we ignore, so its tool
        # log is drained rather than left behind.
        event = self.tracker.finish_turn(payload, self._agent)
        if not self._follows(event.session_id, event.cwd):
            return
        self._bind_agent(event.agent)
        if self.summarizer is None:
            self._idle_pill()
            return
        reply = await self.summarizer.summarize(event)
        await self._speak(reply)
        # Bol just said something, so the next thing the user says is the
        # reply to it. One window, then the microphone closes again.
        await self._follow_up_listen()

    async def _on_notification(self, payload: dict) -> None:
        if not self.talk_back:
            return
        note = self.tracker.notification(payload, self._agent)
        if not self._follows(note.session_id):
            return
        self._bind_agent(note.agent)
        if note.notification_type == "permission_prompt":
            await self._ask_permission(
                note.session_id, note.message or f"{self.agent_name} needs your permission."
            )
        elif note.notification_type in {"idle_prompt", "agent_needs_input"}:
            await self._speak(note.message or f"{self.agent_name}'s waiting on you.")

    async def _on_permission_request(self, payload: dict) -> None:
        """Codex is about to ask for approval. Same flow as Claude's prompt.

        Codex has no Notification event, so this is the only place a Codex
        prompt can be read aloud, and the only place "go ahead" gets armed.
        """
        if not self.talk_back:
            return
        note = self.tracker.permission_request(payload)
        if not self._follows(note.session_id, payload.get("cwd", "")):
            return
        self._bind_agent(note.agent)
        await self._ask_permission(note.session_id, note.message)

    async def _ask_permission(self, session_id: str, message: str) -> None:
        self._permission_session = session_id
        await self._speak(
            f"{message} Say 'go ahead' or 'no'.", state="permission", pill=message
        )
        # Bol asked a question out loud, so the microphone opens for the
        # answer. Once, for speak_window_ms, and the question goes back on
        # the pill if nobody says anything.
        await self._follow_up_listen(question=message)

    # ---------------------------------------------------------------- text mode

    async def _text_console(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                if line in {"quit", "exit"}:
                    break
                await self._handle_utterance(line)
        finally:
            if self.server is not None:
                await self.server.stop()
            if self.engine is not None:
                await self.engine.stop()
