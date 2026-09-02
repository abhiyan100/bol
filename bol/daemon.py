"""Bol daemon: the loop.

hotkey/auto-listen → record → transcribe → parse command → tmux inject
        ↑                                                        ↓
   TTS "what next?"  ← persona summary ← Stop hook ← Claude runs turn

Dictated text is injected into Claude's input box immediately, so the user
watches it appear in the TUI exactly like typing. With [hotkey] submit =
"auto" a dictated instruction of a few words or more presses Enter for itself;
shorter text, a "type ..." prefix, and submit = "voice" all paste without it,
and "send it" always submits. The Stop hook closes the loop by speaking what
happened.

While the user is still talking the pill shows the words as they are decoded.
That path is display only and stops at the pill: the text that reaches Claude
is always the one full-buffer decode taken after the recording ends.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
from pathlib import Path

from .audio import Recorder
from .bridge import BridgeError, build_bridge
from .bridge.focused import SubmitBlocked
from .cleanup import CLEANUP_SYSTEM, build_cleaner, clean_transcript
from .config import Config, hook_token, validate_config
from .grammar import Action, Grammar
from .hooks import HookServer, TurnTracker
from .hotkey import HotkeyListener
from .hud import Hud, tool_line
from .llm import LLMEngine
from .speak import build_speaker, play_cue
from .stt import STREAM_END, build_transcriber
from .summarize import build_summarizer

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


# How often the pill may be redrawn with streamed words. Past about four a
# second a line of text reads as flicker, and every redraw is a write down a
# pipe to another process.
LIVE_PILL_HZ = 4.0


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
    def __init__(self, cfg: Config, text_mode: bool = False) -> None:
        # Fail here, with the list of valid values, rather than arming a
        # hotkey whose mode nothing in the listener recognises.
        validate_config(cfg)
        self.cfg = cfg
        self.text_mode = text_mode
        self.bridge = build_bridge(cfg)
        self.grammar = Grammar(cfg.commands)
        self.tracker = TurnTracker()
        self.server = HookServer(
            cfg.server.host,
            cfg.server.port,
            hook_token(),
            allow_remote=cfg.server.allow_remote,
        )
        self.speaker = build_speaker(cfg)
        self.engine = LLMEngine(cfg)
        self.cleaner = build_cleaner(cfg)
        self.summarizer = build_summarizer(cfg, self.engine)
        self.recorder = Recorder(cfg.audio)
        self.transcriber = None if text_mode else build_transcriber(cfg)
        self.hotkey: HotkeyListener | None = None
        # The on-screen pill. Inert until start(), and every call on it is a
        # no-op when the child is missing.
        self.hud = Hud(enabled=cfg.ui.pill, position=cfg.ui.position)

        self._listen_lock = asyncio.Lock()
        # Hook handlers run as independent tasks; without this a Stop and a
        # Notification landing together cut each other off mid-sentence.
        self._speak_lock = asyncio.Lock()
        self._pending_listen = False
        self._active_session = None
        self._active_hands_free = False
        self._ptt_session = None
        # The recording a tap started and left running; the next tap ends it.
        self._tap_session = None
        self._asleep = False
        self._last_reply = ""
        # Claude Code hooks are user-scoped, so every session on this machine
        # posts here. Bol latches onto one (see [server] follow).
        self._bound_session: str | None = None
        self._bound_cwd = ""
        self._warned_other_session = False
        # Which session raised the permission prompt we're waiting on, so a
        # spoken "yes" can never approve a different session's prompt.
        self._permission_session: str | None = None

    # ---------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        target = await self.bridge.attach()
        print(f"bol: injecting into {target}")

        self.server.on("Stop", self._on_stop)
        self.server.on("PostToolUse", self._on_tool)
        self.server.on("Notification", self._on_notification)
        await self.server.start()
        loop = asyncio.get_running_loop()
        # LLM warms in the background; template/raw fallbacks cover the gap
        # (and the first run's model download).
        loop.create_task(self.engine.start()).add_done_callback(_drain)
        # Same for the cleanup model: cold, the first "clean it up" spends its
        # whole deadline loading weights and hands back the text unchanged.
        warmup = getattr(self.cleaner, "warmup", None)
        if callable(warmup):
            loop.create_task(warmup()).add_done_callback(_drain)
        print(f"bol: hook server on http://{self.cfg.server.host}:{self.cfg.server.port}/hook")

        if self.transcriber is not None:
            print("bol: warming up speech model ...")
            await self.transcriber.warmup()

        if self.text_mode or self.transcriber is None:
            print("bol: text mode. Type what you'd say ('send it', 'type ...', 'close').")
            await self._text_console()
            return

        # Build the input stream before the key is armed: construction is the
        # expensive half of opening a mic and it must not land on the press.
        try:
            await self.recorder.open()
        except Exception as exc:
            log.warning("could not prepare the microphone: %s", exc)

        # Started before the key is armed so the very first press has a pill
        # to appear on.
        await self.hud.start()

        self.hotkey = HotkeyListener(
            self.cfg.hotkey, self._hotkey_pressed, self._hotkey_released
        )
        self.hotkey.start()
        key = self.cfg.hotkey.key
        if self.cfg.hotkey.mode == "auto":
            print(f"bol: tap or hold {key} to talk. Ctrl+C to quit.")
        else:
            mode = self.cfg.hotkey.mode.replace("_", "-")
            print(f"bol: hold {key} to talk ({mode}). Ctrl+C to quit.")
        try:
            await asyncio.Event().wait()
        finally:
            self.hotkey.stop()
            await self.hud.stop()
            await self.recorder.close()
            await self.server.stop()
            await self.engine.stop()

    # ---------------------------------------------------------------- listening

    def _hotkey_pressed(self) -> None:
        # First line on purpose: the pill has to appear on the keystroke, not
        # after the mic, the prewarm, or anything else this press starts.
        self.hud.set("listening", "Listening")
        self._asleep = False
        # Warm the KV cache for the next LLM call while the user speaks:
        # api mode cleans the transcript first; local mode's next call is
        # the persona summary.
        if self.cfg.llm.provider == "api":
            self.engine.prewarm(CLEANUP_SYSTEM)
        else:
            system = getattr(self.summarizer, "system_prompt", None)
            if system:
                self.engine.prewarm(system)
        # The session token is minted HERE, synchronously, so a release that
        # lands before recording starts still stops exactly this session.
        session = self.recorder.begin()
        self._ptt_session = session
        loop = asyncio.get_running_loop()
        loop.create_task(self.speaker.stop()).add_done_callback(_drain)  # barge-in over TTS
        # Barge-in over a hands-free recording: it yields to the hotkey, and
        # the press waits for the mic instead of being dropped.
        if self._active_session is not None and self._active_hands_free:
            self._active_session.request_stop()
        listen = loop.create_task(self._listen_session(session, until_silence=False))
        listen.add_done_callback(_drain)

    def _hotkey_released(self, kind: str = "hold") -> None:
        """kind is "hold" (end the recording now) or "tap" (auto mode)."""
        if kind == "tap":
            self._tap_released()
            return
        # Stops only its own session, inert if that press never got the mic.
        if self._ptt_session is not None:
            self._ptt_session.request_stop()
            self._ptt_session = None
        self._tap_session = None
        self._clear_tap()

    def _tap_released(self) -> None:
        if self._tap_session is not None:
            # Second tap: the user is done early, before the silence gate.
            self._tap_session.request_stop()
            self._tap_session = None
            self._ptt_session = None
            self._clear_tap()
            return
        session, self._ptt_session = self._ptt_session, None
        if session is None:
            # The recording this press started is already over (it hit the
            # cap, or the mic failed). Re-arm the key rather than leaving the
            # listener convinced a recording is still running.
            self._clear_tap()
            return
        # Too short to be a hold, so the user is still talking: hand the
        # ending over to the energy gate instead of to the key.
        session.until_silence = True
        self._tap_session = session

    def _clear_tap(self) -> None:
        if self.hotkey is not None:
            self.hotkey.clear_tap()

    def _cue(self, name: str) -> None:
        """Play a cue without waiting for it.

        play_cue awaits afplay, so awaiting it here put the whole chime in
        front of the microphone (or in front of the paste). Cues are
        decoration; they never gate the thing the user asked for.
        """
        if not self.cfg.sound_cues:
            return
        task = asyncio.get_running_loop().create_task(play_cue(name))
        task.add_done_callback(_drain)

    async def _listen_session(self, session, until_silence: bool) -> None:
        """Own the mic for one recording, then keep it open across hands-free
        follow-ups (the reopen loop: chaining must happen here, not via a
        nested call that would deadlock on our own lock)."""
        if self._pending_listen:
            return  # one queued press is enough; drop extras
        self._pending_listen = True
        try:
            async with self._listen_lock:
                self._pending_listen = False
                while True:
                    reopen = await self._capture_and_handle(session, until_silence)
                    if (
                        not reopen
                        or not self.cfg.hands_free
                        or self._asleep
                        or self.transcriber is None
                    ):
                        break
                    session = self.recorder.begin()
                    until_silence = True
        finally:
            self._pending_listen = False

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

    async def _capture_and_handle(self, session, until_silence: bool) -> bool:
        self._active_session = session
        self._active_hands_free = until_silence
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
                return False
            if audio is None:
                log.debug("no speech captured")
                self.hud.set("idle")
                return False
            # The mic is closed and the words are on their way to the model:
            # say so, because this is the part with a wait in it.
            self.hud.set("finalizing", "Finalizing")
            assert self.transcriber is not None
            try:
                text = await self.transcriber.transcribe(audio, self.cfg.audio.sample_rate)
            except Exception as exc:
                log.warning("transcription failed: %s", exc)
                await self._speak("Couldn't transcribe that one. Try again.", state="error")
                return False
            if not text:
                self.hud.set("idle")
                return False
            print(f"you: {text}")
            return await self._handle_utterance(text)
        finally:
            self._active_session = None
            # This recording is over however it ended, so no stale tap state
            # can swallow the next press.
            if self._tap_session is session:
                self._tap_session = None
            if self._ptt_session is session:
                self._ptt_session = None
            self._clear_tap()

    async def _auto_listen(self) -> None:
        """Reopen the mic after Bol speaks (hook-driven). No-op if a listen
        is already running or queued."""
        if not self.cfg.hands_free or self._asleep or self.text_mode:
            return
        if self.transcriber is None:
            return
        if self._listen_lock.locked() or self._pending_listen:
            return
        await self._listen_session(self.recorder.begin(), until_silence=True)

    # ---------------------------------------------------------------- actions

    _YES = {"yes", "yeah", "yep", "approve", "go ahead", "do it"}
    _NO = {"no", "nope", "deny", "don't", "dont"}

    async def _handle_utterance(self, text: str) -> bool:
        """Act on one utterance. Returns True if the mic should reopen
        immediately (hands-free chaining), False if the turn passed to Claude
        or the loop should go quiet."""
        try:
            if self._permission_session is not None:
                norm = text.strip().lower().rstrip(".!,")
                if norm in self._YES:
                    return await self._answer_permission(approve=True)
                if norm in self._NO:
                    return await self._answer_permission(approve=False)
            return await self._apply(self.grammar.parse(text))
        except SubmitBlocked as exc:
            # The text WAS typed; only the Enter was withheld.
            log.info("submit withheld: %s", exc)
            await self._speak(
                "Typed it, but that window doesn't look like Claude, "
                "so I didn't press Enter."
            )
            return True
        except BridgeError as exc:
            await self._speak(f"Couldn't reach Claude: {exc}", state="error")
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
                "That prompt came from another Claude Code session, so I left it alone."
            )
            return True
        if approve:
            await self.bridge.inject_keys("Enter")
            self.hud.set("sending", "Approved")
            print("bol: approved.")
            return False
        await self.bridge.inject_keys("Escape")
        self.hud.set("sending", "Denied")
        print("bol: denied.")
        return True

    def _auto_sends(self, text: str) -> bool:
        """Whether plain dictation should press Enter for the user.

        The word floor is the guard: a stray noise or a one-word misfire gets
        pasted and can be deleted, where sending it would have cost a whole
        Claude turn. Anything ending in "send it" never reaches here, it is
        already Action.SEND.
        """
        if self.cfg.hotkey.submit != "auto" or not text:
            return False
        return len(text.split()) >= self.cfg.hotkey.auto_send_min_words

    async def _apply(self, parsed) -> bool:
        action, text = parsed.action, parsed.text
        mode = self.cfg.cleanup.mode
        wants_clean = (parsed.clean and mode != "off") or (
            mode == "always" and action in (Action.DICTATE, Action.SEND)
        )
        if wants_clean and text:
            cleaned = await clean_transcript(
                self.engine,
                text,
                self.cfg.cleanup.deadline_s,
                use_llm=self.cfg.llm.provider == "api",
                cleaner=self.cleaner,
            )
            if cleaned != text:
                print(f"bol: cleaned -> {cleaned}")
                text = cleaned
        if action is Action.DICTATE and self._auto_sends(text):
            # Plain speech is a whole instruction, so send it. A "type ..."
            # prefix (Action.TYPE) still pastes without Enter, and a trailing
            # "send it" is still stripped and honoured; only the silent
            # "say the magic words or nothing happens" default goes away.
            action = Action.SEND
        if action in (Action.DICTATE, Action.TYPE):
            # A bare "clean it up" parses as DICTATE with no text; injecting
            # a lone space would litter Claude's input box.
            if not text:
                return True
            await self.bridge.inject(
                text + " " if action is Action.DICTATE else text, submit=False
            )
            # The words are in Claude's box where the user can see them, so
            # the pill has nothing left to say.
            self.hud.set("idle")
            return True
        if action is Action.SEND:
            self._permission_session = None
            await self.bridge.inject(text, submit=True)
            self.hud.set("sending", "Sent")
            self._cue("done")
            print("bol: sent. Claude's turn.")
            return False
        if action is Action.DISCARD:
            # C-u wipes Claude Code's input line.
            await self.bridge.inject_keys("C-u")
            self.hud.set("idle")
            self._cue("discard")
            return True
        if action is Action.INTERRUPT:
            await self.bridge.interrupt()
            await self._speak("Interrupted.")
            return True
        if action is Action.SLEEP:
            self._asleep = True
            self.hud.set("idle")
            print("bol: sleeping. Press the hotkey when you need me.")
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
            try:
                await self.speaker.speak(text)
            finally:
                # An error and a permission question stay on screen: one is a
                # remedy to read, the other a question still waiting for its
                # answer. Both leave on their own, or on the next state.
                if state == "speaking":
                    self.hud.set("idle")

    # ---------------------------------------------------------------- sessions

    def _follows(self, session_id: str, cwd: str = "") -> bool:
        """Whether this hook event belongs to the session Bol is narrating.

        Claude Code hooks are user-scoped, so a second `claude` in another
        terminal posts here too. Left unfiltered, each session's Stop cuts the
        previous summary off mid-sentence and reopens the mic, and a "yes"
        meant for one prompt approves whichever terminal is frontmost. Bol
        latches onto the first session it hears from; [server] follow = "all"
        opts back into narrating every session.
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
                "bol: another Claude Code session is running; Bol is only "
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

    async def _on_tool(self, payload: dict) -> None:
        # Recorded for every session: the tracker is bounded, and the Stop
        # handler is where the session filter decides who gets narrated.
        tool = self.tracker.record_tool(payload)
        if self._shows(payload.get("session_id", "")):
            self.hud.set("thinking", "Thinking", tool_line(tool.tool_name, tool.detail))

    async def _on_stop(self, payload: dict) -> None:
        # Always finish the turn, even for a session we ignore, so its tool
        # log is drained rather than left behind.
        event = self.tracker.finish_turn(payload)
        if not self._follows(event.session_id, event.cwd):
            return
        reply = await self.summarizer.summarize(event)
        await self._speak(reply)
        await self._auto_listen()

    async def _on_notification(self, payload: dict) -> None:
        note = self.tracker.notification(payload)
        if not self._follows(note.session_id):
            return
        if note.notification_type == "permission_prompt":
            self._permission_session = note.session_id
            msg = note.message or "Claude needs your permission."
            await self._speak(
                f"{msg} Say 'go ahead' or 'no'.", state="permission", pill=msg
            )
            await self._auto_listen()
        elif note.notification_type in {"idle_prompt", "agent_needs_input"}:
            await self._speak(note.message or "Claude's waiting on you.")
            await self._auto_listen()

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
            await self.server.stop()
            await self.engine.stop()
