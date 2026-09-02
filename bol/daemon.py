"""Bol daemon: the loop.

hotkey/auto-listen → record → transcribe → parse command → tmux inject
        ↑                                                        ↓
   TTS "what next?"  ← persona summary ← Stop hook ← Claude runs turn

Dictated text is injected into Claude's input box immediately (no Enter), so
the user watches it appear in the TUI exactly like typing. "send it" presses
Enter. The Stop hook closes the loop by speaking what happened.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from .audio import Recorder
from .bridge import BridgeError, build_bridge
from .bridge.focused import SubmitBlocked
from .cleanup import CLEANUP_SYSTEM, build_cleaner, clean_transcript
from .config import Config, hook_token
from .grammar import Action, Grammar
from .hooks import HookServer, TurnTracker
from .hotkey import HotkeyListener
from .llm import LLMEngine
from .speak import build_speaker, play_cue
from .stt import build_transcriber
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


class Daemon:
    def __init__(self, cfg: Config, text_mode: bool = False) -> None:
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

        self._listen_lock = asyncio.Lock()
        # Hook handlers run as independent tasks; without this a Stop and a
        # Notification landing together cut each other off mid-sentence.
        self._speak_lock = asyncio.Lock()
        self._pending_listen = False
        self._active_session = None
        self._active_hands_free = False
        self._ptt_session = None
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

        self.hotkey = HotkeyListener(
            self.cfg.hotkey, self._hotkey_pressed, self._hotkey_released
        )
        self.hotkey.start()
        mode = self.cfg.hotkey.mode.replace("_", "-")
        print(f"bol: hold {self.cfg.hotkey.key} to talk ({mode}). Ctrl+C to quit.")
        try:
            await asyncio.Event().wait()
        finally:
            self.hotkey.stop()
            await self.server.stop()
            await self.engine.stop()

    # ---------------------------------------------------------------- listening

    def _hotkey_pressed(self) -> None:
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

    def _hotkey_released(self) -> None:
        # Stops only its own session, inert if that press never got the mic.
        if self._ptt_session is not None:
            self._ptt_session.request_stop()
            self._ptt_session = None

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

    async def _capture_and_handle(self, session, until_silence: bool) -> bool:
        self._active_session = session
        self._active_hands_free = until_silence
        try:
            try:
                if self.cfg.sound_cues:
                    await play_cue("listen")
                audio = await self.recorder.record(session, until_silence=until_silence)
            except Exception as exc:
                # A dead input device (headphones unplugged, another app
                # grabbing the mic) used to surface only as an unretrieved
                # task exception, leaving the hotkey silently dead forever.
                log.warning("capture failed: %s", exc)
                await self._speak("Lost the microphone. Check your input device.")
                return False
            if audio is None:
                log.debug("no speech captured")
                return False
            assert self.transcriber is not None
            try:
                text = await self.transcriber.transcribe(audio, self.cfg.audio.sample_rate)
            except Exception as exc:
                log.warning("transcription failed: %s", exc)
                await self._speak("Couldn't transcribe that one. Try again.")
                return False
            if not text:
                return False
            print(f"you: {text}")
            return await self._handle_utterance(text)
        finally:
            self._active_session = None

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
            await self._speak(f"Couldn't reach Claude: {exc}")
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
            print("bol: approved.")
            return False
        await self.bridge.inject_keys("Escape")
        print("bol: denied.")
        return True

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
        if action in (Action.DICTATE, Action.TYPE):
            # A bare "clean it up" parses as DICTATE with no text; injecting
            # a lone space would litter Claude's input box.
            if not text:
                return True
            await self.bridge.inject(
                text + " " if action is Action.DICTATE else text, submit=False
            )
            return True
        if action is Action.SEND:
            self._permission_session = None
            await self.bridge.inject(text, submit=True)
            if self.cfg.sound_cues:
                await play_cue("done")
            print("bol: sent. Claude's turn.")
            return False
        if action is Action.DISCARD:
            # C-u wipes Claude Code's input line.
            await self.bridge.inject_keys("C-u")
            if self.cfg.sound_cues:
                await play_cue("discard")
            return True
        if action is Action.INTERRUPT:
            await self.bridge.interrupt()
            await self._speak("Interrupted.")
            return True
        if action is Action.SLEEP:
            self._asleep = True
            print("bol: sleeping. Press the hotkey when you need me.")
            return False
        if action is Action.REPEAT:
            if self._last_reply:
                await self._speak(self._last_reply)
            return True
        return False

    async def _speak(self, text: str) -> None:
        # Serialized so two hook handlers can't cut each other off; barge-in
        # still works because the hotkey calls speaker.stop() directly.
        async with self._speak_lock:
            self._last_reply = text
            print(f"bol: {text}")
            await self.speaker.speak(text)

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

    def _session_label(self) -> str:
        if self._bound_cwd:
            return Path(self._bound_cwd).name or self._bound_cwd
        return (self._bound_session or "")[:8] or "this session"

    # ---------------------------------------------------------------- hooks

    async def _on_tool(self, payload: dict) -> None:
        # Recorded for every session: the tracker is bounded, and the Stop
        # handler is where the session filter decides who gets narrated.
        self.tracker.record_tool(payload)

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
            await self._speak(f"{msg} Say 'go ahead' or 'no'.")
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
