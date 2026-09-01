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

from .audio import Recorder
from .bridge import BridgeError, build_bridge
from .cleanup import CLEANUP_SYSTEM, build_cleaner, clean_transcript
from .config import Config, hook_token
from .grammar import Action, parse_transcript
from .hooks import HookServer, TurnTracker
from .hotkey import HotkeyListener
from .llm import LLMEngine
from .speak import build_speaker, play_cue
from .stt import build_transcriber
from .summarize import build_summarizer

log = logging.getLogger("bol")


class Daemon:
    def __init__(self, cfg: Config, text_mode: bool = False) -> None:
        self.cfg = cfg
        self.text_mode = text_mode
        self.bridge = build_bridge(cfg)
        self.tracker = TurnTracker()
        self.server = HookServer(cfg.server.host, cfg.server.port, hook_token())
        self.speaker = build_speaker(cfg)
        self.engine = LLMEngine(cfg)
        self.cleaner = build_cleaner(cfg)
        self.summarizer = build_summarizer(cfg, self.engine)
        self.recorder = Recorder(cfg.audio)
        self.transcriber = None if text_mode else build_transcriber(cfg)
        self.hotkey: HotkeyListener | None = None

        self._listen_lock = asyncio.Lock()
        self._pending_listen = False
        self._active_session = None
        self._active_hands_free = False
        self._ptt_session = None
        self._asleep = False
        self._last_reply = ""
        self._awaiting_permission = False

    # ---------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        target = await self.bridge.attach()
        print(f"bol: injecting into {target}")

        self.server.on("Stop", self._on_stop)
        self.server.on("PostToolUse", self._on_tool)
        self.server.on("Notification", self._on_notification)
        await self.server.start()
        # LLM warms in the background; template/raw fallbacks cover the gap
        # (and the first run's model download).
        engine_task = asyncio.get_running_loop().create_task(self.engine.start())
        engine_task.add_done_callback(lambda t: t.exception())
        print(f"bol: hook server on http://{self.cfg.server.host}:{self.cfg.server.port}/hook")

        if self.transcriber is not None:
            print("bol: warming up speech model…")
            await self.transcriber.warmup()

        if self.text_mode or self.transcriber is None:
            print("bol: text mode — type what you'd say ('send it', 'type …', 'close').")
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
        loop.create_task(self.speaker.stop())  # barge-in over TTS
        # Barge-in over a hands-free recording: it yields to the hotkey, and
        # the press waits for the mic instead of being dropped.
        if self._active_session is not None and self._active_hands_free:
            self._active_session.request_stop()
        loop.create_task(self._listen_session(session, until_silence=False))

    def _hotkey_released(self) -> None:
        # Stops only its own session — inert if that press never got the mic.
        if self._ptt_session is not None:
            self._ptt_session.request_stop()
            self._ptt_session = None

    async def _listen_session(self, session, until_silence: bool) -> None:
        """Own the mic for one recording, then keep it open across hands-free
        follow-ups (the reopen loop — chaining must happen here, not via a
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
            if self.cfg.sound_cues:
                await play_cue("listen")
            audio = await self.recorder.record(session, until_silence=until_silence)
            if audio is None:
                log.debug("no speech captured")
                return False
            assert self.transcriber is not None
            text = await self.transcriber.transcribe(audio, self.cfg.audio.sample_rate)
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
            if self._awaiting_permission:
                norm = text.strip().lower().rstrip(".!,")
                if norm in self._YES:
                    self._awaiting_permission = False
                    await self.bridge.inject_keys("Enter")
                    print("bol: approved.")
                    return False
                if norm in self._NO:
                    self._awaiting_permission = False
                    await self.bridge.inject_keys("Escape")
                    print("bol: denied.")
                    return True
            return await self._apply(parse_transcript(text))
        except BridgeError as exc:
            msg = f"Couldn't reach Claude: {exc}"
            print(f"bol: {msg}")
            await self._speak(msg)
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
        if action is Action.DICTATE:
            await self.bridge.inject(text + " ", submit=False)
            return True
        if action is Action.TYPE:
            await self.bridge.inject(text, submit=False)
            return True
        if action is Action.SEND:
            self._awaiting_permission = False
            await self.bridge.inject(text, submit=True)
            if self.cfg.sound_cues:
                await play_cue("done")
            print("bol: sent — Claude's turn.")
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
            print("bol: sleeping — press the hotkey when you need me.")
            return False
        if action is Action.REPEAT:
            if self._last_reply:
                await self._speak(self._last_reply)
            return True
        return False

    async def _speak(self, text: str) -> None:
        self._last_reply = text
        print(f"bol: {text}")
        await self.speaker.speak(text)

    # ---------------------------------------------------------------- hooks

    async def _on_tool(self, payload: dict) -> None:
        self.tracker.record_tool(payload)

    async def _on_stop(self, payload: dict) -> None:
        event = self.tracker.finish_turn(payload)
        reply = await self.summarizer.summarize(event)
        await self._speak(reply)
        await self._auto_listen()

    async def _on_notification(self, payload: dict) -> None:
        note = self.tracker.notification(payload)
        if note.notification_type == "permission_prompt":
            self._awaiting_permission = True
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
