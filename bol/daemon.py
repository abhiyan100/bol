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
from .bridge import TmuxBridge, TmuxError
from .config import Config
from .grammar import Action, parse_transcript
from .hooks import HookServer, TurnTracker
from .hotkey import HotkeyListener
from .speak import build_speaker, play_cue
from .stt import build_transcriber
from .summarize import build_summarizer

log = logging.getLogger("bol")


class Daemon:
    def __init__(self, cfg: Config, text_mode: bool = False) -> None:
        self.cfg = cfg
        self.text_mode = text_mode
        self.bridge = TmuxBridge(cfg.bridge.pane, cfg.bridge.enter_delay_s)
        self.tracker = TurnTracker()
        self.server = HookServer(cfg.server.host, cfg.server.port)
        self.speaker = build_speaker(cfg)
        self.summarizer = build_summarizer(cfg)
        self.recorder = Recorder(cfg.audio)
        self.transcriber = None if text_mode else build_transcriber(cfg)
        self.hotkey: HotkeyListener | None = None

        self._listen_lock = asyncio.Lock()
        self._asleep = False
        self._last_reply = ""
        self._awaiting_permission = False

    # ---------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        pane = await self.bridge.attach()
        print(f"bol: attached to Claude in tmux pane {pane.pane_id} ({pane.target})")

        self.server.on("Stop", self._on_stop)
        self.server.on("PostToolUse", self._on_tool)
        self.server.on("Notification", self._on_notification)
        await self.server.start()
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

    # ---------------------------------------------------------------- listening

    def _hotkey_pressed(self) -> None:
        self._asleep = False
        asyncio.get_running_loop().create_task(self.speaker.stop())
        asyncio.get_running_loop().create_task(self._listen_once(until_silence=False))

    def _hotkey_released(self) -> None:
        self.recorder.request_stop()

    async def _listen_once(self, until_silence: bool) -> None:
        if self._listen_lock.locked():
            return
        async with self._listen_lock:
            if self.cfg.sound_cues:
                await play_cue("listen")
            audio = await self.recorder.record(until_silence=until_silence)
            if audio is None:
                log.debug("no speech captured")
                return
            assert self.transcriber is not None
            text = await self.transcriber.transcribe(audio, self.cfg.audio.sample_rate)
            if not text:
                return
            print(f"you: {text}")
            await self._handle_utterance(text)

    async def _auto_listen(self) -> None:
        if not self.cfg.hands_free or self._asleep or self.text_mode:
            return
        if self.transcriber is None:
            return
        await self._listen_once(until_silence=True)

    # ---------------------------------------------------------------- actions

    _YES = {"yes", "yeah", "yep", "approve", "go ahead", "do it"}
    _NO = {"no", "nope", "deny", "don't", "dont"}

    async def _handle_utterance(self, text: str) -> None:
        try:
            if self._awaiting_permission:
                norm = text.strip().lower().rstrip(".!,")
                if norm in self._YES:
                    self._awaiting_permission = False
                    await self.bridge.inject_keys("Enter")
                    print("bol: approved.")
                    return
                if norm in self._NO:
                    self._awaiting_permission = False
                    await self.bridge.inject_keys("Escape")
                    print("bol: denied.")
                    await self._auto_listen()
                    return
            await self._apply(parse_transcript(text))
        except TmuxError as exc:
            msg = f"Lost the Claude pane: {exc}"
            print(f"bol: {msg}")
            await self._speak(msg)

    async def _apply(self, parsed) -> None:
        action, text = parsed.action, parsed.text
        if action is Action.DICTATE:
            await self.bridge.inject(text + " ", submit=False)
            await self._auto_listen()
        elif action is Action.TYPE:
            await self.bridge.inject(text, submit=False)
            await self._auto_listen()
        elif action is Action.SEND:
            self._awaiting_permission = False
            await self.bridge.inject(text, submit=True)
            if self.cfg.sound_cues:
                await play_cue("done")
            print("bol: sent — Claude's turn.")
        elif action is Action.DISCARD:
            # C-u wipes Claude Code's input line.
            await self.bridge.inject_keys("C-u")
            if self.cfg.sound_cues:
                await play_cue("discard")
            await self._auto_listen()
        elif action is Action.INTERRUPT:
            await self.bridge.interrupt()
            await self._speak("Interrupted.")
        elif action is Action.SLEEP:
            self._asleep = True
            print("bol: sleeping — press the hotkey when you need me.")
        elif action is Action.REPEAT:
            if self._last_reply:
                await self._speak(self._last_reply)
            await self._auto_listen()

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
