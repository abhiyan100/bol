"""macOS `say` speaker: zero-dependency, instant, always available."""

from __future__ import annotations

import asyncio
import logging

from ..config import Config
from .base import clamp_speech

log = logging.getLogger("bol.speak")


class SaySpeaker:
    def __init__(self, cfg: Config) -> None:
        self._voice = cfg.tts.say_voice
        self._rate = cfg.tts.say_rate
        self._proc: asyncio.subprocess.Process | None = None
        self._warned = False

    async def speak(self, text: str) -> None:
        await self.stop()
        args = ["say", "-r", str(self._rate)]
        if self._voice:
            args += ["-v", self._voice]
        args.append(clamp_speech(text))
        # Held locally: a concurrent stop() clears self._proc mid-wait.
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._proc = proc
        code = await proc.wait()
        if self._proc is proc:
            self._proc = None
        # A bad voice name makes `say` exit non-zero and print nothing, so
        # Bol would be mute forever with no clue why. Negative codes are our
        # own barge-in terminate, not a failure.
        if code is not None and code > 0 and not self._warned:
            self._warned = True
            log.warning(
                "`say` exited %d with voice %r; check [tts] say_voice "
                "(list them with: say -v '?')",
                code,
                self._voice or "system default",
            )

    async def stop(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            self._proc = None


class NullSpeaker:
    async def speak(self, text: str) -> None:
        pass

    async def stop(self) -> None:
        pass
