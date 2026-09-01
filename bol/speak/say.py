"""macOS `say` speaker — zero-dependency, instant, always available."""

from __future__ import annotations

import asyncio

from ..config import Config


class SaySpeaker:
    def __init__(self, cfg: Config) -> None:
        self._voice = cfg.tts.say_voice
        self._rate = cfg.tts.say_rate
        self._proc: asyncio.subprocess.Process | None = None

    async def speak(self, text: str) -> None:
        await self.stop()
        args = ["say", "-r", str(self._rate)]
        if self._voice:
            args += ["-v", self._voice]
        args.append(text)
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await self._proc.wait()
        self._proc = None

    async def stop(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            self._proc = None


class NullSpeaker:
    async def speak(self, text: str) -> None:
        pass

    async def stop(self) -> None:
        pass
