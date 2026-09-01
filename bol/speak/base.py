from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from ..config import Config

log = logging.getLogger("bol.speak")

_CUES = {
    "listen": "/System/Library/Sounds/Pop.aiff",
    "done": "/System/Library/Sounds/Bottle.aiff",
    "discard": "/System/Library/Sounds/Basso.aiff",
}


class Speaker(Protocol):
    async def speak(self, text: str) -> None: ...
    async def stop(self) -> None: ...


async def play_cue(name: str) -> None:
    path = _CUES.get(name)
    if not path:
        return
    try:
        proc = await asyncio.create_subprocess_exec(
            "afplay", path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    except FileNotFoundError:
        pass


def build_speaker(cfg: Config) -> Speaker:
    if cfg.tts.engine == "kokoro":
        try:
            from .kokoro import KokoroSpeaker

            return KokoroSpeaker(cfg)
        except ImportError:
            log.warning("mlx-audio not installed (pip install 'bol[kokoro]'); using say")
    if cfg.tts.engine == "none":
        from .say import NullSpeaker

        return NullSpeaker()
    from .say import SaySpeaker

    return SaySpeaker(cfg)
