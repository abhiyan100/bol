"""Global hotkey via pynput.

macOS: the terminal app running Bol needs Input Monitoring permission
(System Settings → Privacy & Security → Input Monitoring). `bol doctor`
checks reachability.

push_to_talk: key down → on_press, key up → on_release.
toggle:       each tap alternates on_press / on_release.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from pynput import keyboard

from ..config import HotkeyConfig

log = logging.getLogger("bol.hotkey")


def _resolve(name: str):
    try:
        return getattr(keyboard.Key, name)
    except AttributeError:
        return keyboard.KeyCode.from_char(name)


class HotkeyListener:
    def __init__(
        self,
        cfg: HotkeyConfig,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> None:
        self._cfg = cfg
        self._key = _resolve(cfg.key)
        self._on_press = on_press
        self._on_release = on_release
        self._loop = asyncio.get_event_loop()
        self._down = False
        self._toggled = False
        self._listener: keyboard.Listener | None = None

    def _handle_press(self, key) -> None:
        if key != self._key or self._down:
            return
        self._down = True
        if self._cfg.mode == "toggle":
            self._toggled = not self._toggled
            cb = self._on_press if self._toggled else self._on_release
        else:
            cb = self._on_press
        self._loop.call_soon_threadsafe(cb)

    def _handle_release(self, key) -> None:
        if key != self._key:
            return
        self._down = False
        if self._cfg.mode == "push_to_talk":
            self._loop.call_soon_threadsafe(self._on_release)

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._handle_press, on_release=self._handle_release
        )
        self._listener.daemon = True
        self._listener.start()
        log.info("hotkey armed: %s (%s)", self._cfg.key, self._cfg.mode)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
