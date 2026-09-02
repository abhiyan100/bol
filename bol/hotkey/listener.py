"""Global hotkey via pynput.

macOS: the terminal app running Bol needs Input Monitoring permission
(System Settings > Privacy & Security > Input Monitoring). Without it pynput
still starts cleanly and simply never delivers a key, so start() checks
IS_TRUSTED and raises PermissionError with the fix rather than leaving a
hotkey that looks armed and is dead.

Hold to talk, and nothing else: key down calls on_press, key up calls
on_release, and the recording lasts exactly as long as the key is held. There
are no modes and no tap: a tap is a hold that was over quickly, which is a
recording with nothing in it, and everything a tap used to do is a trigger
word now ("type", "send it", "scratch that").
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from pynput import keyboard

from ..config import HotkeyConfig

log = logging.getLogger("bol.hotkey")

# Enough of the pynput vocabulary to fix a typo without opening its docs.
_KEY_EXAMPLES = "alt_r, alt_l, cmd_r, ctrl_r, shift_r, f13"

NOT_TRUSTED = (
    "Input Monitoring is off, so the hotkey can't fire. "
    "System Settings > Privacy & Security > Input Monitoring > enable your "
    "terminal app, then restart Bol."
)


def _resolve(name: str):
    """Config key name to a pynput key. Raises on anything pynput doesn't
    know: KeyCode.from_char() validates nothing, so a typo like
    "right_option" would silently match no key for the whole session."""
    try:
        return keyboard.Key[name]
    except KeyError:
        pass
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(
        f"unknown hotkey key {name!r}. Use a named key ({_KEY_EXAMPLES}) "
        "or a single character."
    )


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
        # Key repeat fires press over and over while the key is held; one
        # press per hold is the whole contract.
        self._down = False
        self._listener: keyboard.Listener | None = None

    def _handle_press(self, key) -> None:
        if key != self._key or self._down:
            return
        self._down = True
        self._loop.call_soon_threadsafe(self._on_press)

    def _handle_release(self, key) -> None:
        if key != self._key:
            return
        self._down = False
        # Fired even for a release nobody pressed for (Bol started with the
        # key already down): the daemon's release is inert without a session.
        self._loop.call_soon_threadsafe(self._on_release)

    def start(self) -> None:
        listener = keyboard.Listener(
            on_press=self._handle_press, on_release=self._handle_release
        )
        listener.daemon = True
        listener.start()
        # wait() returns once the backend has set up (or failed to set up) its
        # event tap; IS_TRUSTED is only meaningful after that.
        listener.wait()
        if not getattr(listener, "IS_TRUSTED", True):
            listener.stop()
            raise PermissionError(NOT_TRUSTED)
        self._listener = listener
        log.info("hotkey armed: hold %s", self._cfg.key)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
