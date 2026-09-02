"""Global hotkey via pynput.

macOS: the terminal app running Bol needs Input Monitoring permission
(System Settings > Privacy & Security > Input Monitoring). Without it pynput
still starts cleanly and simply never delivers a key, so start() checks
IS_TRUSTED and raises PermissionError with the fix rather than leaving a
hotkey that looks armed and is dead.

auto:         key down starts recording. A release under tap_ms was a tap, so
              on_release("tap") fires and the recording carries on until the
              speaker stops talking; a longer release was a hold, so
              on_release("hold") ends it. While a tap-started recording runs,
              the next press starts nothing and its release ends that
              recording instead. The daemon calls clear_tap() when the
              recording is over, so a turn that ended on its own can never
              swallow the next press.
push_to_talk: key down calls on_press, key up calls on_release("hold").
toggle:       each tap alternates on_press / on_release("hold").
"""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import Callable

from pynput import keyboard

from ..config import HotkeyConfig

log = logging.getLogger("bol.hotkey")

# Enough of the pynput vocabulary to fix a typo without opening its docs.
_KEY_EXAMPLES = "alt_r, alt_l, cmd_r, ctrl_r, shift_r, f13"

# Indirection so tests can drive the tap/hold decision with a fake clock
# instead of patching time.monotonic out from under asyncio.
_monotonic = monotonic

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
        on_release: Callable[[str], None],
    ) -> None:
        self._cfg = cfg
        self._key = _resolve(cfg.key)
        self._on_press = on_press
        self._on_release = on_release
        self._loop = asyncio.get_event_loop()
        self._down = False
        self._down_at = 0.0
        self._toggled = False
        # auto mode: a tap-started recording is still running.
        self._tap_active = False
        # auto mode: this press was spent ending that recording, so its
        # release must not be read as a fresh tap.
        self._consumed = False
        self._listener: keyboard.Listener | None = None

    def clear_tap(self) -> None:
        """The tap-started recording is over (it endpointed on silence, or
        failed). Called by the daemon so the next press starts a new one."""
        self._tap_active = False

    def _fire_release(self, kind: str) -> None:
        self._loop.call_soon_threadsafe(self._on_release, kind)

    def _handle_press(self, key) -> None:
        if key != self._key or self._down:
            return
        self._down = True
        self._down_at = _monotonic()
        if self._cfg.mode == "toggle":
            self._toggled = not self._toggled
            if self._toggled:
                self._loop.call_soon_threadsafe(self._on_press)
            else:
                self._fire_release("hold")
            return
        if self._cfg.mode == "auto" and self._tap_active:
            # Second tap on a running recording: this press ends it rather
            # than starting another, so on_press is deliberately skipped.
            self._consumed = True
            return
        self._consumed = False
        self._loop.call_soon_threadsafe(self._on_press)

    def _handle_release(self, key) -> None:
        if key != self._key:
            return
        held_ms = (_monotonic() - self._down_at) * 1000
        self._down = False
        if self._cfg.mode == "push_to_talk":
            self._fire_release("hold")
            return
        if self._cfg.mode != "auto":
            return  # toggle acts on the press
        if self._consumed:
            self._consumed = False
            self._tap_active = False
            self._fire_release("tap")
            return
        if held_ms < self._cfg.tap_ms:
            self._tap_active = True
            self._fire_release("tap")
        else:
            self._fire_release("hold")

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
        log.info("hotkey armed: %s (%s)", self._cfg.key, self._cfg.mode)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
