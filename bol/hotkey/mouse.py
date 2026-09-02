"""A click, anywhere, as a way out of a recording nobody started on purpose.

A hotkey recording ends when the key comes up: the user's hand is already on
the answer. A recording a trigger word started has no such handle, and the
commonest way people say "no, not that" to a microphone that opened itself is
to reach for the mouse and carry on with something else. So a click cancels
it, and the words are dropped rather than pasted somewhere nobody was looking.

Same pynput backend and the same Input Monitoring permission as the hotkey, so
this costs no new prompt. It is strictly optional all the same: every failure
here is one debug line and a listener that is simply not there, because the
hotkey is the thing that must never break and it shares a process with this.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

log = logging.getLogger("bol.hotkey")


class MouseListener:
    """Calls on_click() on the event loop the first time a button goes down.

    Presses only, not releases: one click is one cancel, and reporting both
    halves would cancel the recording and then cancel whatever replaced it.
    """

    def __init__(self, on_click: Callable[[], None], backend=None) -> None:
        self._on_click = on_click
        # Injected in tests; None means "really watch the mouse".
        self._backend = backend
        self._loop = asyncio.get_event_loop()
        self._listener = None

    def _handle(self, _x, _y, _button, pressed) -> None:
        # pynput's thread, not the loop. Same hop the hotkey makes, for the
        # same reason: everything on the other side of it touches daemon state.
        if not pressed:
            return
        try:
            self._loop.call_soon_threadsafe(self._on_click)
        except RuntimeError:
            pass  # the loop is closing

    def start(self) -> bool:
        """True if the mouse is being watched. Never raises."""
        try:
            listener = self._build()
            listener.daemon = True
            listener.start()
            wait = getattr(listener, "wait", None)
            if wait is not None:
                wait()
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            log.debug("no click-to-cancel (%s)", exc)
            return False
        if not getattr(listener, "IS_TRUSTED", True):
            # The hotkey listener raises on this; here it is not worth a line
            # the user can do anything with, because the hotkey's own message
            # already told them about Input Monitoring.
            listener.stop()
            log.debug("no click-to-cancel: Input Monitoring is off")
            return False
        self._listener = listener
        return True

    def _build(self):
        if self._backend is not None:
            return self._backend(on_click=self._handle)
        from pynput import mouse

        return mouse.Listener(on_click=self._handle)

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is None:
            return
        try:
            listener.stop()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            log.debug("could not stop the mouse listener: %s", exc)

    @property
    def running(self) -> bool:
        return self._listener is not None
