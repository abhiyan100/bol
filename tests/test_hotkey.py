"""Hotkey tests: a mistyped key name and a missing Input Monitoring grant
both used to produce a listener that looked armed and never fired, and the
one gesture there is (hold, then release) has to survive key repeat.
"""

import asyncio
import types

import pytest
from pynput import keyboard

from bol.config import HotkeyConfig
from bol.hotkey import listener as listener_mod
from bol.hotkey.listener import HotkeyListener, _resolve


class FakeListener:
    """pynput's Listener minus the event tap."""

    IS_TRUSTED = True

    def __init__(self, on_press=None, on_release=None):
        self.on_press = on_press
        self.on_release = on_release
        self.daemon = False
        self.started = False
        self.waited = False
        self.stopped = False

    def start(self):
        self.started = True

    def wait(self):
        self.waited = True

    def stop(self):
        self.stopped = True


def _install(monkeypatch, is_trusted=True):
    """Patch the keyboard module the listener imported; real Key/KeyCode so
    key resolution stays honest."""
    made = []

    class Listener(FakeListener):
        IS_TRUSTED = is_trusted

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            made.append(self)

    monkeypatch.setattr(
        listener_mod,
        "keyboard",
        types.SimpleNamespace(
            Key=keyboard.Key, KeyCode=keyboard.KeyCode, Listener=Listener
        ),
    )
    return made


def _hotkey(key="alt_r"):
    events = []
    hk = HotkeyListener(
        HotkeyConfig(key=key),
        lambda: events.append("press"),
        lambda: events.append("release"),
    )
    return hk, events


# ------------------------------------------------------------- key resolution


def test_resolve_named_key():
    assert _resolve("alt_r") is keyboard.Key.alt_r
    assert _resolve("f13") is keyboard.Key.f13


def test_resolve_single_character():
    assert _resolve("z") == keyboard.KeyCode.from_char("z")


def test_resolve_rejects_a_typo():
    # "right_option" is the plausible typo: pynput would have accepted it as
    # a KeyCode and then matched nothing for the rest of the session.
    with pytest.raises(ValueError) as err:
        _resolve("right_option")
    message = str(err.value)
    assert "right_option" in message
    assert "alt_r" in message  # the message names valid keys


# ------------------------------------------------------------ input monitoring


async def test_start_raises_without_input_monitoring(monkeypatch):
    made = _install(monkeypatch, is_trusted=False)
    hk, _ = _hotkey()

    with pytest.raises(PermissionError) as err:
        hk.start()

    message = str(err.value)
    assert "Input Monitoring" in message
    assert "System Settings > Privacy & Security > Input Monitoring" in message
    assert made[0].stopped  # dead listener cleaned up


async def test_start_arms_when_trusted(monkeypatch):
    made = _install(monkeypatch)
    hk, _ = _hotkey()

    hk.start()

    assert made[0].started and made[0].waited
    hk.stop()
    assert made[0].stopped


# ------------------------------------------------------------------- dispatch


async def test_hold_dispatch(monkeypatch):
    _install(monkeypatch)
    hk, events = _hotkey()
    hk.start()

    hk._handle_press(keyboard.Key.alt_r)
    hk._handle_press(keyboard.Key.alt_r)  # key repeat while held, ignored
    hk._handle_press(keyboard.Key.alt_l)  # some other key, ignored
    hk._handle_release(keyboard.Key.alt_r)
    await asyncio.sleep(0)

    assert events == ["press", "release"]


async def test_how_long_the_key_was_held_changes_nothing(monkeypatch):
    # There is one gesture, and a short hold is that gesture done quickly.
    # No clock, no threshold, and nothing a brief press means on its own.
    _install(monkeypatch)
    hk, events = _hotkey()
    hk.start()

    for _ in range(3):
        hk._handle_press(keyboard.Key.alt_r)
        hk._handle_release(keyboard.Key.alt_r)
    await asyncio.sleep(0)

    assert events == ["press", "release"] * 3


async def test_the_key_is_armed_again_after_every_release(monkeypatch):
    # Nothing in the daemon has to tell the listener a recording is over:
    # the next press starts the next one, always.
    _install(monkeypatch)
    hk, events = _hotkey()
    hk.start()

    hk._handle_press(keyboard.Key.alt_r)
    hk._handle_release(keyboard.Key.alt_r)
    hk._handle_press(keyboard.Key.alt_r)
    hk._handle_release(keyboard.Key.alt_r)
    await asyncio.sleep(0)

    assert events == ["press", "release", "press", "release"]
