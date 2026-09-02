"""Hotkey tests: a mistyped key name and a missing Input Monitoring grant
both used to produce a listener that looked armed and never fired, and auto
mode has to tell a tap from a hold without a real clock.
"""

import asyncio
import types

import pytest
from pynput import keyboard

from bol.config import HotkeyConfig
from bol.hotkey import MouseListener, listener as listener_mod
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


def _hotkey(mode="push_to_talk", key="alt_r", tap_ms=400):
    events = []
    hk = HotkeyListener(
        HotkeyConfig(mode=mode, key=key, tap_ms=tap_ms),
        lambda: events.append("press"),
        lambda kind: events.append(f"release:{kind}"),
    )
    return hk, events


class FakeClock:
    """Drives the tap/hold decision without patching time out from under
    asyncio: the listener reads its clock through one module-level name."""

    def __init__(self, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr(listener_mod, "_monotonic", lambda: self.now)

    def advance(self, ms):
        self.now += ms / 1000


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


async def test_push_to_talk_dispatch(monkeypatch):
    _install(monkeypatch)
    hk, events = _hotkey()
    hk.start()

    hk._handle_press(keyboard.Key.alt_r)
    hk._handle_press(keyboard.Key.alt_r)  # key repeat while held, ignored
    hk._handle_press(keyboard.Key.alt_l)  # some other key, ignored
    hk._handle_release(keyboard.Key.alt_r)
    await asyncio.sleep(0)

    assert events == ["press", "release:hold"]


async def test_toggle_alternates(monkeypatch):
    _install(monkeypatch)
    hk, events = _hotkey(mode="toggle")
    hk.start()

    for _ in range(2):
        hk._handle_press(keyboard.Key.alt_r)
        hk._handle_release(keyboard.Key.alt_r)  # no-op in toggle mode
    await asyncio.sleep(0)

    assert events == ["press", "release:hold"]


async def test_push_to_talk_ignores_how_long_the_key_was_held(monkeypatch):
    # tap_ms belongs to auto mode only: push-to-talk must keep working the
    # way it always did, however briefly the key was down.
    _install(monkeypatch)
    clock = FakeClock(monkeypatch)
    hk, events = _hotkey(mode="push_to_talk")
    hk.start()

    hk._handle_press(keyboard.Key.alt_r)
    clock.advance(50)  # far under tap_ms
    hk._handle_release(keyboard.Key.alt_r)
    await asyncio.sleep(0)

    assert events == ["press", "release:hold"]


# ----------------------------------------------------------------------- auto


async def _tap(hk, clock, ms=120):
    hk._handle_press(keyboard.Key.alt_r)
    clock.advance(ms)
    hk._handle_release(keyboard.Key.alt_r)
    await asyncio.sleep(0)


async def test_auto_hold_releases_on_key_up(monkeypatch):
    _install(monkeypatch)
    clock = FakeClock(monkeypatch)
    hk, events = _hotkey(mode="auto")
    hk.start()

    hk._handle_press(keyboard.Key.alt_r)
    clock.advance(900)
    hk._handle_release(keyboard.Key.alt_r)
    await asyncio.sleep(0)

    assert events == ["press", "release:hold"]


async def test_auto_tap_keeps_the_recording_running(monkeypatch):
    _install(monkeypatch)
    clock = FakeClock(monkeypatch)
    hk, events = _hotkey(mode="auto")
    hk.start()

    await _tap(hk, clock)

    # "tap" tells the daemon to let the energy gate end this one, not to stop
    # recording the instant the key came up.
    assert events == ["press", "release:tap"]


async def test_a_second_tap_ends_the_tap_recording(monkeypatch):
    _install(monkeypatch)
    clock = FakeClock(monkeypatch)
    hk, events = _hotkey(mode="auto")
    hk.start()

    await _tap(hk, clock)
    await _tap(hk, clock)  # while the first recording is still running

    # No second "press": this tap ended the recording instead of starting one.
    assert events == ["press", "release:tap", "release:tap"]

    await _tap(hk, clock)  # and the key is armed again afterwards
    assert events == ["press", "release:tap", "release:tap", "press", "release:tap"]


async def test_a_hold_while_a_tap_recording_runs_still_ends_it(monkeypatch):
    _install(monkeypatch)
    clock = FakeClock(monkeypatch)
    hk, events = _hotkey(mode="auto")
    hk.start()

    await _tap(hk, clock)
    hk._handle_press(keyboard.Key.alt_r)
    clock.advance(900)  # held long, but the press was already spent
    hk._handle_release(keyboard.Key.alt_r)
    await asyncio.sleep(0)

    assert events == ["press", "release:tap", "release:tap"]


async def test_clear_tap_rearms_the_key(monkeypatch):
    # The recording endpointed on silence on its own, so the daemon tells the
    # listener. Without this the next press would be swallowed forever.
    _install(monkeypatch)
    clock = FakeClock(monkeypatch)
    hk, events = _hotkey(mode="auto")
    hk.start()

    await _tap(hk, clock)
    hk.clear_tap()
    await _tap(hk, clock)

    assert events == ["press", "release:tap", "press", "release:tap"]


# ------------------------------------------------------------- the mouse


class FakeMouse:
    """pynput's mouse.Listener minus the event tap."""

    IS_TRUSTED = True

    def __init__(self, on_click=None):
        self.on_click = on_click
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

    def click(self, pressed=True):
        self.on_click(10, 20, "left", pressed)


async def test_a_click_reaches_the_callback():
    clicks = []
    mouse = MouseListener(lambda: clicks.append(True), backend=FakeMouse)

    assert mouse.start() is True
    backend = mouse._listener
    backend.click()
    await asyncio.sleep(0)

    assert clicks == [True]
    assert backend.daemon is True  # never keeps the process alive
    mouse.stop()
    assert backend.stopped is True


async def test_the_release_half_of_a_click_is_not_a_second_click():
    # One click is one cancel. Reporting both halves would cancel the
    # recording and then cancel whatever replaced it.
    clicks = []
    mouse = MouseListener(lambda: clicks.append(True), backend=FakeMouse)
    mouse.start()

    mouse._listener.click(pressed=True)
    mouse._listener.click(pressed=False)
    await asyncio.sleep(0)

    assert clicks == [True]


async def test_a_mouse_listener_that_cannot_start_is_simply_not_there():
    # It shares a process with the hotkey, which is the thing that must never
    # break, so every failure here is a listener that is not running.
    def explode(on_click=None):
        raise OSError("no event tap for you")

    mouse = MouseListener(lambda: None, backend=explode)

    assert mouse.start() is False
    assert mouse.running is False
    mouse.stop()  # and stopping one that never started is inert


async def test_a_mouse_listener_without_input_monitoring_is_not_used():
    class Untrusted(FakeMouse):
        IS_TRUSTED = False

    mouse = MouseListener(lambda: None, backend=Untrusted)

    assert mouse.start() is False
    assert mouse.running is False
