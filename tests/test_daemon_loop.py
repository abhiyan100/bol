"""Regression tests for the listening loop: the races and dead paths the code
reviews confirmed (session-token stops, awake-window chaining, multi-session
hook affinity, mic failures), plus the two rules the loop is built on now:
nothing is ever sent by itself, and one-way mode loads nothing that speaks."""

import asyncio
import logging
import queue as _queue

import numpy as np
import pytest

from bol.audio.capture import RecordingSession
from bol import daemon as daemon_mod
from bol.bridge.focused import BridgeError, SubmitBlocked
from bol.config import Config
from bol.daemon import Daemon
from bol.wake import CANCEL, SEND, TYPE


class FakeRecorder:
    """Scripted recorder: returns queued 'audio' markers, no hardware."""

    def __init__(self, utterance_count, end_reason=""):
        self.remaining = utterance_count
        self.calls = []
        # How every recording this fake makes ends. Empty by default so the
        # tests that predate the rule keep asking only what they meant to.
        self.end_reason = end_reason

    def begin(self):
        return RecordingSession()

    async def record(self, session, until_silence):
        self.calls.append(until_silence)
        if session.stopped or self.remaining == 0:
            return None
        self.remaining -= 1
        if self.end_reason:
            session.note_end(self.end_reason)
        return np.zeros(16000, dtype=np.float32)


class FakeTranscriber:
    def __init__(self, texts):
        self.texts = list(texts)

    async def warmup(self):
        pass

    async def transcribe(self, audio, sample_rate):
        return self.texts.pop(0) if self.texts else ""


class FakeBridge:
    """The focused bridge, minus the app it could be wrong about.

    Every call carries explicit=: True when the user asked for this in words,
    False when it is Bol's own doing. The flag is recorded in call order,
    because where a keystroke is allowed to land depends on it.
    """

    def __init__(self):
        self.injected = []
        self.keys = []
        self.explicit = []

    async def inject(self, text, submit, explicit=False):
        self.explicit.append(explicit)
        self.injected.append((text, submit))

    async def inject_keys(self, *keys, explicit=False):
        self.explicit.append(explicit)
        self.keys.append(keys)

    async def interrupt(self):
        self.explicit.append(True)
        self.keys.append(("Escape",))


class FakeSpeaker:
    def __init__(self):
        self.spoken = []

    async def speak(self, text):
        self.spoken.append(text)

    async def stop(self):
        pass


class FakeSummarizer:
    async def summarize(self, event):
        return f"summary of {event.session_id}"


class FakeHud:
    """The pill, without a window: it just records what it was told to show."""

    def __init__(self, order=None):
        self.calls = []
        # Per-line hold overrides, parallel to calls: the hint after a paste
        # has to outstay "Sent", and that is the only evidence it does.
        self.holds = []
        self.order = order

    async def start(self):
        pass

    async def stop(self):
        pass

    def set(self, state, text="", detail="", hold=0.0):
        self.calls.append((state, text, detail))
        self.holds.append(hold)
        if self.order is not None:
            self.order.append(f"pill:{state}")

    @property
    def states(self):
        return [state for state, _text, _detail in self.calls]


class FakeWake:
    """The keyword listener, as far as the daemon's awake window is concerned.

    Only its presence matters: _touch_awake and _awake are both no-ops when
    there is no listener, so a test about the window needs one to exist.
    """

    def __init__(self):
        self.muted = 0
        self.unmuted = 0

    def mute(self):
        self.muted += 1

    def unmute(self):
        self.unmuted += 1


class Ticks:
    """A clock that moves a second every time it is read.

    The awake window is a minute long, so a scripted recorder that has run
    dry would keep the reopen loop spinning for a minute of real time. Here
    every check of the window costs a second of it instead.
    """

    def __init__(self, step=1.0):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def _daemon(utterances, texts, talk_back=True, clock=None, type_phrases=()):
    cfg = Config()
    cfg.ui.sounds = False
    # The short trigger word is off by default ([wake] type_phrases = []), so
    # "hey bol" is the one that starts a dictation. A test that wants the old
    # split back asks for it here, before the Daemon reads the phrase table.
    cfg.wake.type_phrases = list(type_phrases)
    # The loop tests are about the loop. Cleanup is "always" by default now,
    # and a real cleaner would load (and on a cold machine download) a 195 MB
    # model on every utterance; the cleanup tests below build their own fake.
    cfg.cleanup.mode = "off"
    cfg.cleanup.model = ""
    # Two-way by default here: most of these tests assert on what Bol says,
    # and one-way has nothing to say. The one-way tests build their own.
    cfg.talk_back = talk_back
    # The command window after a paste is its own feature with its own tests
    # below; everywhere else a paste must stay a paste, not paste plus listen.
    cfg.wake.command_window_s = 0
    d = Daemon(cfg, text_mode=False, clock=clock or Ticks(0.0))
    d.recorder = FakeRecorder(utterances)
    d.transcriber = FakeTranscriber(texts)
    d.bridge = FakeBridge()
    d.speaker = FakeSpeaker() if talk_back else None
    d.hud = FakeHud()
    return d


def _awake_daemon(utterances, texts):
    """A daemon with the awake window open, so the mic reopens after a turn."""
    d = _daemon(utterances, texts, clock=Ticks())
    d.cfg.wake.awake_s = 3.0
    d.wake = FakeWake()
    d._touch_awake()
    return d


@pytest.mark.asyncio
async def test_the_awake_window_chains_dictation_then_send():
    d = _awake_daemon(2, ["add a login test", "send it"])
    await d._listen_session(d.recorder.begin(), until_silence=False)
    # Both utterances were captured in ONE mic ownership (the reopen loop),
    # and the second one submitted.
    assert d.recorder.calls == [False, True]
    assert d.bridge.injected == [("add a login test ", False), ("", True)]


@pytest.mark.asyncio
async def test_send_closes_the_mic():
    d = _awake_daemon(5, ["ship it send", "should never be heard"])
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.recorder.calls == [False]
    assert d.bridge.injected == [("ship it", True)]


@pytest.mark.asyncio
async def test_a_closed_awake_window_never_chains():
    # No trigger words running, so nothing reopens the mic: one hold is one
    # recording, and the next one needs the key again.
    d = _daemon(3, ["hello there", "more words"])
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.recorder.calls == [False]


@pytest.mark.asyncio
async def test_sleep_stops_chaining():
    d = _awake_daemon(3, ["stop listening", "unheard"])
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.recorder.calls == [False]
    assert d._asleep is True


@pytest.mark.asyncio
async def test_release_before_record_stops_only_its_session():
    d = _daemon(1, ["unused"])
    session = d.recorder.begin()
    session.request_stop()  # released before the mic opened
    await d._listen_session(session, until_silence=False)
    # The stopped session yields no audio and nothing is injected...
    assert d.bridge.injected == []
    # ...and a FRESH session is unaffected by the stale stop.
    assert not d.recorder.begin().stopped


@pytest.mark.asyncio
async def test_stale_hotkey_release_is_inert():
    d = _daemon(1, ["unused"])
    d._ptt_session = None
    d._hotkey_released()  # no session in flight, must not raise or affect anything


@pytest.mark.asyncio
async def test_a_second_listen_while_the_mic_is_owned_is_dropped():
    # One queued press is enough; a second must return rather than queue up
    # behind the lock and open the mic again once the first is done.
    d = _daemon(1, ["hello"])
    async with d._listen_lock:
        d._pending_listen = True
        await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.recorder.calls == []


@pytest.mark.asyncio
async def test_permission_yes_presses_enter_and_yields_to_claude():
    d = _daemon(1, ["yes"])
    d._permission_session = "s1"
    reopen = await d._handle_utterance("yes")
    assert reopen is False
    assert ("Enter",) in d.bridge.keys


# ------------------------------------------------------------------ A1 sessions


def _stop(session_id, message="Claude finished the job it was given.", cwd="/w/repo"):
    return {
        "hook_event_name": "Stop",
        "session_id": session_id,
        "prompt_id": f"p-{session_id}",
        "last_assistant_message": message,
        "cwd": cwd,
    }


def _tool(session_id, command="pytest -q"):
    return {
        "hook_event_name": "PostToolUse",
        "session_id": session_id,
        "prompt_id": f"p-{session_id}",
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_response": {},
    }


class Dial:
    """A clock the test moves by hand, for the binding window."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _hook_daemon(clock=None):
    d = _daemon(0, [], clock=clock)
    d.summarizer = FakeSummarizer()
    return d


@pytest.mark.asyncio
async def test_only_the_first_session_is_narrated():
    d = _hook_daemon()
    await d._on_stop(_stop("A", cwd="/work/alpha"))
    await d._on_stop(_stop("B", cwd="/work/beta"))
    await d._on_stop(_stop("C", cwd="/work/gamma"))
    assert d.speaker.spoken == ["summary of A"]
    assert d._bound_session == "A"


@pytest.mark.asyncio
async def test_before_any_send_only_a_stop_binds_bol_to_a_session(capsys):
    # Hooks are user-scoped, so the first thing Bol hears is often an
    # unrelated session grinding through tool calls. That must not decide
    # who gets narrated; a Stop (someone who never dictates but wants
    # read-back) is the only event allowed to claim an unbound Bol.
    d = _hook_daemon()

    await d._on_tool(_tool("A", "rm -rf build"))
    assert d._bound_session is None
    assert d.hud.calls == []

    await d._on_stop(_stop("B", cwd="/work/beta"))

    assert d._bound_session == "B"
    assert d.speaker.spoken == ["summary of B"]
    assert "bol: narrating beta (Claude)." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_send_moves_the_narration_to_whoever_answers(capsys):
    # The bug in one test: Bol is stuck on an unrelated session, the user
    # dictates and says "send it", and the session that picks the turn up is
    # the one Bol should be narrating from then on.
    clock = Dial()
    d = _hook_daemon(clock)
    await d._on_stop(_stop("A", cwd="/work/other"))
    d.speaker.spoken.clear()
    capsys.readouterr()

    await d._handle_utterance("ship it send")  # the Enter the user asked for

    await d._on_tool(_tool("B"))  # B answers first, inside the window
    assert d._bound_session == "B"
    assert "bol: narrating B (Claude)." in capsys.readouterr().out

    await d._on_stop(_stop("B", cwd="/work/beta"))
    await d._on_stop(_stop("A", cwd="/work/other"))

    # B is narrated, A is not, and the window shut behind the first event.
    assert d.speaker.spoken == ["summary of B"]
    assert d._bound_session == "B"


@pytest.mark.asyncio
async def test_outside_the_window_a_foreign_session_draws_nothing():
    clock = Dial()
    d = _hook_daemon(clock)
    await d._on_stop(_stop("A", cwd="/work/alpha"))
    await d._handle_utterance("ship it send")
    clock.advance(d.cfg.server.bind_window_s + 1.0)
    d.hud.calls.clear()
    d.speaker.spoken.clear()

    await d._on_tool(_tool("B"))
    await d._on_stop(_stop("B", cwd="/work/beta"))
    await d._on_notification(
        {
            "hook_event_name": "Notification",
            "session_id": "B",
            "notification_type": "permission_prompt",
            "message": "Claude wants to run rm -rf.",
        }
    )

    assert d.hud.calls == []
    assert d.speaker.spoken == []
    assert d._permission_session is None
    assert d._bound_session == "A"


@pytest.mark.asyncio
async def test_follow_all_still_narrates_everything_and_binds_nothing():
    clock = Dial()
    d = _hook_daemon(clock)
    d.cfg.server.follow = "all"

    await d._on_tool(_tool("A"))
    await d._on_stop(_stop("A"))
    clock.advance(600.0)  # no send, no window, and it makes no difference
    await d._on_stop(_stop("B"))

    assert d.speaker.spoken == ["summary of A", "summary of B"]
    assert d._bound_session is None
    assert "thinking" not in d.hud.states


@pytest.mark.asyncio
async def test_follow_all_narrates_every_session():
    d = _hook_daemon()
    d.cfg.server.follow = "all"
    await d._on_stop(_stop("A"))
    await d._on_stop(_stop("B"))
    await d._on_stop(_stop("C"))
    assert d.speaker.spoken == ["summary of A", "summary of B", "summary of C"]


@pytest.mark.asyncio
async def test_permission_from_a_foreign_session_never_presses_enter():
    d = _hook_daemon()
    await d._on_stop(_stop("A"))  # Bol latches onto session A
    await d._on_notification(
        {
            "hook_event_name": "Notification",
            "session_id": "B",
            "notification_type": "permission_prompt",
            "message": "Claude wants to run rm -rf.",
        }
    )
    # Session B's prompt was never announced, so nothing is armed...
    assert d._permission_session is None
    reopen = await d._handle_utterance("yes")
    # ...and "yes" is plain dictation, not an approval keystroke.
    assert ("Enter",) not in d.bridge.keys
    assert reopen is True


@pytest.mark.asyncio
async def test_permission_answer_checks_the_prompting_session():
    d = _hook_daemon()
    await d._on_stop(_stop("A"))
    d._permission_session = "B"  # e.g. armed under follow="all", then rebound
    reopen = await d._handle_utterance("yes")
    assert ("Enter",) not in d.bridge.keys
    assert reopen is True
    assert d._permission_session is None


@pytest.mark.asyncio
async def test_ignored_session_still_drains_its_tool_log():
    d = _hook_daemon()
    await d._on_stop(_stop("A"))
    d.tracker.record_tool(
        {"prompt_id": "p-B", "tool_name": "Edit",
         "tool_input": {"file_path": "/x.py"}, "tool_response": {}}
    )
    await d._on_stop(_stop("B"))
    assert d.tracker._tools == {}


# ------------------------------------------------------------------ A3 failures


class ExplodingRecorder(FakeRecorder):
    """Fails the first record() the way a yanked input device does."""

    def __init__(self, utterance_count):
        super().__init__(utterance_count)
        self.exploded = False

    async def record(self, session, until_silence):
        if not self.exploded:
            self.exploded = True
            self.calls.append(until_silence)
            raise OSError("PortAudioError: device unavailable")
        return await super().record(session, until_silence)


@pytest.mark.asyncio
async def test_mic_failure_is_spoken_and_the_hotkey_survives():
    d = _daemon(1, ["add a login test"])
    d.recorder = ExplodingRecorder(1)
    d.transcriber = FakeTranscriber(["add a login test"])
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.speaker.spoken == ["Lost the microphone. Check your input device."]
    assert d.bridge.injected == []
    # The next press still records and injects: the loop is not wedged.
    await d._listen_session(d.recorder.begin(), until_silence=False)
    assert d.bridge.injected == [("add a login test ", False)]


@pytest.mark.asyncio
async def test_empty_dictation_is_not_injected():
    d = _daemon(1, ["clean it up"])
    reopen = await d._handle_utterance("clean it up")
    assert reopen is True
    assert d.bridge.injected == []


# ------------------------------------------------ nothing sends by itself


@pytest.mark.asyncio
async def test_dictation_is_pasted_and_waits():
    d = _daemon(1, ["add a login test"])
    reopen = await d._handle_utterance("add a login test")
    assert d.bridge.injected == [("add a login test ", False)]
    assert d._pending_paste is True
    assert reopen is True


@pytest.mark.asyncio
async def test_a_long_finished_instruction_is_still_only_pasted():
    # The rule with no exceptions: length, wording and how the recording
    # ended change nothing. An Enter is the user saying so, or it does not
    # happen.
    d = _daemon(1, ["refactor the auth module and add tests for the parser"])
    await d._handle_utterance("refactor the auth module and add tests for the parser")
    assert d.bridge.injected == [
        ("refactor the auth module and add tests for the parser ", False)
    ]


@pytest.mark.asyncio
async def test_send_it_submits_however_short_the_text_is():
    d = _daemon(1, ["go send it"])
    await d._handle_utterance("go send it")
    assert d.bridge.injected == [("go", True)]


@pytest.mark.asyncio
async def test_type_prefix_pastes_the_literal_text():
    d = _daemon(1, ["type hello world there"])
    reopen = await d._handle_utterance("type hello world there")
    assert d.bridge.injected == [("hello world there", False)]
    assert reopen is True


@pytest.mark.asyncio
async def test_scratch_that_discards():
    d = _daemon(1, ["scratch that"])
    await d._handle_utterance("scratch that")
    assert d.bridge.injected == []
    assert d.bridge.keys == [("C-u",)]


@pytest.mark.asyncio
async def test_close_discards_too():
    d = _daemon(1, ["close"])
    await d._handle_utterance("close")
    assert d.bridge.keys == [("C-u",)]


@pytest.mark.asyncio
async def test_interrupt_still_interrupts():
    d = _daemon(1, ["stop claude"])
    await d._handle_utterance("stop claude")
    assert d.bridge.keys == [("Escape",)]
    assert d.bridge.injected == []


# ----------------------------------------------------------------------- cues


class OrderedRecorder(FakeRecorder):
    def __init__(self, order):
        super().__init__(1)
        self.order = order

    async def record(self, session, until_silence):
        self.order.append("record")
        return await super().record(session, until_silence)


@pytest.mark.asyncio
async def test_the_listen_cue_never_delays_the_microphone(monkeypatch):
    # The regression: play_cue awaits afplay's proc.wait(), so awaiting it
    # before record() put the whole chime in front of the open mic.
    order = []

    async def fake_cue(name):
        order.append(f"cue:{name}")

    monkeypatch.setattr(daemon_mod, "play_cue", fake_cue)
    d = _daemon(1, ["add a login test"])
    d.cfg.ui.sounds = True
    d.recorder = OrderedRecorder(order)

    await d._listen_session(d.recorder.begin(), until_silence=False)
    await asyncio.sleep(0)  # let the fired-and-forgotten cue task run

    assert order == ["record", "cue:listen"]


@pytest.mark.asyncio
async def test_the_done_cue_never_delays_the_paste(monkeypatch):
    order = []

    async def fake_cue(name):
        order.append(f"cue:{name}")

    monkeypatch.setattr(daemon_mod, "play_cue", fake_cue)
    d = _daemon(1, ["add a login test send it"])
    d.cfg.ui.sounds = True

    class OrderedBridge(FakeBridge):
        async def inject(self, text, submit, explicit=False):
            order.append("inject")
            await super().inject(text, submit, explicit=explicit)

    d.bridge = OrderedBridge()
    await d._handle_utterance("add a login test send it")
    await asyncio.sleep(0)

    assert order == ["inject", "cue:done"]


# --------------------------------------------------------------- hold to talk


@pytest.mark.asyncio
async def test_a_release_ends_the_recording_it_started():
    d = _daemon(1, ["unused"])
    session = d.recorder.begin()
    d._ptt_session = session

    d._hotkey_released()

    assert session.stopped is True
    assert session.end_reason == "release"
    # A hold never hands the ending over to the speech gate: the key is the
    # gate, which is the whole point of holding it.
    assert session.until_silence is False
    assert d._ptt_session is None


@pytest.mark.asyncio
async def test_a_release_after_the_recording_ended_is_inert():
    # The race: the utterance hit the cap, or the mic failed, before the key
    # came up. Nothing to stop, and nothing to break.
    d = _daemon(1, ["unused"])
    d._ptt_session = None

    d._hotkey_released()

    assert d._ptt_session is None


@pytest.mark.asyncio
async def test_a_finished_recording_forgets_its_session():
    d = _daemon(1, ["add a login test"])
    session = d.recorder.begin()
    d._ptt_session = session

    await d._listen_session(session, until_silence=False)

    assert d._ptt_session is None


@pytest.mark.asyncio
async def test_a_hold_pastes_without_enter_and_leaves_it_pending():
    # The whole gesture: hold, talk, let go. The words land where the cursor
    # is, no Enter is pressed, and the pill says how to send them.
    d = _awake_daemon(1, ["add a login test"])
    d.recorder = FakeRecorder(1, end_reason="release")
    session = d.recorder.begin()
    d._ptt_session = session

    await d._listen_session(session, until_silence=False)

    assert d.bridge.injected == [("add a login test ", False)]
    assert d.bridge.explicit == [False]        # a paste claims no intent
    assert d._pending_paste is True
    # The pill goes away the moment the words land: no blue hint, no hold.
    assert "sending" not in d.hud.states
    assert d.hud.states[-1] == "idle"
    # And the words heard kept the microphone awake, so the reopen loop went
    # round again instead of handing the turn back to the key. (How many more
    # times is the Ticks clock's business, not this test's.)
    assert d.recorder.calls[:2] == [False, True]


@pytest.mark.asyncio
async def test_send_it_after_a_hold_paste_presses_enter():
    d = _daemon(2, ["add a login test", "send it"])

    await d._handle_utterance("add a login test")
    reopen = await d._handle_utterance("send it")

    assert d.bridge.injected == [("add a login test ", False), ("", True)]
    assert d.bridge.explicit == [False, True]  # the Enter is the user's word
    assert d._pending_paste is False
    assert reopen is False                     # the turn is Claude's now


# -------------------------------------------------------------------- defaults


def test_talk_back_is_off_by_default():
    # One-way is the whole loop most people want, and it is the cheap half.
    assert Config().talk_back is False


# ------------------------------------------------------------------ the pill


@pytest.mark.asyncio
async def test_the_pill_shows_listening_on_the_keystroke():
    # The whole point of the pill: the user presses the key and sees it, with
    # nothing (prewarm, mic, TTS stop) allowed in front.
    order = []
    d = _daemon(1, ["add a login test"])
    d.recorder = OrderedRecorder(order)
    d.hud = FakeHud(order)

    d._hotkey_pressed()

    assert order == ["pill:listening"]
    assert d.hud.calls[0] == ("listening", "Listening", "")

    await asyncio.sleep(0.05)  # let the press's own tasks finish
    assert order[:3] == ["pill:listening", "record", "pill:finalizing"]


@pytest.mark.asyncio
async def test_the_pill_says_finalizing_and_then_gets_out_of_the_way():
    d = _daemon(1, ["add a login test"])

    await d._listen_session(d.recorder.begin(), until_silence=False)

    # The words are in the box, which the user can see. Nothing is left on
    # screen saying so.
    assert d.hud.states == ["finalizing", "idle"]
    assert d.hud.calls[-1] == ("idle", "", "")


@pytest.mark.asyncio
async def test_the_pill_stays_hidden_when_the_turn_goes_to_claude():
    # "if it hears send it, it just sends, the pill should stop." The chime
    # is the receipt; the screen says nothing at all.
    d = _daemon(1, ["ship it send"])
    await d._handle_utterance("ship it send")
    assert d.hud.calls == [("idle", "", "")]


@pytest.mark.asyncio
async def test_a_tool_the_agent_ran_never_reaches_the_pill():
    # There is nothing on screen while the agent works. A tool call is not
    # something the user has to act on, and a "Thinking" pill bouncing at
    # every one of them was noise. The tool log still feeds the summary.
    d = _hook_daemon()
    await d._on_stop(_stop("A"))  # narrating A, so this is not a filter test
    d.hud.calls.clear()

    await d._on_tool(
        {
            "session_id": "A",
            "prompt_id": "p-A2",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"},
            "tool_response": {},
        }
    )

    assert d.hud.calls == []
    assert "thinking" not in d.hud.states
    # ... and the tool is still on the turn, for the summary to read.
    assert [t.tool_name for t in d.tracker._tools["p-A2"]] == ["Bash"]


@pytest.mark.asyncio
async def test_another_sessions_tool_never_reaches_the_pill():
    d = _hook_daemon()
    await d._on_stop(_stop("A"))  # Bol latches onto session A
    d.hud.calls.clear()

    await d._on_tool(
        {
            "session_id": "B",
            "prompt_id": "p-B",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "tool_response": {},
        }
    )

    assert d.hud.calls == []
    # And the pill's read of the session never binds Bol to one.
    assert d._bound_session == "A"


@pytest.mark.asyncio
async def test_a_permission_prompt_becomes_a_question_on_the_pill():
    d = _hook_daemon()
    await d._on_notification(
        {
            "hook_event_name": "Notification",
            "session_id": "A",
            "notification_type": "permission_prompt",
            "message": "Claude wants to run rm -rf build.",
        }
    )
    # The pill keeps the question, not the whole spoken sentence. Bol then
    # opens the microphone once for the answer, and when nobody says anything
    # the question goes back up: it is still outstanding.
    question = ("permission", "Claude wants to run rm -rf build.", "")
    assert d.hud.calls == [question, ("listening", "Listening", ""), ("idle", "", ""), question]
    assert d.hud.calls[-1] == question


@pytest.mark.asyncio
async def test_the_pill_follows_bol_speaking_and_then_clears():
    d = _hook_daemon()
    await d._speak("Tests pass, all thirty of them.")
    assert d.hud.calls == [
        ("speaking", "Tests pass, all thirty of them.", ""),
        ("idle", "", ""),
    ]


@pytest.mark.asyncio
async def test_a_mic_failure_turns_the_pill_red_and_leaves_it_up():
    d = _daemon(1, ["add a login test"])
    d.recorder = ExplodingRecorder(1)

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.hud.calls == [("error", "Lost the microphone. Check your input device.", "")]
    # No idle behind it: the remedy has to stay readable, and the pill takes
    # itself down after three seconds.
    assert "idle" not in d.hud.states


@pytest.mark.asyncio
async def test_a_transcription_failure_turns_the_pill_red():
    class DeafTranscriber(FakeTranscriber):
        async def transcribe(self, audio, sample_rate):
            raise RuntimeError("model went away")

    d = _daemon(1, ["unused"])
    d.transcriber = DeafTranscriber([])

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.hud.calls == [
        ("finalizing", "Finalizing", ""),
        ("error", "Couldn't transcribe that one. Try again.", ""),
    ]


# ------------------------------------------------------------- live words


class LiveRecorder(FakeRecorder):
    """A recorder that takes long enough for partials to arrive, and that
    remembers whether the daemon handed the recording a tap."""

    def __init__(self, utterance_count, pause=0.02):
        super().__init__(utterance_count)
        self.pause = pause
        self.taps = []

    async def record(self, session, until_silence):
        self.taps.append(session.tap)
        await asyncio.sleep(self.pause)
        return await super().record(session, until_silence)


class LiveTranscriber(FakeTranscriber):
    """Reports scripted partials, then waits for the daemon's sentinel the
    way the real streaming loop does."""

    def __init__(self, texts, partials=(("add a", "login"), ("add a login", "test"))):
        super().__init__(texts)
        self.partials = list(partials)
        self.stream_args = []
        self.ended = False
        self.fail = False

    async def stream(self, blocks, emit, *, context_size, chunk_ms, sample_rate):
        self.stream_args.append((context_size, chunk_ms, sample_rate))
        if self.fail:
            raise RuntimeError("this model does not stream")
        for committed, draft in self.partials:
            emit(committed, draft)
            await asyncio.sleep(0)
        while True:
            try:
                item = blocks.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.001)
                continue
            if item is None:
                self.ended = True
                return


def _live_daemon(monkeypatch, texts, partials=None, pause=0.02):
    d = _daemon(1, list(texts))
    # One recording, so the assertions are about one pill and one decode.
    # The throttle has its own tests; here it must not swallow a partial.
    monkeypatch.setattr(daemon_mod, "LIVE_PILL_HZ", 1000.0)
    d.recorder = LiveRecorder(1, pause=pause)
    kwargs = {} if partials is None else {"partials": partials}
    d.transcriber = LiveTranscriber(list(texts), **kwargs)
    return d


@pytest.mark.asyncio
async def test_the_pill_shows_words_while_you_are_still_talking(monkeypatch):
    d = _live_daemon(monkeypatch, ["add a login test for the parser"])

    await d._listen_session(d.recorder.begin(), until_silence=False)

    # Committed and draft arrive apart, so the panel can dim the unsure half.
    listening = [c for c in d.hud.calls if c[0] == "listening"]
    assert listening == [
        ("listening", "add a", "login"),
        ("listening", "add a login", "test"),
    ]
    # And they are on screen before the mic closes, not after.
    assert d.hud.states.index("finalizing") > d.hud.states.index("listening")


@pytest.mark.asyncio
async def test_partial_words_never_reach_claude(monkeypatch):
    # The whole safety property of the phase: the pill is a mirror, and the
    # bridge only ever sees the full-buffer decode.
    d = _live_daemon(monkeypatch, ["add a login test for the parser"])

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.bridge.injected == [("add a login test for the parser ", False)]
    await asyncio.sleep(0.01)  # the fake decoder polls its queue
    assert d.transcriber.ended  # the sentinel was delivered
    assert d.recorder.taps and d.recorder.taps[0] is not None


@pytest.mark.asyncio
async def test_the_live_decoder_gets_the_configured_stream_settings(monkeypatch):
    d = _live_daemon(monkeypatch, ["add a login test"])
    d.cfg.stt.stream_context = [256, 8]
    d.cfg.stt.stream_chunk_ms = 160

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.transcriber.stream_args == [([256, 8], 160, d.cfg.audio.sample_rate)]


@pytest.mark.asyncio
async def test_nothing_streamed_lands_on_the_pill_after_the_recording(monkeypatch):
    # A partial that arrives late would overwrite "Finalizing" with a
    # half-decoded sentence, right when the user is waiting on the real one.
    d = _live_daemon(monkeypatch, ["add a login test"], partials=[])

    async def stream(blocks, emit, **kwargs):
        while True:
            try:
                item = blocks.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.001)
                continue
            if item is None:
                # The decoder finishing one step behind the sentinel: this
                # word is already stale and must go nowhere.
                emit("too", "late")
                return

    d.transcriber.stream = stream
    await d._listen_session(d.recorder.begin(), until_silence=False)
    await asyncio.sleep(0.01)

    assert "listening" not in d.hud.states
    assert d.hud.states[0] == "finalizing"


@pytest.mark.asyncio
async def test_live_off_means_no_stream_and_no_tap(monkeypatch):
    d = _live_daemon(monkeypatch, ["add a login test"])
    d.cfg.stt.live = False

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.transcriber.stream_args == []
    assert d.recorder.taps == [None]
    assert d.bridge.injected == [("add a login test ", False)]


@pytest.mark.asyncio
async def test_a_transcriber_that_cannot_stream_is_simply_not_asked(monkeypatch):
    # [stt] engine = "none" leaves no transcriber at all, and an engine
    # without a streaming decoder must not break the recording either.
    d = _daemon(1, ["add a login test"])
    monkeypatch.setattr(daemon_mod, "LIVE_PILL_HZ", 1000.0)

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert not hasattr(d.transcriber, "stream")
    assert d.bridge.injected == [("add a login test ", False)]


@pytest.mark.asyncio
async def test_a_broken_live_decoder_costs_the_words_and_nothing_else(caplog):
    d = _daemon(1, ["add a login test"])
    d.recorder = LiveRecorder(1)
    d.transcriber = LiveTranscriber(["add a login test"])
    d.transcriber.fail = True

    with caplog.at_level(logging.DEBUG, logger="bol"):
        await d._listen_session(d.recorder.begin(), until_silence=False)

    assert "listening" not in d.hud.states
    assert d.bridge.injected == [("add a login test ", False)]
    failures = [r for r in caplog.records if "live words failed" in r.getMessage()]
    assert len(failures) == 1  # once, at debug, per recording


# ----------------------------------------------------------- the throttle


class FakeLoop:
    """Just the clock and the timer _LiveWords uses, under the test's thumb."""

    def __init__(self):
        self.now = 0.0
        self.timers = []

    def time(self):
        return self.now

    def call_later(self, delay, fn):
        timer = _Timer(fn)
        self.timers.append((self.now + delay, timer))
        return timer

    def advance(self, seconds):
        self.now += seconds
        due = [(at, t) for at, t in self.timers if at <= self.now]
        self.timers = [(at, t) for at, t in self.timers if at > self.now]
        for _at, timer in due:
            timer.fire()


class _Timer:
    def __init__(self, fn):
        self.fn = fn
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.fn()


def _live_words(interval=0.25):
    hud = FakeHud()
    loop = FakeLoop()
    return hud, loop, daemon_mod._LiveWords(hud, loop, interval)


def test_a_burst_of_partials_redraws_the_pill_once():
    hud, loop, live = _live_words()
    live.emit("add", "a")
    live.emit("add a", "login")
    live.emit("add a login", "test")

    assert hud.calls == [("listening", "add", "a")]  # the first one, at once


def test_the_last_partial_of_a_burst_is_always_delivered():
    hud, loop, live = _live_words()
    live.emit("add", "a")
    live.emit("add a", "login")
    live.emit("add a login", "test")
    loop.advance(0.25)

    # Not the middle one: the newest text is the only one worth showing.
    assert hud.calls == [
        ("listening", "add", "a"),
        ("listening", "add a login", "test"),
    ]


def test_closing_drops_the_pending_redraw_and_ends_the_decode():
    hud, loop, live = _live_words()
    live.blocks = _queue.Queue()
    live.emit("add", "a")
    live.emit("add a", "login")
    live.close()
    loop.advance(1.0)

    assert hud.calls == [("listening", "add", "a")]
    assert live.blocks is None
    assert live.emit("anything", "at all") is None
    assert hud.calls == [("listening", "add", "a")]


def test_closing_twice_is_harmless():
    hud, loop, live = _live_words()
    live.blocks = _queue.Queue()
    blocks = live.blocks
    live.close()
    live.close()

    assert blocks.get_nowait() is None
    assert blocks.empty()


def test_an_empty_partial_leaves_listening_alone():
    # Before the first word is decoded there is nothing to say, and blanking
    # the pill would read as the microphone having died.
    hud, loop, live = _live_words()
    live.emit("", "")

    assert hud.calls == []


# ------------------------------------------------------- A4 startup and the mic


class SilentTranscriber(FakeTranscriber):
    def __init__(self, boom=None):
        super().__init__([])
        self.boom = boom
        self.warmups = 0

    async def warmup(self):
        self.warmups += 1
        if self.boom is not None:
            raise self.boom


@pytest.mark.asyncio
async def test_the_startup_wait_puts_nothing_on_screen():
    # The user's rule: at `bol run` only the microphone turns on. The pill
    # first appears on a trigger word, a key press or a talk-back question.
    d = _daemon(0, [])
    d.transcriber = SilentTranscriber()

    await d._warm_speech_model()

    assert d.hud.calls == []
    assert d.transcriber.warmups == 1


@pytest.mark.asyncio
async def test_a_failed_warmup_does_not_leave_the_pill_loading_forever():
    d = _daemon(0, [])
    d.transcriber = SilentTranscriber(boom=RuntimeError("weights are missing"))

    with pytest.raises(RuntimeError):
        await d._warm_speech_model()

    assert d.hud.calls == []


@pytest.mark.asyncio
async def test_text_mode_has_no_speech_model_to_wait_for():
    d = _daemon(0, [])
    d.transcriber = None

    await d._warm_speech_model()

    assert d.hud.calls == []


class MissingMicRecorder(FakeRecorder):
    """A recorder whose configured device is not there any more."""

    def __init__(self, failures=1, label="AirPods Pro"):
        super().__init__(0)
        self.failures = failures
        self.device_label = label
        self.opens = 0
        self.defaulted = False

    async def open(self):
        self.opens += 1
        if self.opens <= self.failures:
            raise OSError("PortAudioError: device unavailable")

    def use_default_device(self):
        self.defaulted = True


@pytest.mark.asyncio
async def test_a_missing_mic_is_named_and_the_default_is_tried_once():
    d = _daemon(0, [])
    d.cfg.audio.input_device = "AirPods"
    d.recorder = MissingMicRecorder(failures=1)

    await d._open_microphone()

    assert d.hud.calls == [("error", "Mic lost: AirPods Pro", "")]
    assert d.recorder.defaulted is True
    assert d.recorder.opens == 2  # the named device, then the default one


@pytest.mark.asyncio
async def test_the_default_device_is_not_retried_against_itself():
    # Nothing was configured, so the open that just failed WAS the default.
    d = _daemon(0, [])
    d.cfg.audio.input_device = ""
    d.recorder = MissingMicRecorder(failures=1, label="the default input")

    await d._open_microphone()

    assert d.hud.calls == [("error", "Mic lost: the default input", "")]
    assert d.recorder.defaulted is False
    assert d.recorder.opens == 1


@pytest.mark.asyncio
async def test_no_microphone_at_all_still_arms_the_hotkey():
    # Both devices are gone. Bol says so and keeps going: the hotkey, the
    # hook server, and text still work, and the mic may come back.
    d = _daemon(0, [])
    d.cfg.audio.input_device = "AirPods"
    d.recorder = MissingMicRecorder(failures=2)

    await d._open_microphone()

    assert d.recorder.opens == 2
    assert d.hud.states == ["error"]


@pytest.mark.asyncio
async def test_a_microphone_that_opens_says_nothing():
    d = _daemon(0, [])
    d.recorder = MissingMicRecorder(failures=0)

    await d._open_microphone()

    assert d.recorder.opens == 1
    assert d.hud.calls == []


# --------------------------------------------------- every paste is pending

# The user feedback that started this: "it immediately sends". Nothing does
# any more. Whatever ended the recording, a dictation is pasted and waits, and
# the pill says the one phrase that finishes it.


@pytest.mark.asyncio
async def test_a_paste_is_a_paste_however_the_recording_ended():
    for reason in ("release", "silence", "window", "max", ""):
        d = _daemon(1, ["add a login test"])
        d.recorder = FakeRecorder(1, end_reason=reason)

        reopen = await d._handle_utterance("add a login test")

        assert d.bridge.injected == [("add a login test ", False)], reason
        assert d._pending_paste is True, reason
        assert reopen is True, reason


@pytest.mark.asyncio
async def test_the_pill_hides_itself_after_every_paste():
    d = _daemon(1, ["add a login test"])

    await d._handle_utterance("add a login test")

    assert d.hud.calls[-1] == ("idle", "", "")
    assert d.hud.holds[-1] == 0.0
    # The hint is gone entirely, not merely shortened.
    assert not hasattr(daemon_mod, "PASTE_HINT")


@pytest.mark.asyncio
async def test_a_typed_dictation_hides_the_pill_too():
    # "type ..." was always paste-only, and it takes its pill down like every
    # other paste: one rule, not two.
    d = _daemon(1, ["type hello world"])

    await d._handle_utterance("type hello world")

    assert d.bridge.injected == [("hello world", False)]
    assert d.hud.calls[-1] == ("idle", "", "")


@pytest.mark.asyncio
async def test_the_pause_hint_is_still_what_the_pill_would_show():
    # The one line left that the user has to read rather than glance at.
    from bol.hud.render import hold_for, label_for

    assert label_for("sending", daemon_mod.SLEEP_HINT) == daemon_mod.SLEEP_HINT
    assert hold_for("sending", daemon_mod.HINT_S) == 2.5
    assert daemon_mod.HINT_S == 2.5
    assert hold_for("sending") == 1.0  # every other "Sent" is unchanged


@pytest.mark.asyncio
async def test_a_pause_then_send_it_presses_enter():
    # The whole point of pasting instead of sending: the prompt waits, and
    # one phrase submits it.
    d = _daemon(2, ["add a login test", "send it"])

    await d._handle_utterance("add a login test")
    reopen = await d._handle_utterance("send it")

    assert d.bridge.injected == [("add a login test ", False), ("", True)]
    assert reopen is False


@pytest.mark.asyncio
async def test_a_whole_conversation_reaches_claude_only_on_send_it():
    # End to end, through the reopen loop: two things said in one mic
    # ownership, and Claude sees them only when the user says so.
    d = _awake_daemon(2, ["add a login test", "send it"])
    d.recorder = FakeRecorder(2, end_reason="silence")

    await d._listen_session(d.recorder.begin(), until_silence=True)

    assert d.recorder.calls == [True, True]
    assert d.bridge.injected == [("add a login test ", False), ("", True)]


@pytest.mark.asyncio
async def test_a_typed_line_is_pasted_too():
    # Text mode has no recording to end, and it changes nothing: the line is
    # pasted and waits for "send it" like everything else.
    d = _daemon(1, ["add a login test"])

    reopen = await d._handle_utterance("add a login test")

    assert d.bridge.injected == [("add a login test ", False)]
    assert reopen is True


# ------------------------------------------- the user's words vs Bol's doing

# Enter goes wherever the cursor is when the user asked for it out loud, and
# only into a Claude window otherwise. The daemon is the one place that knows
# which of the two a keystroke is, so this is where it is tested; what each
# kind is then allowed to do lives in tests/test_bridge.py.


def _explicit_daemon(utterances, texts):
    return _daemon(utterances, texts)


@pytest.mark.asyncio
async def test_a_spoken_send_it_is_an_explicit_enter():
    d = _explicit_daemon(1, ["add a login test send it"])

    await d._handle_utterance("add a login test send it")

    assert d.bridge.injected == [("add a login test", True)]
    assert d.bridge.explicit == [True]


@pytest.mark.asyncio
async def test_the_send_trigger_word_is_an_explicit_enter():
    d = _explicit_daemon(0, [])
    d._pending_paste = True

    await d._run_wake_command(SEND)

    assert d.bridge.keys == [("Enter",)]
    assert d.bridge.explicit == [True]
    assert d._pending_paste is False


@pytest.mark.asyncio
async def test_every_enter_bol_presses_is_one_the_user_asked_for():
    # There is no other kind left. A dictation never reaches the bridge with
    # submit=True unless the user said a send phrase.
    d = _explicit_daemon(1, ["add a login test"])

    await d._handle_utterance("add a login test")

    assert d.bridge.injected == [("add a login test ", False)]
    assert True not in d.bridge.explicit


@pytest.mark.asyncio
async def test_a_blocked_send_leaves_a_pending_paste():
    """The words were typed into the wrong window; the Enter was not."""

    class BlockedBridge(FakeBridge):
        async def inject(self, text, submit, explicit=False):
            self.explicit.append(explicit)
            if submit:
                raise SubmitBlocked("Notes isn't Claude", "Notes isn't Claude")
            self.injected.append((text, submit))

    d = _explicit_daemon(1, ["add a login test send it"])
    d.bridge = BlockedBridge()

    reopen = await d._handle_utterance("add a login test send it")

    assert d._pending_paste is True            # so "send it" can still finish it
    assert "didn't press Enter" in d.speaker.spoken[-1]
    assert reopen is True


@pytest.mark.asyncio
async def test_a_permission_answer_is_an_explicit_enter():
    d = _explicit_daemon(1, ["yes"])
    d._permission_session = "s1"

    await d._handle_utterance("yes")

    assert d.bridge.keys == [("Enter",)]
    assert d.bridge.explicit == [True]


@pytest.mark.asyncio
async def test_a_denied_permission_is_an_explicit_escape():
    d = _explicit_daemon(1, ["no"])
    d._permission_session = "s1"

    await d._handle_utterance("no")

    assert d.bridge.keys == [("Escape",)]
    assert d.bridge.explicit == [True]


@pytest.mark.asyncio
async def test_a_spoken_discard_is_explicit_so_it_works_outside_a_terminal():
    # C-u only clears a line in a terminal; the flag is what lets the bridge
    # turn it into the one Cmd+Z that undoes the paste in Notes or Slack.
    d = _explicit_daemon(1, ["scratch that"])

    await d._handle_utterance("scratch that")

    assert d.bridge.keys == [("C-u",)]
    assert d.bridge.explicit == [True]


@pytest.mark.asyncio
async def test_the_cancel_trigger_word_is_explicit_too():
    d = _explicit_daemon(0, [])
    d._pending_paste = True

    await d._run_wake_command(CANCEL)

    assert d.bridge.keys == [("C-u",)]
    assert d.bridge.explicit == [True]


@pytest.mark.asyncio
async def test_a_plain_paste_claims_no_intent():
    # A paste presses nothing, so it needs no permission to speak of: it is
    # allowed wherever the cursor is on its own terms.
    d = _explicit_daemon(1, ["add a login test"])

    await d._handle_utterance("add a login test")

    assert d.bridge.injected == [("add a login test ", False)]
    assert d.bridge.explicit == [False]


# ------------------------------------------------------- one-way and two-way


def _one_way(utterances=0, texts=()):
    """The default Bol: dictation in, nothing out loud."""
    d = _daemon(utterances, list(texts), talk_back=False)
    d.summarizer = None
    return d


def test_one_way_startup_builds_nothing_that_speaks(monkeypatch):
    # Not "switched off at the point of use": never built, so no voice model,
    # no summarizer and no mlx_lm.server on a 16 GB Mac.
    # The cleaner is faked: on a machine without mlx-lm (CI) build_cleaner
    # returns None, and this test is about wiring, not about the extra.
    import bol.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "build_cleaner", lambda cfg: object())
    cfg = Config()
    cfg.talk_back = False

    d = Daemon(cfg, text_mode=True)

    assert d.talk_back is False
    assert d.speaker is None
    assert d.engine is None
    assert d.summarizer is None
    # The cleanup model is dictation quality, not talk-back, so it stays.
    assert d.cleaner is not None


def test_two_way_builds_all_three():
    cfg = Config()
    cfg.talk_back = True

    d = Daemon(cfg, text_mode=True)

    assert d.talk_back is True
    assert d.speaker is not None
    assert d.engine is not None
    assert d.summarizer is not None


def test_the_startup_line_says_which_mode_this_is():
    one_way = Daemon(Config(), text_mode=True).mode_line()
    assert one_way == (
        "bol: one-way (dictation). Add --talk-back to hear what Claude did."
    )
    cfg = Config()
    cfg.talk_back = True
    assert Daemon(cfg, text_mode=True).mode_line() == "bol: two-way (talk-back on)."


@pytest.mark.asyncio
async def test_one_way_speak_prints_and_draws_instead_of_talking(capsys):
    d = _one_way()

    await d._speak("Tests pass, all thirty of them.")

    assert "Tests pass, all thirty of them." in capsys.readouterr().out
    assert d.hud.calls == [
        ("speaking", "Tests pass, all thirty of them.", ""),
        ("idle", "", ""),
    ]


@pytest.mark.asyncio
async def test_one_way_keeps_an_error_on_the_pill():
    d = _one_way()

    await d._speak("Lost the microphone. Check your input device.", state="error")

    assert d.hud.states == ["error"]  # no idle behind it: the remedy stays up


@pytest.mark.asyncio
async def test_one_way_says_nothing_at_all_when_claude_finishes():
    d = _one_way()

    await d._on_stop(_stop("A"))

    # Not a summary, not a pill, not a bound session, not a printed name.
    # One-way is dictation, and dictation has no opinion about a coding agent.
    assert d.hud.calls == []
    assert d._bound_session is None


@pytest.mark.asyncio
async def test_one_way_shows_neither_the_tool_nor_the_permission_question():
    # Pure dictation: no Thinking pill, no amber question, and nothing armed
    # to answer one with. In a real run these handlers are never even
    # registered, because one-way starts no hook server at all.
    d = _one_way()

    await d._on_tool(
        {
            "session_id": "A",
            "prompt_id": "p-A",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"},
            "tool_response": {},
        }
    )
    await d._on_notification(
        {
            "hook_event_name": "Notification",
            "session_id": "A",
            "notification_type": "permission_prompt",
            "message": "Claude wants to run rm -rf build.",
        }
    )
    await d._on_permission_request(
        {
            "hook_event_name": "PermissionRequest",
            "session_id": "A",
            "tool_name": "shell",
            "tool_input": {"command": "rm -rf build"},
        }
    )

    assert d.hud.calls == []
    assert d._permission_session is None


@pytest.mark.asyncio
async def test_one_way_starts_no_hook_server_and_registers_no_handlers():
    # The whole promise in one assertion: there is no server object, so there
    # is nothing to bind a port, write a token file, or hand a hook event to.
    cfg = Config()
    cfg.talk_back = False
    assert Daemon(cfg, text_mode=True).server is None

    cfg.talk_back = True
    assert Daemon(cfg, text_mode=True).server is not None


@pytest.mark.asyncio
async def test_one_way_pastes_dictation_the_same_way():
    d = _one_way(1, ["add a login test"])

    reopen = await d._handle_utterance("add a login test")

    assert d.bridge.injected == [("add a login test ", False)]
    assert d._pending_paste is True
    assert reopen is True


@pytest.mark.asyncio
async def test_one_way_barge_in_is_inert():
    # Nothing to interrupt, and the press must still open the microphone.
    d = _one_way(1, ["add a login test"])
    d._hush()
    d._prewarm()  # no engine to warm either

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert d.bridge.injected == [("add a login test ", False)]


# --------------------------------------------------- cleanup on every paste
#
# "the dictation itself without the cleanups is not that good, we have to be
# professional." So cleanup mode is "always" out of the box, and "always" has
# to mean every way a dictation can start.


class FakeCleaner:
    """The tuned cleanup model, minus the model."""

    def __init__(self, reply=None, boom=None):
        self.seen = []
        self.reply = reply
        self.boom = boom

    async def clean(self, text, deadline_s):
        self.seen.append((text, deadline_s))
        if self.boom is not None:
            raise self.boom
        return self.reply if self.reply is not None else text


def _clean_daemon(utterances, texts, cleaner=None):
    d = _daemon(utterances, texts)
    d.cfg.cleanup.mode = "always"
    d.cleaner = cleaner or FakeCleaner("Add a login test.")
    return d


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "said, injected",
    [
        # A hold, or a "type" trigger: both arrive as a plain dictation.
        ("add a login test", "Add a login test. "),
        # The spoken "type ..." command, which used to skip cleanup entirely.
        ("type add a login test", "Add a login test."),
        # And a dictation that ends by asking to be sent.
        ("add a login test send it", "Add a login test."),
    ],
)
async def test_cleanup_always_runs_before_the_paste(said, injected):
    d = _clean_daemon(1, [said])

    await d._handle_utterance(said)

    assert d.cleaner.seen, f"the cleaner never saw {said!r}"
    assert d.bridge.injected[0][0] == injected


@pytest.mark.asyncio
async def test_cleanup_runs_on_an_awake_follow_up_too():
    d = _awake_daemon(2, ["add a login test", "and a logout test as well"])
    d.cfg.cleanup.mode = "always"
    d.cleaner = FakeCleaner()

    await d._listen_session(d.recorder.begin(), until_silence=False)

    assert [text for text, _deadline in d.cleaner.seen][:2] == [
        "Add a login test",
        "And a logout test as well",
    ]


@pytest.mark.asyncio
async def test_cleanup_is_bounded_by_the_configured_deadline():
    d = _clean_daemon(1, ["add a login test"])
    d.cfg.cleanup.deadline_s = 1.25

    await d._handle_utterance("add a login test")

    assert d.cleaner.seen[0][1] == 1.25


class TitledBridge(FakeBridge):
    """A bridge that can say which window it is about to paste into."""

    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.reads = 0

    async def front_title(self):
        self.reads += 1
        return self.title


@pytest.mark.asyncio
async def test_the_window_title_spells_this_sessions_own_words():
    # The project name is in the title of the window the words are going into,
    # so a dictated "bowl" in a Bol window is Bol.
    d = _clean_daemon(1, ["the bowl daemon pastes it"], FakeCleaner())
    d.bridge = TitledBridge("Bol - claude - 180x48")

    await d._handle_utterance("the bowl daemon pastes it")

    assert d.cleaner.seen[0][0] == "The Bol daemon pastes it"
    assert d.bridge.injected == [("The Bol daemon pastes it ", False)]
    assert d.bridge.reads == 1


@pytest.mark.asyncio
async def test_a_vocabulary_word_is_one_of_this_sessions_words_too():
    d = _clean_daemon(1, ["the bowl daemon pastes it"], FakeCleaner())
    d.bridge = TitledBridge("")          # no title: the config is the source
    d.cfg.vocabulary.words = ["Bol"]

    await d._handle_utterance("the bowl daemon pastes it")

    assert d.cleaner.seen[0][0] == "The Bol daemon pastes it"


@pytest.mark.asyncio
async def test_a_paste_teaches_the_next_dictation_how_to_spell_it():
    # Nothing in the config and nothing in the title: the only evidence is
    # what the user themselves put in the box a moment ago.
    d = _clean_daemon(
        2, ["Refactor the Kokoro loader", "the kokora loader is slow"],
        FakeCleaner(),
    )
    d.bridge = TitledBridge("")

    await d._handle_utterance("Refactor the Kokoro loader")
    await d._handle_utterance("the kokora loader is slow")

    assert d.cleaner.seen[-1][0] == "The Kokoro loader is slow"


@pytest.mark.asyncio
async def test_a_bridge_that_cannot_name_its_window_still_cleans():
    class Mute(TitledBridge):
        async def front_title(self):
            raise BridgeError("no Automation permission")

    d = _clean_daemon(1, ["add a login test"], FakeCleaner())
    d.bridge = Mute()

    await d._handle_utterance("add a login test")

    assert d.bridge.injected == [("Add a login test ", False)]


@pytest.mark.asyncio
async def test_a_cleanup_failure_pastes_the_words_anyway():
    # The model is missing, or it hung, or it raised. Whatever the user said
    # still has to reach the box: cleanup is polish, never a gate.
    d = _clean_daemon(1, ["add a login test"], FakeCleaner(boom=RuntimeError("no model")))

    reopen = await d._handle_utterance("add a login test")

    assert d.bridge.injected == [("Add a login test ", False)]
    assert d._pending_paste is True
    assert reopen is True


@pytest.mark.asyncio
async def test_a_cleanup_that_never_returns_still_pastes():
    class Hangs:
        async def clean(self, text, deadline_s):
            await asyncio.sleep(30)
            return "never"

    d = _clean_daemon(1, ["add a login test"], Hangs())
    d.cfg.cleanup.deadline_s = 0.02

    await d._handle_utterance("add a login test")

    assert d.bridge.injected == [("Add a login test ", False)]


@pytest.mark.asyncio
async def test_cleanup_off_leaves_the_words_exactly_as_dictated():
    d = _daemon(1, ["add a login test"])   # the fixture turns cleanup off
    d.cleaner = FakeCleaner("SHOULD NEVER BE USED")

    await d._handle_utterance("add a login test")

    assert d.cleaner.seen == []
    assert d.bridge.injected == [("add a login test ", False)]


# ------------------------------------------------- what may open the pill
#
# "if the person wants to type again they say type, the pill waits and makes a
# sound, for like five seconds." Three things open the pill and nothing else
# does: a trigger word, the key, and a question Bol asked out loud.


class SessionRecorder(FakeRecorder):
    """Keeps the sessions it was handed, so a test can read their timings."""

    def __init__(self, utterance_count, end_reason=""):
        super().__init__(utterance_count, end_reason)
        self.sessions = []

    def begin(self):
        session = super().begin()
        self.sessions.append(session)
        return session


@pytest.fixture
def cues(monkeypatch):
    """Every cue Bol played, in order, without a speaker."""
    played = []

    async def fake_cue(name):
        played.append(name)

    monkeypatch.setattr(daemon_mod, "play_cue", fake_cue)
    return played


@pytest.mark.asyncio
async def test_the_key_shows_a_listening_pill_and_plays_a_cue(cues):
    d = _daemon(1, ["add a login test"])
    d.cfg.ui.sounds = True

    d._hotkey_pressed()
    assert d.hud.calls[0] == ("listening", "Listening", "")

    await asyncio.sleep(0.05)
    assert cues[0] == "listen"


@pytest.mark.asyncio
async def test_saying_hey_bol_shows_a_listening_pill_and_plays_a_cue(cues):
    d = _daemon(1, ["add a login test"])
    d.cfg.ui.sounds = True
    d.wake = FakeWake()

    d._wake_detected(0.6, "hey bol")
    assert d.hud.calls[0] == ("listening", "Listening", "")

    await asyncio.sleep(0.05)
    assert cues[0] == "listen"


@pytest.mark.asyncio
async def test_a_trigger_word_waits_speak_window_ms_and_then_hides():
    # Nothing said inside the window: the pill goes away and stays away.
    d = _daemon(0, [], type_phrases=["type"])
    d.recorder = SessionRecorder(0)
    d.wake = FakeWake()

    d._wake_detected(0.6, "type")
    await asyncio.sleep(0.05)

    assert d.recorder.sessions[0].window_ms == d.cfg.wake.speak_window_ms == 5000
    assert d.recorder.sessions[0].silence_ms == d.cfg.wake.pause_ms == 2000
    assert d.hud.states == ["listening", "idle"]


@pytest.mark.asyncio
async def test_room_noise_never_opens_the_pill():
    # The regression this exists for: with a rolling awake window, speech in
    # the room kept reopening the microphone and the pill "came back again
    # and again". awake_s is 0 now, so a listen that hears nothing is the end
    # of it, and a capture nobody started paints nothing.
    d = _daemon(0, [])
    d.wake = FakeWake()

    assert d.cfg.wake.awake_s == 0.0
    assert d._awake() is False
    assert d._reopens(daemon_mod.QUIET) is False

    await d._capture_and_handle(d.recorder.begin(), until_silence=True)

    assert d.hud.states == ["idle"]
    assert "listening" not in d.hud.states


# ------------------------------------------ the one window after Bol speaks


@pytest.mark.asyncio
async def test_a_spoken_summary_opens_the_microphone_exactly_once():
    d = _hook_daemon()
    d.recorder = SessionRecorder(4)
    d.transcriber = FakeTranscriber(["and now add a logout test", "unheard"])
    # Even with the awake window deliberately switched back on, the window
    # after Bol speaks is one window and not a rolling one.
    d.cfg.wake.awake_s = 60.0
    d.wake = FakeWake()

    await d._on_stop(_stop("A"))

    assert d.recorder.calls == [True]
    assert d.recorder.sessions[0].window_ms == d.cfg.wake.speak_window_ms
    assert d.bridge.injected == [("and now add a logout test ", False)]


@pytest.mark.asyncio
async def test_a_summary_nobody_answers_leaves_a_blank_screen():
    d = _hook_daemon()

    await d._on_stop(_stop("A"))

    assert d.recorder.calls == [True]      # one window ...
    assert d.hud.states[-1] == "idle"      # ... and then nothing on screen
    assert d.bridge.injected == []


@pytest.mark.asyncio
async def test_one_way_opens_no_window_after_anything():
    # There is no summary in one-way, so there is nothing to answer and the
    # microphone is never opened on Bol's own say-so.
    d = _one_way(2, ["never heard"])

    await d._on_stop(_stop("A"))
    await d._follow_up_listen()

    assert d.recorder.calls == []
    assert d.hud.calls == []


# ------------------------------------------------ the command window after a paste


def _window_daemon(utterances, texts, window_s=10.0, type_phrases=()):
    """One-way, no pill, wake ear faked: the daemon that types and then waits
    for a bare command with nothing on screen."""
    d = _daemon(
        utterances, texts, talk_back=False, clock=Ticks(),
        type_phrases=type_phrases,
    )
    d.summarizer = None
    d.wake = FakeWake()
    d.cfg.wake.command_window_s = window_s
    return d


async def _window_done(d):
    task = d._command_task
    if task is not None:
        await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_a_paste_opens_a_command_window_that_hears_send_it():
    d = _window_daemon(2, ["add a login test", "Send it."])
    await d._listen_session(d._begin(TYPE), until_silence=True, trigger=TYPE)
    await _window_done(d)
    # A spoken send is an empty paste plus Enter, the way SEND always was.
    assert d.bridge.injected == [("add a login test ", False), ("", True)]
    assert d.bridge.explicit[-1] is True
    assert d._pending_paste is False
    # Two recordings: the dictation, then the invisible command listen.
    assert d.recorder.calls == [True, True]
    # The window drew nothing: after the paste the pill only went idle.
    assert "listening" not in d.hud.states[1:]


@pytest.mark.asyncio
async def test_keyword_and_window_together_press_enter_once():
    d = _window_daemon(2, ["add a login test", "send it"])
    await d._listen_session(d._begin(TYPE), until_silence=True, trigger=TYPE)
    # The keyword ear fires while the window is still recording the same words.
    d._wake_command(SEND)
    await _window_done(d)
    assert d.bridge.keys == [("Enter",)]


@pytest.mark.asyncio
async def test_a_plain_sentence_in_the_window_pastes_nothing():
    d = _window_daemon(2, ["add a login test", "so anyway the weather is nice"])
    await d._listen_session(d._begin(TYPE), until_silence=True, trigger=TYPE)
    await _window_done(d)
    assert d.bridge.injected == [("add a login test ", False)]
    assert d.bridge.keys == []
    assert d._pending_paste is True  # still waiting for a real "send it"


@pytest.mark.asyncio
async def test_hey_bol_inside_the_window_starts_the_next_dictation():
    d = _window_daemon(
        3, ["add a login test", "hey bol and run pytest", "send it"]
    )
    await d._listen_session(d._begin(TYPE), until_silence=True, trigger=TYPE)
    await _window_done(d)
    assert d.bridge.injected == [
        ("add a login test ", False),
        ("and run pytest ", False),
        ("", True),
    ]
    assert d._pending_paste is False


@pytest.mark.asyncio
async def test_a_misheard_hey_bol_chains_inside_the_window_too():
    # The window hears the full recognizer, which writes down whatever it
    # heard: "babel" is what a real dictation of "hey Bol" came back as.
    d = _window_daemon(2, ["add a login test", "babel and run pytest"])
    await d._listen_session(d._begin(TYPE), until_silence=True, trigger=TYPE)
    await _window_done(d)
    assert d.bridge.injected == [
        ("add a login test ", False),
        ("and run pytest ", False),
    ]


@pytest.mark.asyncio
async def test_a_configured_short_word_chains_inside_the_window_as_well():
    d = _window_daemon(
        2, ["add a login test", "type and run pytest"], type_phrases=["type"]
    )
    await d._listen_session(d._begin(TYPE), until_silence=True, trigger=TYPE)
    await _window_done(d)
    assert d.bridge.injected == [
        ("add a login test ", False),
        ("and run pytest ", False),
    ]


@pytest.mark.asyncio
async def test_the_window_times_out_silently():
    # One utterance only: the command listen gets None back, as the recorder
    # does when nobody speaks before the window closes.
    d = _window_daemon(1, ["add a login test"])
    await d._listen_session(d._begin(TYPE), until_silence=True, trigger=TYPE)
    await _window_done(d)
    assert d.bridge.keys == []
    assert d._command_task is None
    assert d._pending_paste is True


@pytest.mark.asyncio
async def test_command_window_zero_means_keyword_ear_only():
    d = _window_daemon(2, ["add a login test", "send it"], window_s=0)
    await d._listen_session(d._begin(TYPE), until_silence=True, trigger=TYPE)
    assert d._command_task is None
    assert d.recorder.calls == [True]


@pytest.mark.asyncio
async def test_warming_the_speech_model_draws_nothing():
    d = _daemon(0, [], talk_back=False)
    d.summarizer = None
    await d._warm_speech_model()
    assert d.hud.calls == []
