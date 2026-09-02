"""Microphone capture with two stop conditions:

- push-to-talk: caller stops the recording explicitly (hotkey released);
- until-silence: a speech gate ends the utterance after trailing silence.

Stop signalling uses per-recording session tokens, not shared state: begin()
mints a token synchronously at the triggering event, and request_stop() on a
token only ever ends *that* recording. A release for a session that never got
the mic, or that already finished, is inert by construction: no cross-talk,
no lost-stop races. The token also carries the mutable until_silence flag, so
a tap can switch a running push-to-talk recording over to the speech gate,
and end_reason, which records how the recording actually ended. That is the
difference between "I am done" and "I paused", and the daemon spends it on
whether to press Enter.

The stream is prepared once and kept. Building an sd.InputStream costs about
33 ms on an M-series Mac and start() another 10 to 25 ms, and both used to sit
between the hotkey and the first syllable. Now the object is built once, the
callback feeds a 2 s ring buffer whenever the stream runs, and a recording
prepends pre_roll_ms of that ring so the opening word survives the press. The
stream is released warm_s after the last recording rather than held forever:
a Bluetooth headset pinned open stays in its tinny headset profile.

The speech gate ends a recording only while session.until_silence is set.
Holding the key is already an explicit request to record, so push-to-talk
hands back whatever it captured and lets the transcriber decide whether there
were words in it. Gating there dropped the utterance of anyone who starts
talking the instant the key goes down, because their own voice defined the
noise floor. Every block is still put to the gate regardless, so the gate is
warm the moment the flag flips mid-recording.

Which gate runs is [audio] vad: Silero v6 by default, the old RMS rule as
"energy". Both answer with a probability, and the hysteresis lives here:
speech starts above START_PROB for START_BLOCKS in a row and ends after
silence_ms below END_PROB. The block size is Silero's window, 512 samples at
16 kHz, so nothing between the microphone and the decision re-buffers.

A recording can also carry a tap: set session.tap before record() and every
block of that recording, pre-roll included, is handed to it as it arrives. The
tap is a copy for someone watching (the live transcriber that puts words in the
pill); the recording buffer this function returns stays authoritative, and a
tap that raises, fills up, or goes away never touches the recording.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from time import perf_counter

import numpy as np
import sounddevice as sd

from ..config import AudioConfig
from .vad import BLOCK_MS, END_PROB, START_BLOCKS, START_PROB, build_gate

log = logging.getLogger("bol.audio")

# How much recent audio the always-on ring keeps. Caps pre_roll_ms and keeps
# the buffer's memory flat no matter how long the stream stays warm.
_RING_MS = 2000

# Every way a recording can end, as RecordingSession.end_reason.
#   release: the held key came up.
#   tap:     a second tap ended a recording the first tap started.
#   silence: the speech gate endpointed after trailing silence.
#   window:  the no-speech window passed with nobody speaking.
#   max:     max_utterance_s (or its wall-clock backstop) ran out.
#   cancelled: the user did something else instead (a click, another app).
#   stop:    someone else stopped it: barge-in, shutdown, an unnamed caller.
# The first two are the user saying "done"; the rest are Bol deciding.
END_REASONS = ("release", "tap", "silence", "window", "max", "cancelled", "stop")

# What a recording ended by a click or an app switch is called. Its audio is
# dropped rather than transcribed: the user is looking at something else.
CANCELLED = "cancelled"


def _resolve_input_device(spec: str) -> int | None:
    """Map [audio] input_device to a sounddevice id. Accepts an index or a
    case-insensitive substring of the device name; empty means system default."""
    spec = spec.strip()
    if not spec:
        return None
    if spec.lstrip("-").isdigit():
        return int(spec)
    wanted = spec.lower()
    names: list[str] = []
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) < 1:
            continue
        name = device.get("name", "")
        if wanted in name.lower():
            return index
        names.append(name)
    available = ", ".join(names) if names else "none found"
    raise ValueError(
        f"no input device matches {spec!r}. Available: {available}. "
        "Set [audio] input_device to one of those names (or its index)."
    )


def _feed_tap(tap, block: np.ndarray) -> None:
    """Hand one block to a recording's tap, on PortAudio's thread.

    Swallows everything. The tap is a spectator: a full queue or a listener
    that has already gone away must never cost the recording a block, and an
    exception raised here would take the whole audio callback down with it.
    """
    try:
        put = getattr(tap, "put_nowait", None)
        if put is not None:
            put(block)
        else:
            tap(block)
    except Exception:  # noqa: BLE001 - see the docstring
        pass


class RecordingSession:
    """Stop token for one recording. Mint via Recorder.begin() at the event
    that starts the recording (hotkey press / auto-listen decision).

    until_silence is mutable and read every loop iteration: a tap sets it
    after the recording is already running, which turns "record while held"
    into "record until they stop talking" without restarting the mic.

    tap is optional and set before record(): a queue.Queue (thread-safe, since
    the audio callback fills it from PortAudio's thread) or any callable that
    takes one block. It sees every block this recording captures.

    end_reason is how the recording finished, and the daemon reads it to
    decide whether the words were finished too. Letting the key go or tapping
    a second time is someone saying "done"; the silence gate ending an
    utterance only means they paused, and people pause mid-sentence.

    silence_ms overrides [audio] silence_ms for this recording alone, and is
    how a "type" dictation gets its own, much longer pause: dictating a prompt
    has thinking pauses in it that a conversational turn does not, and one
    number cannot be both. window_ms is the same idea for the other end, the
    wait for anyone to start speaking at all: a recording a trigger word
    opened by mistake should give the microphone back in seconds, where one
    the user asked for can afford to wait.
    """

    __slots__ = (
        "_stop", "until_silence", "tap", "end_reason", "silence_ms", "window_ms",
    )

    def __init__(self, until_silence: bool = False) -> None:
        self._stop = asyncio.Event()
        self.until_silence = until_silence
        self.tap = None
        # One of END_REASONS once this recording is over; "" until then.
        self.end_reason = ""
        # None means "whatever [audio] silence_ms says".
        self.silence_ms = None
        # None means "whatever [audio] listen_window_s says".
        self.window_ms = None

    def note_end(self, reason: str) -> None:
        """Name what ended this recording; the first answer wins.

        A hold released in the same breath as the gate endpointing must not
        relabel a pause as a deliberate finish, or the other way round:
        whatever actually ended the recording got there first.
        """
        if not self.end_reason:
            self.end_reason = reason

    def request_stop(self, reason: str = "stop") -> None:
        self.note_end(reason)
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()


class Recorder:
    def __init__(self, cfg: AudioConfig, gate=None) -> None:
        self._cfg = cfg
        self._device: int | None = None
        self._device_resolved = False
        self._stream = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # Set for the duration of one recording; the callback reads it once.
        self._sink: asyncio.Queue[np.ndarray] | None = None
        # The live listener for the recording in flight, if it asked for one.
        self._tap = None
        # Wake mode's listener: unlike a tap it is not tied to a recording,
        # it sees every block for as long as the stream runs.
        self._monitor = None
        self._hold_open = False
        self._ring: deque[np.ndarray] = deque(maxlen=max(1, _RING_MS // BLOCK_MS))
        self._warm_task: asyncio.Task | None = None
        self._logged_latency = False
        # Built on first use from [audio] vad (loading Silero costs a few ms),
        # then reused and reset per recording. Injectable so endpointing can
        # be tested against scripted probabilities instead of scripted audio.
        self._gate = gate

    def begin(self, until_silence: bool = False) -> RecordingSession:
        return RecordingSession(until_silence)

    def _input_device(self) -> int | None:
        # Resolved once: the config is static, and a name lookup on every
        # hotkey press would add latency at the worst possible moment.
        if not self._device_resolved:
            self._device = _resolve_input_device(self._cfg.input_device)
            self._device_resolved = True
        return self._device

    @property
    def device_label(self) -> str:
        """What to call the current input device in a message to the user."""
        return self._cfg.input_device.strip() or "the default input"

    def use_default_device(self) -> None:
        """Forget [audio] input_device and take whatever macOS calls default.

        The retry after a mic that vanished: the configured name may now match
        nothing (the headset is off) while the built-in mic is right there.
        """
        self._device = None
        self._device_resolved = True
        self._discard()

    def speech_gate(self):
        """The gate for the until-silence path, built once per Recorder."""
        if self._gate is None:
            self._gate = build_gate(self._cfg)
        return self._gate

    # ------------------------------------------------------------------ stream

    def _callback(self, indata, _frames, _time, status) -> None:
        # PortAudio's thread, not the event loop. deque.append on a bounded
        # deque is atomic and never grows, so the ring needs no lock.
        if status:
            log.debug("audio status: %s", status)
        mono = indata[:, 0]
        if np.issubdtype(mono.dtype, np.integer):
            # The stream asks for float32, but a backend that hands back
            # int16 would overflow the RMS and break the transcriber.
            mono = mono.astype(np.float32) / float(np.iinfo(mono.dtype).max)
        block = np.asarray(mono, dtype=np.float32).copy()
        self._ring.append(block)
        # Wake mode's ear, fed before anything else can fail: it listens
        # between recordings, which is exactly when the sink and the tap are
        # both empty.
        monitor = self._monitor
        if monitor is not None:
            _feed_tap(monitor, block)
        sink, loop = self._sink, self._loop
        if sink is not None and loop is not None:
            loop.call_soon_threadsafe(sink.put_nowait, block)
        # Fed straight from this thread: the tap is a thread-safe queue on
        # purpose, so the words on screen don't wait for the event loop.
        tap = self._tap
        if tap is not None:
            _feed_tap(tap, block)

    def _build(self) -> None:
        cfg = self._cfg
        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype="float32",
            blocksize=int(cfg.sample_rate * BLOCK_MS / 1000),
            device=self._input_device(),
            callback=self._callback,
        )
        self._running = False

    async def open(self) -> None:
        """Build the stream before the hotkey is armed.

        Construction is the expensive half and it does not touch the device's
        record light, so it is safe to pay it at startup and not at the press.
        """
        self._loop = asyncio.get_running_loop()
        if self._stream is None:
            self._build()
        # Same argument for the gate: loading Silero at the first press would
        # put its model load between the key and the first syllable.
        self.speech_gate()

    async def monitor(self, listener) -> None:
        """Forward every captured block to `listener` and keep the mic open.

        Wake mode's one-microphone rule. The daemon already owns a stream and
        a 2 s ring; the keyword listener reads the blocks that stream is
        producing anyway rather than opening a second device, so there is one
        device session, one recording indicator, and nothing to contend with
        when a recording starts.

        The warm window is suspended for as long as a listener is set: warm_s
        exists to give a Bluetooth headset its profile back between
        recordings, and wake mode is a deliberate decision to hold the mic.
        Pass None to stop forwarding and hand the device back to that timer.
        """
        self._monitor = listener
        self._hold_open = listener is not None
        if listener is None:
            self._schedule_warm_release()
            return
        self._loop = asyncio.get_running_loop()
        self._cancel_warm()
        if self._stream is None:
            self._build()
        self._start_stream()

    def _start_stream(self) -> None:
        if self._running:
            return
        self._ring.clear()  # whatever is in there predates this device session
        started = perf_counter()
        self._stream.start()
        self._running = True
        if not self._logged_latency:
            self._logged_latency = True
            log.debug(
                "input stream start latency: %.1f ms",
                (perf_counter() - started) * 1000,
            )

    def _stop_stream(self) -> None:
        stream, self._running = self._stream, False
        self._ring.clear()
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as exc:  # a yanked device: rebuild on the next press
            log.debug("could not stop the input stream: %s", exc)
            self._discard()

    def _discard(self) -> None:
        """Drop the stream so the next record() builds a fresh one."""
        stream, self._stream = self._stream, None
        self._running = False
        self._ring.clear()
        if stream is None:
            return
        try:
            stream.close()
        except Exception as exc:
            log.debug("could not close the input stream: %s", exc)

    def _cancel_warm(self) -> None:
        task, self._warm_task = self._warm_task, None
        if task is not None:
            task.cancel()

    def _schedule_warm_release(self) -> None:
        if self._stream is None:
            return  # already discarded; nothing to hold open
        if self._hold_open:
            return  # wake mode is listening; the warm window does not apply
        warm = float(self._cfg.warm_s)
        if warm <= 0 or self._loop is None:
            self._stop_stream()
            return
        self._warm_task = self._loop.create_task(self._release_after(warm))

    async def _release_after(self, warm: float) -> None:
        try:
            await asyncio.sleep(warm)
        except asyncio.CancelledError:
            return
        log.debug("warm window elapsed, releasing the microphone")
        self._stop_stream()

    async def close(self) -> None:
        """Release the device and forget the stream (daemon shutdown)."""
        self._cancel_warm()
        self._sink = None
        self._tap = None
        self._monitor = None
        self._hold_open = False
        self._discard()

    def _pre_roll(self) -> list[np.ndarray]:
        count = int(self._cfg.pre_roll_ms) // BLOCK_MS
        if count <= 0 or not self._ring:
            return []
        return list(self._ring)[-count:]

    # ----------------------------------------------------------------- capture

    async def record(
        self, session: RecordingSession, until_silence: bool
    ) -> np.ndarray | None:
        """Record one utterance; returns float32 mono @ sample_rate, or None.

        until_silence only ever turns the gate ON: the caller's argument and a
        later tap both set the session flag, and the loop re-reads it every
        block. With the gate off the recording returns everything captured;
        only an empty recording (a tap released before the mic opened) gives
        None. With it on, the recording gives the mic up if no speech starts
        inside listen_window_s, endpoints after silence_ms, and discards
        anything under min_speech_ms.
        """
        if session.stopped:
            return None  # quick tap: released before the mic even opened
        if until_silence:
            session.until_silence = True
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._cancel_warm()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        tap = session.tap

        try:
            if self._stream is None:
                self._build()
            was_running = self._running
            # Snapshot before the sink is armed, so a block cannot land in
            # both. Losing the seam block costs 30 ms; doubling it stutters.
            pre_roll = self._pre_roll() if was_running else []
            if tap is not None:
                # The tap gets the pre-roll before it is armed, so the live
                # decoder hears the opening syllable in the right order.
                for block in pre_roll:
                    _feed_tap(tap, block)
            self._sink = queue
            self._tap = tap
            if not was_running:
                self._start_stream()
        except Exception:
            self._sink = None
            self._tap = None
            self._discard()  # dead device: the next press builds a new stream
            raise

        try:
            return await self._pump(session, queue, pre_roll)
        except Exception:
            self._discard()
            raise
        finally:
            self._sink = None
            self._tap = None
            self._schedule_warm_release()

    async def _pump(
        self,
        session: RecordingSession,
        queue: asyncio.Queue[np.ndarray],
        pre_roll: list[np.ndarray],
    ) -> np.ndarray | None:
        cfg = self._cfg
        loop = asyncio.get_running_loop()
        gate = self.speech_gate()
        gate.reset()
        blocks: list[np.ndarray] = list(pre_roll)
        # Pre-roll primes the gate but never votes: it is room tone from
        # before the press, and a door slam in it must not count as the start
        # of speech. The answers are thrown away on purpose.
        for block in pre_roll:
            gate.probability(block)
        speech_blocks = 0
        silence_blocks = 0
        onset_blocks = 0
        # Per recording, because a dictation and a conversational turn do not
        # pause for the same reasons or for the same length of time.
        silence_ms = session.silence_ms or cfg.silence_ms
        silence_limit = max(1, int(silence_ms) // BLOCK_MS)
        max_blocks = cfg.max_utterance_s * 1000 // BLOCK_MS
        window_ms = session.window_ms or cfg.listen_window_s * 1000
        window_blocks = max(1, int(window_ms) // BLOCK_MS)
        heard_speech = False
        # Wall clock as well as block count: a warm stream that goes silent
        # because the device vanished delivers nothing, and a block-only cap
        # would then spin here forever.
        deadline = loop.time() + cfg.max_utterance_s + 1

        while len(blocks) < max_blocks and loop.time() < deadline:
            if session.stopped:
                break
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            blocks.append(chunk)
            # Asked for every block, gate in use or not, so a tap that flips
            # until_silence mid-recording finds a gate already warmed on this
            # room and this voice.
            prob = gate.probability(chunk)
            if heard_speech:
                # Continuing is easier than starting: one quiet block inside a
                # word must not end the utterance.
                if prob >= END_PROB:
                    speech_blocks += 1
                    silence_blocks = 0
                else:
                    silence_blocks += 1
                    if session.until_silence and silence_blocks >= silence_limit:
                        session.note_end("silence")
                        break
            elif prob > START_PROB:
                speech_blocks += 1
                onset_blocks += 1
                heard_speech = onset_blocks >= START_BLOCKS
            else:
                onset_blocks = 0  # a lone loud block is a noise, not a word
                if session.until_silence and len(blocks) >= window_blocks:
                    session.note_end("window")
                    break  # nobody spoke, give the mic up
        else:
            # The while condition gave out rather than a break: the block cap
            # or its wall-clock backstop. Either way the recording ran out of
            # room, which is not the user saying they were finished.
            session.note_end("max")

        if not blocks:
            return None
        if session.until_silence and speech_blocks * BLOCK_MS < cfg.min_speech_ms:
            return None
        return np.concatenate(blocks)
