"""Speech gates: one question, asked of one block of audio.

    gate.probability(block) -> 0.0 (room tone) .. 1.0 (someone is talking)

Two implementations behind that one method:

- SileroGate wraps Silero v6 through pysilero-vad. It is a 2.4 MB wheel with
  no dependencies of its own, and it answers in about 1 ms per 32 ms block,
  so the endpointing decision costs a few percent of one core.
- EnergyGate is the RMS-versus-adaptive-noise-floor rule Bol shipped before,
  kept as `[audio] vad = "energy"` and as the automatic fallback when Silero
  cannot be imported or loaded.

Both report a probability rather than a boolean so the recorder owns the
hysteresis and the two gates are interchangeable at the same thresholds:
speech starts above START_PROB for START_BLOCKS in a row, and only ends
after a run of blocks below END_PROB. The energy gate's probability is a
piecewise-linear map of "RMS over the noise floor" pinned at exactly those
two points, so switching gates changes the quality of the answer and not the
timings around it.

Silero decides on 512 samples at 16 kHz, which is why the recorder's block is
32 ms: the audio callback hands the gate exactly what it wants, with no
re-buffering between the microphone and the decision.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("bol.audio")

# Silero v5 and v6 both fix the window at 512 samples of 16 kHz mono, so this
# is the recorder's block size too.
SAMPLE_RATE = 16000
BLOCK_SAMPLES = 512
BLOCK_MS = 32

VAD_MODES = ("silero", "energy")

# Hysteresis, shared by both gates and applied by the recorder. Starting is
# deliberately harder than continuing: one loud block should not open an
# utterance, and a mid-word dip should not end one.
START_PROB = 0.5
END_PROB = 0.35
START_BLOCKS = 2

# Energy gate only. A block counts as speech above floor * energy_threshold,
# and keeps counting as speech down to this fraction of that.
_RELEASE_RATIO = 0.6
# The noise floor is the quiet fifth of what has been heard so far, not the
# first few blocks: a user who speaks immediately can't raise their own floor.
_FLOOR_PERCENTILE = 20
# What a gate says when it has no answer: too low to start an utterance, too
# high to end one. The recorder's timers still bound the recording.
_UNKNOWN_PROB = 0.5


def fit_block(block) -> np.ndarray:
    """Exactly BLOCK_SAMPLES of float32, by truncating or zero-padding.

    PortAudio hands back full blocks, but a stream that is stopped mid-block
    (or a test feeding its own arrays) can produce a short one, and Silero
    raises on anything that is not its window size.
    """
    data = np.asarray(block, dtype=np.float32).reshape(-1)
    if data.size == BLOCK_SAMPLES:
        return data
    if data.size > BLOCK_SAMPLES:
        return data[:BLOCK_SAMPLES]
    out = np.zeros(BLOCK_SAMPLES, dtype=np.float32)
    out[: data.size] = data
    return out


class EnergyGate:
    """RMS against an adaptive noise floor, reported as a probability.

    The floor is a percentile of every level seen since reset(), so it tracks
    the room rather than the opening blocks. The mapping is anchored so that
    "RMS exactly at the threshold" is exactly START_PROB and "RMS at the
    release ratio" is exactly END_PROB: the recorder's hysteresis then
    reproduces the rule this gate replaced, block for block.
    """

    def __init__(self, cfg) -> None:
        self._threshold = float(getattr(cfg, "energy_threshold", 3.0)) or 3.0
        self._levels: list[float] = []

    def reset(self) -> None:
        self._levels.clear()

    def probability(self, block) -> float:
        data = np.asarray(block, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(data**2)) + 1e-9)
        self._levels.append(rms)
        floor = max(float(np.percentile(self._levels, _FLOOR_PERCENTILE)), 1e-4)
        return _from_ratio(rms / (floor * self._threshold))


def _from_ratio(ratio: float) -> float:
    """Map RMS-over-threshold onto the shared probability scale."""
    if ratio <= 0.0:
        return 0.0
    if ratio < _RELEASE_RATIO:
        return END_PROB * (ratio / _RELEASE_RATIO)
    if ratio < 1.0:
        span = (ratio - _RELEASE_RATIO) / (1.0 - _RELEASE_RATIO)
        return END_PROB + span * (START_PROB - END_PROB)
    # Four times the threshold is as certain as this gate ever gets.
    return min(1.0, START_PROB + 0.5 * min(1.0, (ratio - 1.0) / 3.0))


class SileroGate:
    """Silero v6 through pysilero-vad, with a live fallback.

    The detector is stateful across blocks (it is a recurrent model), so
    reset() is called at the top of every recording; without it the tail of
    the last utterance colours the start of the next one.

    A detector that starts raising mid-session hands the rest of the run to
    the energy gate rather than taking the microphone down with it. That
    fallback's noise floor starts empty, so it needs a few blocks to settle,
    which is still better than a recording that never ends.
    """

    def __init__(self, detector, fallback) -> None:
        self._detector = detector
        self._fallback = fallback

    def reset(self) -> None:
        self._fallback.reset()
        if self._detector is None:
            return
        try:
            self._detector.reset()
        except Exception as exc:  # noqa: BLE001 - see _demote
            self._demote(exc)

    def probability(self, block) -> float:
        if self._detector is None:
            return self._fallback.probability(block)
        try:
            return float(self._detector.process_samples(fit_block(block)))
        except Exception as exc:  # noqa: BLE001 - a gate must not raise
            self._demote(exc)
            return _UNKNOWN_PROB

    def _demote(self, exc: Exception) -> None:
        self._detector = None
        log.warning(
            "Silero VAD stopped answering (%s), using the energy gate from here", exc
        )


def build_gate(cfg):
    """The gate [audio] vad asks for, or the energy gate when it cannot run.

    cfg is an AudioConfig. Falling back is a one-line info message and never
    an error: an endpointing rule that is merely cruder is not a reason to
    refuse to record.
    """
    energy = EnergyGate(cfg)
    if getattr(cfg, "vad", "silero") != "silero":
        return energy
    rate = int(getattr(cfg, "sample_rate", SAMPLE_RATE))
    if rate != SAMPLE_RATE:
        log.info(
            "Silero VAD needs %d Hz audio and this config records at %d Hz, "
            "so the energy gate is used instead",
            SAMPLE_RATE,
            rate,
        )
        return energy
    try:
        from pysilero_vad import SileroVoiceActivityDetector

        detector = SileroVoiceActivityDetector()
    except Exception as exc:  # noqa: BLE001 - import or model load
        log.info(
            "Silero VAD is unavailable (%s), using the energy gate instead", exc
        )
        return energy
    return SileroGate(detector, energy)
