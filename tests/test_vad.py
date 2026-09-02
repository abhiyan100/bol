"""Speech gate tests.

Two things have to hold. The gates have to agree on a scale, because the
recorder applies one set of thresholds to whichever one is running: the
energy gate's probability is a map of RMS over the noise floor pinned at
exactly START_PROB and END_PROB, and this file pins those anchors down. And
Silero has to be optional at runtime even though it ships with Bol, because
a wheel that fails to load must cost endpointing quality and nothing else.

The Silero half runs against a fake detector injected into sys.modules, plus
one smoke test on the real thing: a 2.4 MB model with no dependencies is
worth actually calling once.
"""

import logging
import sys
import types

import numpy as np
import pytest

from bol.audio import vad
from bol.audio.vad import (
    BLOCK_SAMPLES,
    END_PROB,
    START_PROB,
    EnergyGate,
    SileroGate,
    build_gate,
    fit_block,
)
from bol.config import AudioConfig


def _cfg(**over):
    cfg = AudioConfig()
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def _flat(amplitude, size=BLOCK_SAMPLES):
    """A block whose RMS is exactly `amplitude`, so ratios are arithmetic."""
    return np.full(size, amplitude, dtype=np.float32)


# ------------------------------------------------------------------- energy


def _floored(gate, amplitude=0.001, blocks=10):
    """Give the gate a noise floor to measure against."""
    for _ in range(blocks):
        gate.probability(_flat(amplitude))
    return gate


@pytest.mark.parametrize(
    "amplitude, expected",
    [
        (0.003, START_PROB),   # exactly at threshold: not speech yet
        (0.0018, END_PROB),    # exactly at the release ratio: not silence yet
        (0.0009, END_PROB / 2),
        (0.0, 0.0),
    ],
)
def test_the_energy_gate_is_pinned_to_the_shared_thresholds(amplitude, expected):
    # floor 0.001 x energy_threshold 3.0 = 0.003 is "speech starts here", and
    # the recorder's START_PROB is what that has to come out as.
    gate = _floored(EnergyGate(_cfg(energy_threshold=3.0)))
    assert gate.probability(_flat(amplitude)) == pytest.approx(
        expected, rel=0.01, abs=1e-6
    )


def test_a_shout_saturates_the_energy_gate():
    gate = _floored(EnergyGate(_cfg(energy_threshold=3.0)))
    assert gate.probability(_flat(0.3)) == pytest.approx(1.0)


def test_the_energy_gate_needs_a_quiet_moment_to_measure_against():
    # Worth pinning because it is the whole argument for Silero being the
    # default: this gate hears loudness, not voices, so a room that is never
    # quiet reads as never speaking. It only recognises the same voice once
    # it has some room tone to compare it against.
    gate = EnergyGate(_cfg(energy_threshold=3.0))
    assert all(gate.probability(_flat(0.2)) < START_PROB for _ in range(5))
    _floored(gate)
    assert gate.probability(_flat(0.2)) > START_PROB


def test_reset_forgets_the_room():
    gate = _floored(EnergyGate(_cfg()), amplitude=0.2)
    gate.reset()
    # With the loud room forgotten, a quiet block is measured against itself
    # and cannot be speech.
    assert gate.probability(_flat(0.001)) < START_PROB


# -------------------------------------------------------------------- silero


class FakeDetector:
    """pysilero-vad's SileroVoiceActivityDetector, scripted."""

    def __init__(self, script=(), fail_after=None):
        self.script = list(script)
        self.fail_after = fail_after
        self.seen = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def process_samples(self, samples):
        self.seen.append(len(samples))
        if self.fail_after is not None and len(self.seen) > self.fail_after:
            raise RuntimeError("the model went away")
        return self.script.pop(0) if self.script else 0.0


def test_the_silero_gate_reports_what_the_model_says():
    detector = FakeDetector([0.02, 0.91, 0.44])
    gate = SileroGate(detector, EnergyGate(_cfg()))
    answers = [gate.probability(_flat(0.1)) for _ in range(3)]
    assert answers == [0.02, 0.91, 0.44]
    assert detector.seen == [BLOCK_SAMPLES] * 3


def test_reset_resets_the_model_state():
    # Silero is recurrent: without this the tail of the last utterance
    # colours the start of the next one.
    detector = FakeDetector()
    gate = SileroGate(detector, EnergyGate(_cfg()))
    gate.reset()
    assert detector.resets == 1


def test_a_detector_that_starts_failing_hands_over_to_the_energy_gate(caplog):
    detector = FakeDetector([0.9], fail_after=1)
    gate = SileroGate(detector, EnergyGate(_cfg(energy_threshold=3.0)))
    with caplog.at_level(logging.WARNING, logger="bol.audio"):
        assert gate.probability(_flat(0.1)) == 0.9
        gate.probability(_flat(0.1))          # this one raises inside the model
        for _ in range(10):                   # room tone, for the new gate
            gate.probability(_flat(0.001))
        loud = gate.probability(_flat(0.3))
    assert detector.seen == [BLOCK_SAMPLES] * 2  # not called again
    assert loud > START_PROB                     # the energy gate answers now
    assert len([r for r in caplog.records if "Silero" in r.getMessage()]) == 1


def test_a_block_that_is_not_the_window_size_is_made_to_fit():
    short, long = fit_block(_flat(0.5, 100)), fit_block(_flat(0.5, 900))
    assert short.size == long.size == BLOCK_SAMPLES
    assert float(short[99]) == pytest.approx(0.5)
    assert float(short[100]) == 0.0


# --------------------------------------------------------------- build_gate


def _fake_pysilero(monkeypatch, detector=None, boom=None):
    module = types.ModuleType("pysilero_vad")

    def factory(*_a, **_k):
        if boom is not None:
            raise boom
        return detector

    module.SileroVoiceActivityDetector = factory
    monkeypatch.setitem(sys.modules, "pysilero_vad", module)
    return module


def test_the_default_gate_is_silero(monkeypatch):
    _fake_pysilero(monkeypatch, FakeDetector())
    assert isinstance(build_gate(_cfg()), SileroGate)


def test_energy_is_the_explicit_opt_out(monkeypatch):
    _fake_pysilero(monkeypatch, FakeDetector())
    assert isinstance(build_gate(_cfg(vad="energy")), EnergyGate)


def test_a_missing_wheel_falls_back_to_energy(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "pysilero_vad", None)  # ImportError
    with caplog.at_level(logging.INFO, logger="bol.audio"):
        gate = build_gate(_cfg())
    assert isinstance(gate, EnergyGate)
    assert len([r for r in caplog.records if "energy gate" in r.getMessage()]) == 1


def test_a_model_that_will_not_load_falls_back_to_energy(monkeypatch, caplog):
    _fake_pysilero(monkeypatch, boom=OSError("no such file: ggml-silero.bin"))
    with caplog.at_level(logging.INFO, logger="bol.audio"):
        gate = build_gate(_cfg())
    assert isinstance(gate, EnergyGate)
    assert "no such file" in caplog.text


def test_silero_is_declined_at_the_wrong_sample_rate(monkeypatch, caplog):
    # Silero's window is 512 samples of 16 kHz. At any other rate the answer
    # would be confidently wrong, which is worse than a cruder rule.
    _fake_pysilero(monkeypatch, FakeDetector())
    with caplog.at_level(logging.INFO, logger="bol.audio"):
        gate = build_gate(_cfg(sample_rate=48000))
    assert isinstance(gate, EnergyGate)
    assert "48000" in caplog.text


# --------------------------------------------------------------- the real one


def _resonator(x, centre, bandwidth):
    """One two-pole formant resonator, the cheapest way to make a vowel."""
    radius = np.exp(-np.pi * bandwidth / vad.SAMPLE_RATE)
    theta = 2 * np.pi * centre / vad.SAMPLE_RATE
    a1, a2 = -2 * radius * np.cos(theta), radius * radius
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = (
            x[i]
            - a1 * (out[i - 1] if i > 0 else 0.0)
            - a2 * (out[i - 2] if i > 1 else 0.0)
        )
    return out


def _voiced(blocks: int) -> np.ndarray:
    """Synthetic speech: a glottal pulse train with a wobbling pitch, three
    formants, and a syllable-rate envelope. Not words, but voiced the way a
    voice is, which is what Silero is looking for. A steady two-tone buzz is
    not: the model watches it for a moment and then decides it is machinery.
    """
    samples = np.arange(blocks * BLOCK_SAMPLES) / vad.SAMPLE_RATE
    f0 = 130 + 25 * np.sin(2 * np.pi * 2.0 * samples)
    phase = np.cumsum(f0 / vad.SAMPLE_RATE)
    pulses = (np.diff(np.floor(phase), prepend=0.0) > 0).astype(np.float64)
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * samples)) ** 2
    voice = pulses * envelope
    for centre, bandwidth in ((700, 90), (1200, 110), (2600, 160)):
        voice = _resonator(voice, centre, bandwidth)
    return (0.5 * voice / np.max(np.abs(voice))).astype(np.float32)


def test_the_shipped_model_hears_a_voice_and_not_a_hiss():
    # The one test that actually loads Silero. It is 2.4 MB with no
    # dependencies, and "the wheel installed but the native library does not
    # run here" is exactly the failure worth catching in CI.
    gate = build_gate(_cfg())
    if not isinstance(gate, SileroGate):
        pytest.skip("pysilero-vad is not installed in this environment")
    blocks = 14
    rng = np.random.default_rng(11)
    hiss = max(
        gate.probability(rng.normal(0, 0.3, BLOCK_SAMPLES).astype(np.float32))
        for _ in range(blocks)
    )
    gate.reset()
    voice = _voiced(blocks)
    speech = max(
        gate.probability(voice[i * BLOCK_SAMPLES : (i + 1) * BLOCK_SAMPLES])
        for i in range(blocks)
    )
    assert hiss < END_PROB     # loud is not the same as spoken
    assert speech > START_PROB
