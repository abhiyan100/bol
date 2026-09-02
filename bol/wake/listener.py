"""The wake phrase listener, in a process of its own.

Run as `python -m bol.wake`. It reads raw little-endian float32 frames of
FRAME_SAMPLES samples from stdin, exactly the blocks the daemon's audio
callback already has, and writes one line per detection to stdout:

    wake <threshold> <phrase>

sherpa-onnx does not report a posterior for a keyword, so the number on that
line is the trigger threshold the keyword fired at: a floor on the confidence,
not a measurement of it. The phrase is the spelling that matched, for the log.

Why a second process. The keyword model is onnxruntime, and the daemon is
already holding Parakeet and a microphone: a native library that segfaults, or
merely takes a thread pool it never gives back, must not be able to take the
hotkey down with it. The daemon reads this process's stdout and stops caring
if it dies.

Why stdin rather than a second microphone. macOS shows one recording
indicator either way, but two streams on one device is two chances to lose it,
and the daemon's stream is already running. One owner, one device session.

The keyword file format is sherpa-onnx's: one keyword per line as BPE pieces
separated by spaces, then ":<boosting score>" and "#<trigger threshold>".
The pieces have to be the ones the model would emit, so they come from the
BPE model shipped with the weights, not from a guess about spelling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .model import model_dir, model_files, model_present

SAMPLE_RATE = 16000
# Silero's window and the daemon's block size, so a frame is one callback.
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 4
FRAME_DTYPE = "<f4"

# How much the keyword's own tokens are boosted inside the beam search.
# Measured against `say` voices: at 4.0 the beam starts preferring the keyword
# so strongly that the threshold has to rise to compensate, which costs
# recall on the voices that were already marginal. 2.0 is the plateau.
KEYWORDS_SCORE = 2.0

# A gap this long in the frames means the daemon stopped forwarding (Bol was
# speaking, or a recording was in flight). The decoder is recurrent, so the
# audio on either side of that gap is not one utterance and must not be
# decoded as one.
RESET_GAP_S = 0.5

READY = "ready"


class WakeError(RuntimeError):
    """Something the user has to fix: a missing extra, a missing model, a
    phrase this model has no tokens for."""


# ------------------------------------------------------------------ keywords


def tokenize(phrase: str, bpe_model: Path, tokens: Path) -> str:
    """One phrase to the BPE pieces the model can emit, space separated.

    The pieces have to come from the model's own BPE, not from a longest-match
    guess over the token table: the keyword graph is matched against exactly
    what the decoder emits, and a plausible-looking segmentation the model
    never produces would simply never fire.
    """
    try:
        import sentencepiece as spm
    except ImportError as exc:  # the wake extra is not installed
        raise WakeError(f"sentencepiece is needed to encode a wake phrase ({exc})")
    encoder = spm.SentencePieceProcessor()
    encoder.load(str(bpe_model))
    pieces = encoder.encode(phrase.strip().upper(), out_type=str)
    table = {line.split()[0] for line in _lines(tokens) if line.split()}
    unknown = [piece for piece in pieces if piece not in table]
    if not pieces or unknown:
        raise WakeError(
            f"the keyword model has no tokens for {phrase!r} "
            f"({', '.join(unknown) or 'nothing encoded'}). "
            "Pick a wake phrase out of common English words."
        )
    return " ".join(pieces)


def _lines(path: Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def keywords_text(
    phrases,
    threshold: float,
    bpe_model: Path,
    tokens: Path,
    score: float = KEYWORDS_SCORE,
) -> str:
    """The keyword file sherpa-onnx reads, built from the spellings.

    Every line carries its own score and threshold rather than relying on the
    spotter-wide defaults, so one phrase can later be loosened without moving
    the others.
    """
    seen = []
    for phrase in phrases:
        encoded = tokenize(phrase, bpe_model, tokens)
        if encoded not in seen:
            seen.append(encoded)
    return "".join(f"{line} :{score:g} #{threshold:g}\n" for line in seen)


# ------------------------------------------------------------------- spotter


def build_spotter(files: dict, keywords_file: Path, threshold: float, threads: int = 1):
    """The sherpa-onnx KeywordSpotter, on one thread.

    One thread on purpose: this runs forever next to a speech model and a
    mic, and the whole budget for it is a few percent of one core.
    """
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise WakeError(f"sherpa-onnx is not installed ({exc})")
    return sherpa_onnx.KeywordSpotter(
        tokens=str(files["tokens"]),
        encoder=str(files["encoder"]),
        decoder=str(files["decoder"]),
        joiner=str(files["joiner"]),
        keywords_file=str(keywords_file),
        keywords_score=KEYWORDS_SCORE,
        keywords_threshold=threshold,
        num_threads=threads,
        provider="cpu",
    )


# ---------------------------------------------------------------------- loop


def read_frame(source) -> bytes | None:
    """Exactly one frame, or None at end of stream.

    A pipe read returns whatever has arrived, so a frame can and does come in
    two pieces; a short read that is not the end of the stream would otherwise
    shift every later frame by a few samples.
    """
    chunks = []
    want = FRAME_BYTES
    while want > 0:
        chunk = source.read(want)
        if not chunk:
            return None
        chunks.append(chunk)
        want -= len(chunk)
    return b"".join(chunks) if len(chunks) > 1 else chunks[0]


def run(source, sink, spotter, threshold: float, clock=time.monotonic) -> int:
    """Decode frames until stdin closes. Returns a process exit code."""
    stream = spotter.create_stream()
    sink.write(READY + "\n")
    sink.flush()
    last = clock()
    while True:
        frame = read_frame(source)
        if frame is None:
            return 0
        now = clock()
        if now - last > RESET_GAP_S:
            # The daemon muted us. Whatever was half decoded belongs to the
            # audio before the gap, and finishing it with the audio after
            # would be how Bol wakes itself on its own voice.
            spotter.reset_stream(stream)
        last = now
        samples = np.frombuffer(frame, dtype=FRAME_DTYPE).astype(np.float32)
        stream.accept_waveform(SAMPLE_RATE, samples)
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
        hit = spotter.get_result(stream)
        if not hit:
            continue
        spotter.reset_stream(stream)
        sink.write(f"wake {threshold:g} {hit.strip().lower()}\n")
        sink.flush()


# ---------------------------------------------------------------------- main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bol.wake")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--phrase", action="append", default=[])
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.model_dir) if args.model_dir else model_dir()
    phrases = args.phrase or ["hey bol"]
    try:
        if not model_present(root):
            raise WakeError(
                f"the keyword model is not in {root}. Run `bol setup` with "
                "[wake] enabled = true."
            )
        files = model_files(root)
        keywords = root / "keywords.txt"
        keywords.write_text(
            keywords_text(phrases, args.threshold, files["bpe"], files["tokens"]),
            encoding="utf-8",
        )
        spotter = build_spotter(files, keywords, args.threshold, args.threads)
    except WakeError as exc:
        print(f"bol.wake: {exc}", file=sys.stderr)
        return 2
    return run(sys.stdin.buffer, sys.stdout, spotter, args.threshold)
