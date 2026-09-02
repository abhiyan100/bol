"""Parakeet TDT via parakeet-mlx, the low-latency local engine FluidVoice
ships. ~0.05 RTF on Apple Silicon; model downloads once from Hugging Face.

Two ways in, and only one of them decides what Claude gets:

- transcribe() decodes the finished recording in one pass. This is the text
  that reaches the bridge, always.
- stream() decodes the recording as it arrives, for the words in the pill. It
  runs as ONE mlx_thread.run call for the whole recording (MLX streams belong
  to the thread that made them), pulls blocks off a thread-safe queue, and
  reports (committed, draft) after each step. Nothing it produces is ever
  injected: it is a mirror held up to the microphone.

Bol runs the right context wide open, at parakeet-mlx's own default of 256
frames, and that is the whole trick. It reads backwards, so it is written
down here.

context_size[1] * depth is the drop size: the number of encoder frames a step
holds back from finalizing. Set it small (Bol used 16, or 1.3 s) and every
step commits the frames it just added -- which are the frames at the LEFT EDGE
of the window parakeet-mlx re-encodes, and that edge is where its seams land.
get_logmel re-estimates the per_feature mean and std from the new chunk alone,
and Conformer.__call__ passes pre_encode no cache, so the subsampling
convolution sees zero padding on the left. Measured against the same absolute
frames inside a longer window, encoder frame 0 comes out 59% off while frames
1 onward are bit-identical. So a small right context commits one garbage frame
per step, forever, and the damage accumulates in finalized_tokens: "parser"
came out "Marcer", "the hotkey listener" came out "the hikey l".

Set the drop size larger than the recording instead and the step finalizes
nothing at all. The mel buffer keeps everything, draft_tokens becomes a fresh
decode of the whole utterance with full context, and no seam ever reaches the
text. Real time, through stream(), on four recordings of 6 to 21 s, the live
view went from 0.55-0.72 similarity against the full-buffer decode to
0.87-1.00, and the first words landed at 0.84 s instead of 1.63 s.

It pays for that by re-encoding the buffer every step, but the cost is capped
rather than quadratic: past 256 frames (20.5 s) the drop size is reached, the
window starts sliding again, and long recordings degrade to the old behaviour
instead of falling behind. On 20.9 s of speech, at the shipped chunk: 0.74 s
between redraws, 0.10 s behind the end of the audio, 0.92 GB peak.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from .. import mlx_thread
from ..config import Config

log = logging.getLogger("bol.stt")

# How long a streaming step waits for the next block before looking at the
# clock again. Short enough that the sentinel is never sat on.
_POLL_S = 0.05
# Backstop for one streaming loop. The recording is capped far below this
# ([audio] max_utterance_s); this only exists so a sentinel lost to a crashed
# caller cannot pin the single MLX thread forever.
_MAX_STREAM_S = 300.0
# Right context wide enough that a normal recording finalizes nothing, so the
# live view is a re-decode rather than a pile of seam-damaged commits. See the
# module docstring; this is also parakeet-mlx's own default.
DEFAULT_CONTEXT = (256, 256)
# One step per 640 ms of audio. Each step now re-encodes the whole buffer, so
# this is the redraw rate AND the CPU bill; 320 ms scored identically and cost
# twice as many encodes, 640 ms still redraws every 0.74 s on a 21 s recording.
DEFAULT_CHUNK_MS = 640
# Below this the re-encodes stop keeping up with the microphone on a long
# recording, and a config typo must not be able to reach that.
MIN_CHUNK_MS = 320
# How long a word stays dim in the pill. With the wide right context
# parakeet-mlx finalizes nothing, so Bol draws the solid/dim line itself, at
# the same 1.3 s lag the old context enforced -- but on tokens that came out
# of a full-context decode instead of off a window seam.
DRAFT_LAG_S = 1.3


def stream_context(value: object) -> tuple[int, int]:
    """Read [stt] stream_context, falling back rather than raising.

    A malformed value in a config file must cost the live words, not the
    recording: everything here is display only.
    """
    try:
        left, right = (int(x) for x in value)  # type: ignore[misc]
    except Exception:  # noqa: BLE001 - any shape but a pair of numbers
        return DEFAULT_CONTEXT
    if left < 1 or right < 0:
        return DEFAULT_CONTEXT
    return left, right


def tokens_text(tokens) -> str:
    """The text of a run of tokens. Token text carries its own leading spaces
    (sentencepiece "_" is decoded to " "), so a plain join is the sentence."""
    return "".join(getattr(token, "text", "") for token in tokens or ())


def settled(tokens, elapsed_s: float, lag_s: float = DRAFT_LAG_S) -> int:
    """How many leading tokens ended long enough ago to draw solid.

    The longest PREFIX that is old enough, not every old token: the pill draws
    one dim run at the end, so a settled word after an unsettled one has to
    stay dim too. A token with no usable end time settles nothing.
    """
    cut = elapsed_s - lag_s
    if cut <= 0:
        return 0
    for index, token in enumerate(tokens):
        try:
            end = float(getattr(token, "end", None))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return index
        if end > cut:
            return index
    return len(tokens)


def stream_text(live, elapsed_s: float | None = None) -> tuple[str, str]:
    """(committed, draft) for a StreamingParakeet, as the pill wants it.

    Committed is drawn solid, draft dim. Normally parakeet-mlx decides the
    split for us, in finalized_tokens. Bol's right context is wide enough that
    a recording under 20.5 s finalizes nothing at all, though (that is the
    point: see the module docstring), so given how much audio has gone in we
    draw the line ourselves, on the tokens whose audio is DRAFT_LAG_S behind.

    Falling back to .result.text keeps this working with a decoder that
    reports no token lists at all.
    """
    finalized = getattr(live, "finalized_tokens", ()) or ()
    tokens = list(getattr(live, "draft_tokens", ()) or ())
    committed = tokens_text(finalized)
    # Only while nothing is finalized: once the window slides, token times are
    # relative to the window rather than to the recording, and the comparison
    # against elapsed_s would quietly mean nothing.
    cut = settled(tokens, elapsed_s) if elapsed_s is not None and not finalized else 0
    draft = tokens_text(tokens[cut:])
    committed += tokens_text(tokens[:cut])
    if committed or draft:
        return committed.strip(), draft.strip()
    result = getattr(live, "result", None)
    return str(getattr(result, "text", "") or "").strip(), ""


class ParakeetTranscriber:
    def __init__(self, cfg: Config) -> None:
        import parakeet_mlx  # noqa: F401  (fail fast if missing)

        self._model_name = cfg.stt.parakeet_model
        self._model = None
        self._lock = asyncio.Lock()

    def _ensure_model(self):
        if self._model is None:
            from parakeet_mlx import from_pretrained

            log.info("loading Parakeet model %s ...", self._model_name)
            self._model = from_pretrained(self._model_name)
        return self._model

    async def warmup(self) -> None:
        async with self._lock:
            await mlx_thread.run(self._warmup)

    def _warmup(self) -> None:
        model = self._ensure_model()
        self._warm_stream(model)

    def _warm_stream(self, model) -> None:
        """Run one streaming step on silence, so the first real one is cheap.

        Entering a stream swaps the encoder to local attention and the first
        add_audio compiles the whole streaming path: about 1.1 s that would
        otherwise land on the first thing you say, when the pill is empty and
        the wait is most visible. Silence, one chunk long, on the MLX thread
        that will do the real work.

        Guarded whole. Warmup is an optimisation; a model that cannot stream,
        or a parakeet-mlx that renamed something, must still transcribe.
        """
        try:
            import mlx.core as mx

            samples = 16000 * DEFAULT_CHUNK_MS // 1000
            with model.transcribe_stream(
                context_size=DEFAULT_CONTEXT, depth=1
            ) as live:
                live.add_audio(mx.zeros((samples,), dtype=mx.float32))
        except Exception:  # noqa: BLE001 - warmup never costs a recording
            log.debug("streaming warmup skipped", exc_info=True)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        async with self._lock:
            return await mlx_thread.run(self._transcribe, audio, sample_rate)

    # --------------------------------------------------------------- live view

    async def stream(
        self,
        blocks: "queue.Queue",
        emit,
        *,
        context_size=DEFAULT_CONTEXT,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        sample_rate: int = 16000,
    ) -> None:
        """Decode a recording as it is captured, for display only.

        blocks is a thread-safe queue of float32 mono blocks; put None on it
        to end the loop. emit(committed, draft) is called on this event loop
        after every step. Returns when the sentinel arrives, and holds the
        transcriber's lock until then, so the final transcribe() of the same
        recording queues behind it rather than racing it for the MLX thread.
        """
        loop = asyncio.get_running_loop()
        async with self._lock:
            await mlx_thread.run(
                self._stream_loop,
                blocks,
                emit,
                loop,
                stream_context(context_size),
                max(MIN_CHUNK_MS, int(chunk_ms)),
                int(sample_rate),
            )

    def _stream_loop(
        self,
        blocks: "queue.Queue",
        emit,
        loop: asyncio.AbstractEventLoop,
        context_size: tuple[int, int],
        chunk_ms: int,
        sample_rate: int,
    ) -> None:
        # Runs on the MLX thread, start to finish: the streaming state and
        # every array in it belong to whichever thread created them.
        import mlx.core as mx

        model = self._ensure_model()
        need = max(1, int(sample_rate * chunk_ms / 1000))
        deadline = time.monotonic() + _MAX_STREAM_S
        pending: list[np.ndarray] = []
        held = 0
        fed = 0  # samples handed to the decoder, for the solid/dim line
        with model.transcribe_stream(context_size=context_size, depth=1) as live:
            while time.monotonic() < deadline:
                try:
                    block = blocks.get(timeout=_POLL_S)
                except queue.Empty:
                    continue
                if block is None:
                    return  # the recording ended; whatever is left is stale
                pending.append(block)
                held += len(block)
                # Coalesce: if the decoder fell behind, take everything that
                # piled up in one step instead of walking it chunk by chunk.
                stop = False
                while True:
                    try:
                        extra = blocks.get_nowait()
                    except queue.Empty:
                        break
                    if extra is None:
                        stop = True
                        break
                    pending.append(extra)
                    held += len(extra)
                if stop:
                    return
                if held < need:
                    continue
                chunk = pending[0] if len(pending) == 1 else np.concatenate(pending)
                pending, held = [], 0
                fed += len(chunk)
                live.add_audio(mx.array(chunk))
                committed, draft = stream_text(live, fed / sample_rate)
                loop.call_soon_threadsafe(emit, committed, draft)

    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # parakeet-mlx takes a file path; hand it a temp 16-bit PCM WAV.
        model = self._ensure_model()
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16.tobytes())
            result = model.transcribe(path)
        finally:
            path.unlink(missing_ok=True)
        return result.text.strip()
