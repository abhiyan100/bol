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

The streaming decoder commits a token only once context_size[1] * depth
encoder frames of 80 ms have gone past it. parakeet-mlx defaults that right
context to 256, which is 20 seconds of held-back text: fine for captioning a
file, useless for a pill. Bol runs 16 (1.3 s).
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
DEFAULT_CONTEXT = (256, 16)
DEFAULT_CHUNK_MS = 320
# Below this a step costs more in decode overhead than it buys in freshness.
MIN_CHUNK_MS = 40


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


def stream_text(live) -> tuple[str, str]:
    """(committed, draft) for a StreamingParakeet, as the pill wants it.

    Committed is what the decoder will not take back; draft is the tail it is
    still deciding on. Falling back to .result.text keeps this working with a
    decoder that reports no token lists at all.
    """
    committed = tokens_text(getattr(live, "finalized_tokens", ()))
    draft = tokens_text(getattr(live, "draft_tokens", ()))
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
            await mlx_thread.run(self._ensure_model)

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
                live.add_audio(mx.array(chunk))
                committed, draft = stream_text(live)
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
