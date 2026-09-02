"""The live decoder: the loop that puts words in the pill while you talk.

No model and no microphone. A fake StreamingParakeet records the chunks it is
handed and turns each one into a word, which is enough to pin down the parts
that are ours: how audio is batched, what happens when the decoder falls
behind, what the sentinel does, and the promise that none of this displaces
the one full-buffer decode that actually reaches Claude.
"""

import asyncio
import queue
import wave

import numpy as np
import pytest

from bol.config import Config
from bol.stt import STREAM_END
from bol.stt.parakeet import (
    DEFAULT_CONTEXT,
    ParakeetTranscriber,
    stream_context,
    stream_text,
)

BLOCK = 480  # 30 ms @ 16 kHz, the recorder's block size
CHUNK_MS = 300  # ten blocks exactly, so the batching assertions are exact
NEED = 16000 * CHUNK_MS // 1000


def _block(value=0.1):
    return np.full(BLOCK, value, dtype=np.float32)


class _Token:
    """An AlignedToken, as far as this code cares: text with its own spaces."""

    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text


class _Result:
    def __init__(self, text):
        self.text = text


class FakeLiveStream:
    """Stands in for StreamingParakeet. Each add commits the previous word and
    leaves the newest one in draft, which is what the right context does."""

    def __init__(self):
        self.chunks = []
        self.finalized_tokens = []
        self.draft_tokens = []
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def add_audio(self, audio):
        self.chunks.append(np.asarray(audio, dtype=np.float32))
        count = len(self.chunks)
        self.finalized_tokens = [_Token(f" word{i}") for i in range(1, count)]
        self.draft_tokens = [_Token(f" word{count}")]

    @property
    def result(self):
        tokens = self.finalized_tokens + self.draft_tokens
        return _Result("".join(token.text for token in tokens))


class FakeModel:
    def __init__(self, fail=False):
        self.streams = []
        self.stream_args = []
        self.transcribed = []
        self.fail = fail

    def transcribe_stream(self, context_size=(256, 256), depth=1, **kwargs):
        self.stream_args.append((tuple(context_size), depth))
        if self.fail:
            raise RuntimeError("this model does not stream")
        live = FakeLiveStream()
        self.streams.append(live)
        return live

    def transcribe(self, path):
        with wave.open(str(path), "rb") as handle:
            self.transcribed.append(handle.getnframes())
        return _Result("  the whole recording  ")


def _transcriber(fail=False):
    transcriber = ParakeetTranscriber(Config())
    transcriber._model = FakeModel(fail=fail)
    return transcriber


async def _stream(transcriber, blocks, emit, **over):
    options = {"context_size": (256, 16), "chunk_ms": CHUNK_MS, "sample_rate": 16000}
    options.update(over)
    # Timed out rather than awaited forever: the loop owns the single MLX
    # thread until the sentinel, so a hang here would take every later test.
    await asyncio.wait_for(transcriber.stream(blocks, emit, **options), timeout=5)
    await asyncio.sleep(0.01)  # let the queued emits land on the loop


class Emits:
    """Collects (committed, draft) and lets a test wait for the next one.

    Waiting on the emit rather than on a sleep is what makes the batching
    assertions mean anything: each round of blocks is the only audio the
    decoder can see, so the chunk it builds is the one the test asked for.
    """

    def __init__(self):
        self.seen = []
        self._arrived = asyncio.Event()

    def __call__(self, committed, draft):
        self.seen.append((committed, draft))
        self._arrived.set()

    async def next(self):
        await asyncio.wait_for(self._arrived.wait(), timeout=5)
        self._arrived.clear()


def _collector():
    emits = Emits()
    return emits.seen, emits


async def _feed(blocks, emits, rounds, per_round=10):
    """Hand the decoder one chunk at a time, waiting for each redraw."""
    for _ in range(rounds):
        for _ in range(per_round):
            blocks.put(_block())
        await emits.next()
    blocks.put(STREAM_END)


# ------------------------------------------------------------------ batching


async def test_audio_is_batched_at_the_configured_chunk():
    transcriber = _transcriber()
    blocks = queue.Queue()
    seen, emit = _collector()
    # Three chunks handed over one at a time, so nothing can be coalesced.
    asyncio.get_running_loop().create_task(_feed(blocks, emit, rounds=3))

    await _stream(transcriber, blocks, emit)

    live = transcriber._model.streams[0]
    assert [len(chunk) for chunk in live.chunks] == [NEED, NEED, NEED]
    assert len(seen) == 3


async def test_a_short_recording_never_reaches_a_full_chunk():
    # Under one chunk of audio there is nothing to show yet, and showing a
    # decode of half a syllable would be worse than showing "Listening".
    transcriber = _transcriber()
    blocks = queue.Queue()
    seen, emit = _collector()
    for _ in range(4):
        blocks.put(_block())
    blocks.put(STREAM_END)

    await _stream(transcriber, blocks, emit)

    assert transcriber._model.streams[0].chunks == []
    assert seen == []


async def test_a_backlog_is_coalesced_into_one_add():
    # The decoder fell behind (a slow step, a busy machine). It must catch up
    # in one add rather than walking a queue of stale chunks one at a time.
    transcriber = _transcriber()
    blocks = queue.Queue()
    seen, emit = _collector()
    for _ in range(40):  # 1.2 s of audio, four chunks' worth, all waiting
        blocks.put(_block())

    async def end():
        await asyncio.sleep(0.05)
        blocks.put(STREAM_END)

    asyncio.get_running_loop().create_task(end())
    await _stream(transcriber, blocks, emit)

    live = transcriber._model.streams[0]
    assert [len(chunk) for chunk in live.chunks] == [40 * BLOCK]
    assert len(seen) == 1  # one add, one redraw, no flood


# ------------------------------------------------------------------- emitting


async def test_emit_gets_the_committed_text_and_the_draft_apart():
    transcriber = _transcriber()
    blocks = queue.Queue()
    seen, emit = _collector()
    asyncio.get_running_loop().create_task(_feed(blocks, emit, rounds=2))

    await _stream(transcriber, blocks, emit)

    # Committed grows and the draft is always the newest word: the two halves
    # the pill draws solid and dim.
    assert seen == [("", "word1"), ("word1", "word2")]


def test_stream_text_falls_back_to_the_whole_result():
    class NoTokens:
        finalized_tokens = ()
        draft_tokens = ()
        result = _Result("  everything it has  ")

    assert stream_text(NoTokens()) == ("everything it has", "")


# ---------------------------------------------------------------- the sentinel


async def test_the_sentinel_stops_the_loop_and_drops_stale_audio():
    transcriber = _transcriber()
    blocks = queue.Queue()
    seen, emit = _collector()

    async def feed():
        for _ in range(10):  # one full chunk, decoded and shown
            blocks.put(_block())
        await emit.next()
        for _ in range(5):  # and a half-finished one the sentinel cuts off
            blocks.put(_block())
        blocks.put(STREAM_END)
        for _ in range(20):  # anything queued behind the sentinel is stale
            blocks.put(_block())

    asyncio.get_running_loop().create_task(feed())
    await _stream(transcriber, blocks, emit)

    live = transcriber._model.streams[0]
    assert [len(chunk) for chunk in live.chunks] == [NEED]
    assert live.exited  # and the decoder put the model's attention back


async def test_a_sentinel_inside_the_backlog_still_stops_the_loop():
    transcriber = _transcriber()
    blocks = queue.Queue()
    seen, emit = _collector()
    for _ in range(20):
        blocks.put(_block())
    blocks.put(STREAM_END)

    await _stream(transcriber, blocks, emit)

    assert transcriber._model.streams[0].chunks == []
    assert seen == []


# ------------------------------------------------------------ the final decode


async def test_the_full_buffer_decode_still_runs_exactly_once():
    # The whole point of the phase: partials are a mirror, and the text that
    # leaves this class is the one decode of the finished recording.
    transcriber = _transcriber()
    blocks = queue.Queue()
    seen, emit = _collector()
    for _ in range(20):
        blocks.put(_block())
    blocks.put(STREAM_END)
    await _stream(transcriber, blocks, emit)

    audio = np.concatenate([_block() for _ in range(50)])
    text = await transcriber.transcribe(audio, 16000)

    assert text == "the whole recording"
    assert transcriber._model.transcribed == [50 * BLOCK]  # the whole buffer


async def test_a_broken_stream_leaves_the_final_decode_alone():
    # The lock and the MLX thread have to come back even when streaming dies,
    # or one bad recording would take the transcriber down with it.
    transcriber = _transcriber(fail=True)
    seen, emit = _collector()

    with pytest.raises(RuntimeError):
        await _stream(transcriber, queue.Queue(), emit)

    audio = np.concatenate([_block() for _ in range(10)])
    assert await transcriber.transcribe(audio, 16000) == "the whole recording"


# --------------------------------------------------------------- configuration


async def test_the_configured_context_reaches_the_decoder():
    transcriber = _transcriber()
    blocks = queue.Queue()
    blocks.put(STREAM_END)
    seen, emit = _collector()

    await _stream(transcriber, blocks, emit, context_size=[256, 8])

    assert transcriber._model.stream_args == [((256, 8), 1)]


@pytest.mark.parametrize(
    "value", [None, "256,16", (), (256,), (256, 16, 8), ("a", "b"), (0, 16), (256, -1)]
)
def test_a_junk_context_falls_back_instead_of_raising(value):
    # Display only, so a typo in the config costs the live words at most.
    assert stream_context(value) == DEFAULT_CONTEXT


def test_a_good_context_is_taken_as_written():
    assert stream_context([128, 8]) == (128, 8)
    assert stream_context((256, 16)) == (256, 16)
