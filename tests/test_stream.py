"""The live decoder: the loop that puts words in the pill while you talk.

No model and no microphone. A fake StreamingParakeet records the chunks it is
handed and turns each one into a word, which is enough to pin down the parts
that are ours: how audio is batched, what happens when the decoder falls
behind, what the sentinel does, and the promise that none of this displaces
the one full-buffer decode that actually reaches Claude.
"""

import asyncio
import queue
import sys
import types
import wave

import numpy as np
import pytest

from bol.config import Config
from bol.stt import STREAM_END
from bol.stt.parakeet import (
    DEFAULT_CHUNK_MS,
    DEFAULT_CONTEXT,
    DRAFT_LAG_S,
    MIN_CHUNK_MS,
    ParakeetTranscriber,
    settled,
    stream_context,
    stream_text,
)

BLOCK = 480  # 30 ms @ 16 kHz, the recorder's block size
CHUNK_MS = 600  # twenty blocks exactly, so the batching assertions are exact
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


@pytest.fixture(autouse=True)
def stub_mlx(monkeypatch):
    """Run these tests without parakeet-mlx or mlx installed.

    CI installs the base dependencies only (the stt extra pulls ~2 GB of MLX
    wheels), so the modules this file exercises have to be reachable without
    them. Every import of either package lives inside a method -- the fail-fast
    one in ParakeetTranscriber.__init__, the model factory in _ensure_model,
    mx.array in the streaming loop -- so a stub in sys.modules is enough, and
    the real packages are still used when they are present locally.
    """
    parakeet = types.ModuleType("parakeet_mlx")
    parakeet.from_pretrained = lambda name: FakeModel()

    core = types.ModuleType("mlx.core")
    core.array = lambda value, *a, **kw: np.asarray(value, dtype=np.float32)
    core.zeros = lambda shape, dtype=None: np.zeros(shape, dtype=np.float32)
    core.float32 = np.float32
    mlx = types.ModuleType("mlx")
    mlx.core = core

    monkeypatch.setitem(sys.modules, "parakeet_mlx", parakeet)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return parakeet


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


async def _feed(blocks, emits, rounds, per_round=20):
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
    for _ in range(40):  # 1.2 s of audio, two chunks' worth, all waiting
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
        for _ in range(20):  # one full chunk, decoded and shown
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


# ----------------------------------------------------------------- the seam


async def _blocks_before_a_step(transcriber, count, **over):
    """Feed count blocks, let the decoder have them, and return what it added.

    The sentinel goes in after a beat rather than behind the blocks: queued
    behind them it would be swept up by the same coalescing pass and stop the
    loop before the chunk was ever handed over, which is a different test.
    """
    blocks = queue.Queue()
    for _ in range(count):
        blocks.put(_block())
    seen, emit = _collector()

    async def end():
        await asyncio.sleep(0.05)
        blocks.put(STREAM_END)

    asyncio.get_running_loop().create_task(end())
    options = {"context_size": (256, 16), "sample_rate": 16000}
    options.update(over)
    await asyncio.wait_for(transcriber.stream(blocks, emit, **options), timeout=5)
    return transcriber._model.streams[0].chunks


@pytest.mark.parametrize(
    "chunk_ms, want_ms",
    [(None, DEFAULT_CHUNK_MS), (40, MIN_CHUNK_MS), (1, MIN_CHUNK_MS)],
)
async def test_a_step_never_carries_less_than_the_floor(chunk_ms, want_ms):
    # The step length is not a free knob. parakeet-mlx commits the frames a
    # step added, and its first encoder frame comes off the window seam, so a
    # short step commits a larger share of garbage: at 320 ms one frame in
    # four. The default and the floor are what keep that share small, and a
    # config typo must not be able to talk either of them down.
    need = 16000 * want_ms // 1000
    over = {} if chunk_ms is None else {"chunk_ms": chunk_ms}

    short = (need - 1) // BLOCK  # one block shy of a step
    assert await _blocks_before_a_step(_transcriber(), short, **over) == []

    chunks = await _blocks_before_a_step(_transcriber(), short + 1, **over)
    assert [len(chunk) for chunk in chunks] == [(short + 1) * BLOCK]


# ------------------------------------------------------------------- warmup


async def test_warmup_runs_a_streaming_step_before_the_first_utterance():
    # Entering a stream swaps the encoder to local attention and compiles the
    # streaming path. Paid here on silence, or paid by the first thing you say.
    transcriber = _transcriber()

    await transcriber.warmup()

    live = transcriber._model.streams[0]
    assert transcriber._model.stream_args == [(DEFAULT_CONTEXT, 1)]
    assert [len(chunk) for chunk in live.chunks] == [
        16000 * DEFAULT_CHUNK_MS // 1000
    ]
    assert not live.chunks[0].any()  # silence, not somebody's audio
    assert live.exited  # and the encoder's attention is back where it was


async def test_warmup_survives_a_model_that_cannot_stream():
    # Warmup is an optimisation. A parakeet-mlx that renamed something, or a
    # model with no streaming decoder, must still leave a working transcriber.
    transcriber = _transcriber(fail=True)

    await transcriber.warmup()  # must not raise

    audio = np.concatenate([_block() for _ in range(10)])
    assert await transcriber.transcribe(audio, 16000) == "the whole recording"


async def test_warmup_loads_the_model_when_there_is_none_yet():
    transcriber = ParakeetTranscriber(Config())
    assert transcriber._model is None

    await transcriber.warmup()

    assert transcriber._model is not None
    assert transcriber._model.streams[0].chunks  # and it was warmed, not just loaded


# ------------------------------------------------- drawing the solid/dim line


class _Timed:
    """An AlignedToken with the end time the split reads."""

    __slots__ = ("text", "end")

    def __init__(self, text, end):
        self.text = text
        self.end = end


class _Live:
    def __init__(self, draft, finalized=()):
        self.draft_tokens = list(draft)
        self.finalized_tokens = list(finalized)


def _timed(*pairs):
    return [_Timed(text, end) for text, end in pairs]


def test_settled_takes_a_prefix_not_every_old_token():
    # The pill dims one run at the end, so a settled word sitting behind an
    # unsettled one has to stay dim: the answer is a prefix length.
    tokens = _timed((" a", 1.0), (" b", 9.0), (" c", 2.0))
    assert settled(tokens, elapsed_s=5.0, lag_s=0.0) == 1


def test_nothing_is_settled_before_the_lag_has_passed():
    tokens = _timed((" a", 0.1))
    assert settled(tokens, elapsed_s=DRAFT_LAG_S / 2) == 0


def test_a_token_with_no_end_time_settles_nothing_after_it():
    tokens = [_Timed(" a", 0.5), _Token(" b"), _Timed(" c", 0.6)]
    assert settled(tokens, elapsed_s=10.0) == 1


async def test_the_pill_gets_a_solid_prefix_when_nothing_is_finalized():
    # The shipped context finalizes nothing under 20.5 s, which is the point:
    # the decode is clean. But the pill still wants a solid head and a dim
    # tail, so the line is drawn here, on how much audio has gone in.
    live = _Live(_timed((" the", 1.0), (" whole", 2.0), (" suite", 4.6)))

    assert stream_text(live, 5.0) == ("the whole", "suite")


async def test_the_line_is_left_to_the_decoder_once_it_finalizes():
    # Past 20.5 s the window slides and token times stop being relative to the
    # recording, so comparing them against elapsed audio would be nonsense.
    live = _Live(_timed((" b", 0.1)), finalized=_timed((" a", 0.0)))

    assert stream_text(live, 100.0) == ("a", "b")


def test_stream_text_without_a_clock_leaves_the_split_alone():
    live = _Live(_timed((" one", 0.1), (" two", 0.2)))

    assert stream_text(live) == ("", "one two")


async def test_the_loop_tells_stream_text_how_much_audio_it_has_fed():
    class Clocked(FakeLiveStream):
        def __init__(self):
            super().__init__()
            self.elapsed = []

        def add_audio(self, audio):
            super().add_audio(audio)
            self.finalized_tokens = []  # the shipped context finalizes nothing
            self.draft_tokens = _timed((" word", 0.1))

    live = Clocked()
    transcriber = _transcriber()
    transcriber._model.transcribe_stream = lambda **kw: live
    transcriber._model.streams.append(live)
    blocks = queue.Queue()
    seen, emit = _collector()
    asyncio.get_running_loop().create_task(_feed(blocks, emit, rounds=3))

    await _stream(transcriber, blocks, emit)

    # Chunks of 600 ms. The token ends at 0.1 s, so it stays dim until the
    # recording is DRAFT_LAG_S past that: not at 0.6 s, not at 1.2 s, but at
    # 1.8 s it moves from the dim tail into the solid head.
    assert seen == [("", "word"), ("", "word"), ("word", "")]
