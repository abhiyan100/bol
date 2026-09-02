"""One thread for every in-process MLX model.

MLX (0.32 and later) gives each OS thread its own default streams, and a
lazily built array remembers the stream it was created on. A model loaded on
one thread and evaluated from another fails with "There is no Stream(cpu, N)
in current thread". asyncio's default executor hands work to whichever pool
thread happens to be idle, which is fine while calls are sequential (the same
thread keeps getting reused) and breaks the moment two models warm up at the
same time. So Parakeet, the cleanup model and Kokoro all run their MLX work
through this single-worker executor: same thread every time, in every order.

The local summarizer is unaffected: it lives in its own mlx_lm.server process.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bol-mlx")


async def run(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run fn(*args, **kwargs) on the MLX thread and await its result.

    Calls queue behind each other. That is the point: the models this serves
    (speech in, cleanup, speech out) never need to overlap in the loop, and a
    queued call is always cheaper than a crashed one.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, functools.partial(fn, *args, **kwargs))
