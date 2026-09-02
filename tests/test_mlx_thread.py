"""Every MLX call lands on the same OS thread, in any order, at any concurrency."""

import asyncio
import threading

import pytest

from bol import mlx_thread


def _thread_name() -> str:
    return threading.current_thread().name


@pytest.mark.asyncio
async def test_sequential_calls_share_one_thread():
    first = await mlx_thread.run(_thread_name)
    second = await mlx_thread.run(_thread_name)
    assert first == second
    assert first.startswith("bol-mlx")
    assert first != threading.current_thread().name


@pytest.mark.asyncio
async def test_concurrent_calls_still_share_one_thread():
    # Two models warming up at once is exactly the case that broke the
    # default executor: it grew a second thread and later calls could land on
    # either one.
    names = await asyncio.gather(*(mlx_thread.run(_thread_name) for _ in range(6)))
    assert len(set(names)) == 1


@pytest.mark.asyncio
async def test_arguments_and_results_pass_through():
    def add(a, b, *, scale=1):
        return (a + b) * scale

    assert await mlx_thread.run(add, 2, 3, scale=10) == 50


@pytest.mark.asyncio
async def test_exceptions_propagate():
    def boom():
        raise RuntimeError("There is no Stream(cpu, 3) in current thread.")

    with pytest.raises(RuntimeError, match="Stream"):
        await mlx_thread.run(boom)
