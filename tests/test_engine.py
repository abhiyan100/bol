"""Engine lifecycle: reuse a running server, never orphan a child."""

import asyncio
import sys
import types

import pytest

from bol.config import Config
from bol.llm import engine as engine_mod
from bol.llm.engine import LLMEngine


class FakeProc:
    """Just enough of asyncio.subprocess.Process for the startup paths."""

    def __init__(self) -> None:
        self.pid = 424242
        self.returncode = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode


@pytest.fixture
def local_cfg(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setattr(engine_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(engine_mod, "LLM_LOG_PATH", tmp_path / "llm.log")
    monkeypatch.setattr(engine_mod, "weights_cached", lambda repo: True)
    # A stray process group must never be signalled from a test.
    def _no_group(_pid):
        raise ProcessLookupError

    monkeypatch.setattr("os.getpgid", _no_group)
    cfg = Config()
    cfg.llm.provider = "local"
    return cfg


async def test_reuses_a_running_server(local_cfg, monkeypatch):
    async def healthy(_self):
        return True

    def no_spawn(*args, **kwargs):
        raise AssertionError("spawned a second server instead of reusing one")

    monkeypatch.setattr(LLMEngine, "_probe", healthy)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_spawn)

    engine = LLMEngine(local_cfg)
    await engine._start_local()

    assert engine.available
    assert engine._proc is None  # nothing of ours to shut down


async def test_startup_failure_terminates_the_child(local_cfg, monkeypatch):
    proc = FakeProc()

    async def unhealthy(_self):
        return False

    async def spawn(*args, **kwargs):
        assert kwargs["start_new_session"] is True  # own process group
        return proc

    async def never_ready(_self):
        raise RuntimeError("mlx_lm.server did not become healthy in time")

    monkeypatch.setattr(LLMEngine, "_probe", unhealthy)
    monkeypatch.setattr(LLMEngine, "_await_healthy", never_ready)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    engine = LLMEngine(local_cfg)
    with pytest.raises(RuntimeError):
        await engine._start_local()

    assert proc.terminated, "a half-started server was left holding the port"
    assert engine._proc is None
    assert not engine.available


async def test_download_notice_when_weights_are_missing(
    local_cfg, monkeypatch, capsys
):
    monkeypatch.setattr(engine_mod, "weights_cached", lambda repo: False)
    proc = FakeProc()

    async def unhealthy(_self):
        return False

    async def ready(_self):
        return None

    async def spawn(*args, **kwargs):
        return proc

    monkeypatch.setattr(LLMEngine, "_probe", unhealthy)
    monkeypatch.setattr(LLMEngine, "_await_healthy", ready)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    engine = LLMEngine(local_cfg)
    await engine._start_local()

    out = capsys.readouterr().out
    assert "downloading" in out and "in the background" in out
    assert engine.available
    await engine.stop()


def test_size_hints_are_human():
    assert engine_mod.size_hint("mlx-community/LFM2.5-1.2B-Instruct-4bit") == (
        "about 630 MB"
    )
    assert engine_mod.size_hint("mlx-community/parakeet-tdt-0.6b-v3") == "about 2.2 GB"
    assert engine_mod.size_hint("someone/unknown-model") == "size unknown"
    assert engine_mod.weights_cached("") is False
