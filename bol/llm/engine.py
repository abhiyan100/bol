"""One LLM engine for both cleanup and summaries.

provider=local  spawns a supervised mlx_lm.server subprocess (OpenAI-
                compatible, crash-isolated from the mic/hook event loop) and
                talks to it over loopback.
provider=api    talks to any OpenAI-compatible endpoint with the user's key:
                OpenRouter, Groq, Ollama, LM Studio, OpenAI, Anthropic compat.
provider=off    engine reports unavailable; callers fall back (template
                summary, raw transcript).

All generation goes through one serializing lock: cleanup calls are short and
deadline-bounded; summary calls are longer and simply queue behind. A stable
system-prompt prefix plus prewarm-on-hotkey gives the KV-cache warm start
FluidVoice uses.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import aiohttp

from ..config import CONFIG_DIR

log = logging.getLogger("bol.llm")

_START_TIMEOUT_S = 180  # first run downloads the model

# Where mlx_lm.server's stderr goes. Download progress and load errors live
# here instead of /dev/null, so a stalled first run is diagnosable.
LLM_LOG_PATH = CONFIG_DIR / "llm.log"


# --------------------------------------------------- weights on disk (shared)
# The CLI (`bol setup`, `bol doctor`) reuses these; they are the only place
# that knows how big Bol's models are and whether they are already cached.

# Approximate download sizes in MB, measured from a warm Hugging Face cache.
MODEL_SIZES_MB = {
    "mlx-community/parakeet-tdt-0.6b-v3": 2300,
    "mlx-community/LFM2.5-1.2B-Instruct-4bit": 630,
    "LiquidAI/LFM2.5-350M-MLX-4bit": 200,
    "abhiyan10/bol-cleanup-350m-4bit": 195,
    "mlx-community/Kokoro-82M-bf16": 340,
}


def human_mb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def human_bytes(size: int) -> str:
    return human_mb(size / (1024 * 1024)) if size else "0 MB"


def size_hint(repo_id: str) -> str:
    """'about 630 MB' for models we ship with; 'size unknown' otherwise."""
    mb = MODEL_SIZES_MB.get(repo_id)
    return f"about {human_mb(mb)}" if mb else "size unknown"


def hf_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return True


def _repo_dir(repo_id: str) -> Path | None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    return Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))


def weights_cached(repo_id: str) -> bool:
    """Is this model already on disk? Never touches the network."""
    if not repo_id:
        return False
    local = Path(repo_id).expanduser()
    if local.is_dir():  # a local path, not a Hub id
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        if isinstance(try_to_load_from_cache(repo_id, "config.json"), str):
            return True
    except Exception:
        pass
    repo = _repo_dir(repo_id)
    if repo is None:
        return False
    snapshots = repo / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def weights_size_bytes(repo_id: str) -> int:
    """Bytes this model occupies in the Hugging Face cache (0 if absent)."""
    repo = _repo_dir(repo_id)
    if repo is None or not repo.is_dir():
        return 0
    blobs = repo / "blobs"
    root = blobs if blobs.is_dir() else repo
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


class LLMEngine:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._ready = False

    # ------------------------------------------------------------- lifecycle

    @property
    def available(self) -> bool:
        return self._ready

    @property
    def _base_url(self) -> str:
        if self._cfg.llm.provider == "local":
            return f"http://127.0.0.1:{self._cfg.llm.local_port}/v1"
        return self._cfg.llm.base_url.rstrip("/")

    @property
    def _model(self) -> str:
        if self._cfg.llm.provider == "local":
            return self._cfg.llm.local_model
        return self._cfg.llm.api_model

    async def start(self) -> None:
        provider = self._cfg.llm.provider
        if provider == "off":
            return
        if provider == "api":
            if not (self._cfg.llm.base_url and self._model and self._cfg.api_key):
                log.warning(
                    "llm provider=api but base_url/api_model/%s missing; LLM off",
                    self._cfg.llm.api_key_env,
                )
                return
            self._ready = True
            log.info("llm: api endpoint %s (%s)", self._base_url, self._model)
            return
        try:
            await self._start_local()
        except Exception as exc:
            log.warning("local llm failed to start (%s); falling back", exc)

    async def _start_local(self) -> None:
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            log.warning(
                "mlx-lm not installed (reinstall with the llm extra); "
                "summaries use the free template"
            )
            return
        port = self._cfg.llm.local_port
        model = self._cfg.llm.local_model
        # A server survives a SIGKILLed Bol (or is shared with another Bol).
        # Spawning a second one just loses the port bind, so reuse this one.
        if await self._probe():
            self._ready = True
            log.info("llm: reusing running server on :%d", port)
            return
        if not weights_cached(model):
            name = model.split("/")[-1]
            print(
                f"bol: downloading {name} ({size_hint(model)}) in the background; "
                "summaries use the template until it lands."
            )
            log.info("llm: download progress goes to %s", LLM_LOG_PATH)
        stderr = self._stderr_sink()
        try:
            self._proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mlx_lm", "server",
                "--model", model,
                "--host", "127.0.0.1",
                "--port", str(port),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr,
                # Own process group, so stop() can take down the server and
                # any worker it forked instead of orphaning them.
                start_new_session=True,
            )
        finally:
            if hasattr(stderr, "close"):  # the child holds its own dup now
                stderr.close()
        try:
            await self._await_healthy()
        except BaseException:
            # Never leave a half-started child holding the port.
            await self.stop()
            raise
        self._ready = True
        log.info("llm: local %s ready on :%d", model, port)

    @staticmethod
    def _stderr_sink():
        """A log file for the child's stderr, or DEVNULL if we cannot open one."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            return open(LLM_LOG_PATH, "a")
        except OSError as exc:
            log.warning("cannot write %s (%s); llm stderr discarded", LLM_LOG_PATH, exc)
            return asyncio.subprocess.DEVNULL

    async def _probe(self) -> bool:
        """Is a healthy OpenAI-compatible server already on our port?"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    async def _await_healthy(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _START_TIMEOUT_S
        while loop.time() < deadline:
            if self._proc is not None and self._proc.returncode is not None:
                raise RuntimeError(
                    f"mlx_lm.server exited during startup (see {LLM_LOG_PATH})"
                )
            if await self._probe():
                break
            await asyncio.sleep(1.0)
        else:
            raise RuntimeError("mlx_lm.server did not become healthy in time")
        # The HTTP server answers before the weights are loaded (and before a
        # first-run download finishes). Only a real completion proves ready.
        await asyncio.wait_for(
            self._request("You are a helpful assistant.", "hi", 1, 0.0),
            timeout=_START_TIMEOUT_S,
        )

    async def stop(self) -> None:
        self._ready = False
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        self._signal_group(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            self._signal_group(proc, signal.SIGKILL)
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                log.warning("mlx_lm.server did not exit; pid %s may linger", proc.pid)

    @staticmethod
    def _signal_group(proc, sig) -> None:
        """Signal the child's whole process group (start_new_session gave it
        one), falling back to the child alone if the group is already gone."""
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return
        except (OSError, TypeError, AttributeError):
            pass
        try:
            if sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except OSError:
            pass

    # ------------------------------------------------------------ generation

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int,
        deadline_s: float,
        temperature: float | None = None,
        history: list[dict] | None = None,
    ) -> str:
        """One chat completion under a hard deadline. Raises on any failure;
        callers own their fallback. `history` inserts few-shot turns between
        the system prompt and the user message."""
        if not self._ready:
            raise RuntimeError("llm not available")

        async def _locked() -> str:
            async with self._lock:
                return await self._request(system, user, max_tokens, temperature, history)

        # The deadline covers waiting for the lock too: a queued cleanup
        # call must not blow its budget behind a slow summary.
        return await asyncio.wait_for(_locked(), timeout=deadline_s)

    async def _request(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float | None,
        history: list[dict] | None = None,
    ) -> str:
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": (
                temperature if temperature is not None else self._cfg.llm.temperature
            ),
            "messages": [
                {"role": "system", "content": system},
                *(history or []),
                {"role": "user", "content": user},
            ],
        }
        headers = {}
        if self._cfg.llm.provider == "api":
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            ) as resp:
                data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {str(data)[:200]}")
        text = data["choices"][0]["message"]["content"].strip()
        if not text:
            raise RuntimeError("empty completion")
        return text

    def prewarm(self, system: str) -> None:
        """Fire-and-forget 1-token request so the prompt prefix is in the KV
        cache before the user finishes speaking."""
        if not self._ready or self._lock.locked():
            return

        async def _warm():
            try:
                await self.complete(system, "hi", max_tokens=1, deadline_s=10)
            except Exception:
                pass

        asyncio.get_running_loop().create_task(_warm())
