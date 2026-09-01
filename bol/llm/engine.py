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
import sys

import aiohttp

log = logging.getLogger("bol.llm")

_START_TIMEOUT_S = 180  # first run downloads the model


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
                "mlx-lm not installed (uv sync --extra llm); "
                "summaries use the free template"
            )
            return
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "mlx_lm", "server",
            "--model", self._cfg.llm.local_model,
            "--host", "127.0.0.1",
            "--port", str(self._cfg.llm.local_port),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        deadline = asyncio.get_running_loop().time() + _START_TIMEOUT_S
        async with aiohttp.ClientSession() as session:
            while asyncio.get_running_loop().time() < deadline:
                if self._proc.returncode is not None:
                    raise RuntimeError("mlx_lm.server exited during startup")
                try:
                    async with session.get(
                        f"{self._base_url}/models",
                        timeout=aiohttp.ClientTimeout(total=2),
                    ) as resp:
                        if resp.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(1.0)
            else:
                raise RuntimeError("mlx_lm.server did not become healthy in time")
        # The HTTP server answers before the weights are loaded (and before a
        # first-run download finishes). Only a real completion proves ready.
        await asyncio.wait_for(
            self._request("You are a helpful assistant.", "hi", 1, 0.0),
            timeout=_START_TIMEOUT_S,
        )
        self._ready = True
        log.info(
            "llm: local %s ready on :%d",
            self._cfg.llm.local_model,
            self._cfg.llm.local_port,
        )

    async def stop(self) -> None:
        self._ready = False
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
        self._proc = None

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

        # The deadline covers waiting for the lock too — a queued cleanup
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
