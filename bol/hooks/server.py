"""Loopback HTTP server that receives Claude Code hook events.

Claude Code posts every configured hook event as JSON to /hook; the event name
rides in the payload (hook_event_name), so one endpoint serves them all.
Responses are always `{}` — Bol observes, it never blocks Claude.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from aiohttp import web

log = logging.getLogger("bol.hooks")

Handler = Callable[[dict], Awaitable[None]]


class HookServer:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._handlers: dict[str, Handler] = {}
        self._runner: web.AppRunner | None = None

    def on(self, event: str, handler: Handler) -> None:
        self._handlers[event] = handler

    async def _handle(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({}, status=400)
        event = payload.get("hook_event_name", "")
        handler = self._handlers.get(event)
        if handler is not None:
            # Ack Claude immediately; process in the background so a slow
            # summarizer/TTS never stalls the Claude Code UI.
            asyncio.get_running_loop().create_task(self._run(handler, payload, event))
        return web.json_response({})

    @staticmethod
    async def _run(handler: Handler, payload: dict, event: str) -> None:
        try:
            await handler(payload)
        except Exception:
            log.exception("handler for %s failed", event)

    async def _health(self, _request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "app": "bol"})

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/hook", self._handle)
        app.router.add_get("/health", self._health)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        log.info("hook server on http://%s:%d/hook", self._host, self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
