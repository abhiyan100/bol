"""Loopback HTTP server that receives Claude Code hook events.

Claude Code posts every configured hook event as JSON to /hook; the event name
rides in the payload (hook_event_name), so one endpoint serves them all.
Responses are always `{}`: Bol observes, it never blocks Claude.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import secrets
from typing import Awaitable, Callable

from aiohttp import web

log = logging.getLogger("bol.hooks")

Handler = Callable[[dict], Awaitable[None]]

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


def is_loopback(host: str) -> bool:
    """True only for addresses nothing off this machine can reach."""
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HookServer:
    def __init__(
        self, host: str, port: int, token: str = "", allow_remote: bool = False
    ) -> None:
        # A hook post makes Bol speak and can make it press Enter in your
        # terminal, so binding anywhere but loopback hands that to the LAN.
        if not is_loopback(host):
            if not allow_remote:
                raise ValueError(
                    f"[server] host {host!r} is not a loopback address, so anything "
                    "on your network could drive your terminal. Use 127.0.0.1, or "
                    "set [server] allow_remote = true if you really mean it."
                )
            print(
                f"bol: WARNING, the hook server is listening on {host}. Anything "
                "that can reach it can type into your terminal."
            )
            log.warning("hook server bound to non-loopback host %s", host)
        self._host = host
        self._port = port
        self._token = token
        self._handlers: dict[str, Handler] = {}
        self._runner: web.AppRunner | None = None

    def on(self, event: str, handler: Handler) -> None:
        self._handlers[event] = handler

    def _authorized(self, request: web.Request) -> bool:
        if not self._token:
            return True
        supplied = request.query.get("token", "")
        # Constant time: a plain != leaks the token a byte at a time.
        return secrets.compare_digest(
            supplied.encode("utf-8", "replace"), self._token.encode("utf-8")
        )

    async def _handle(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({}, status=401)
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

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/hook", self._handle)
        # /health stays unauthenticated: it carries nothing, and the doctor
        # uses it to tell "a Bol is already running" from "the port is busy".
        app.router.add_get("/health", self._health)
        return app

    async def start(self) -> None:
        self._runner = web.AppRunner(self._build_app(), access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        log.debug("hook server on http://%s:%d/hook", self._host, self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
