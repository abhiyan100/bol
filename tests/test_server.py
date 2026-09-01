"""Hook server: the endpoint that can type into your terminal.

Two things have to hold. The token check must actually reject, in constant
time, and the socket must never leave loopback by accident.
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from bol.hooks.server import HookServer, is_loopback


async def _client(server: HookServer) -> TestClient:
    client = TestClient(TestServer(server._build_app()))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_wrong_token_is_rejected():
    server = HookServer("127.0.0.1", 0, token="correct-horse")
    seen = []

    async def handler(payload):
        seen.append(payload)

    server.on("Stop", handler)
    client = await _client(server)
    try:
        resp = await client.post(
            "/hook", params={"token": "wrong"}, json={"hook_event_name": "Stop"}
        )
        assert resp.status == 401
        resp = await client.post("/hook", json={"hook_event_name": "Stop"})
        assert resp.status == 401  # missing token too
        assert seen == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_right_token_reaches_the_handler():
    server = HookServer("127.0.0.1", 0, token="correct-horse")
    seen = []

    async def handler(payload):
        seen.append(payload)

    server.on("Stop", handler)
    client = await _client(server)
    try:
        resp = await client.post(
            "/hook",
            params={"token": "correct-horse"},
            json={"hook_event_name": "Stop", "session_id": "a"},
        )
        assert resp.status == 200
        # The handler runs as a background task; let it land.
        for _ in range(20):
            if seen:
                break
            await _tick()
        assert seen and seen[0]["session_id"] == "a"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_health_needs_no_token():
    server = HookServer("127.0.0.1", 0, token="correct-horse")
    client = await _client(server)
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        assert (await resp.json())["ok"] is True
    finally:
        await client.close()


def test_non_loopback_host_is_refused():
    with pytest.raises(ValueError) as exc:
        HookServer("0.0.0.0", 8770)
    assert "allow_remote" in str(exc.value)
    with pytest.raises(ValueError):
        HookServer("192.168.1.20", 8770)


def test_non_loopback_host_allowed_when_opted_in(capsys):
    HookServer("0.0.0.0", 8770, allow_remote=True)
    assert "WARNING" in capsys.readouterr().out


def test_loopback_names():
    assert is_loopback("127.0.0.1")
    assert is_loopback("127.0.0.53")
    assert is_loopback("::1")
    assert is_loopback("localhost")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("")
    assert not is_loopback("bol.local")


async def _tick():
    import asyncio

    await asyncio.sleep(0)
