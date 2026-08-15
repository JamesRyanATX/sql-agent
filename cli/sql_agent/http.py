"""The CLI's side of the API. The only module here that is not a renderer.

Configuration comes from `sql_agent.config` and never from `app.settings`,
which is what keeps this a client of the server rather than a component of it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx

# A turn takes minutes — a T1 against a local model runs near 300s — and httpx's
# default 5s read timeout would kill every one. Connect still fails fast, so a
# server that isn't listening says so immediately.
STREAM_TIMEOUT = httpx.Timeout(None, connect=5.0)
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


class ApiError(Exception):
    """Anything the caller should see as a message rather than a traceback."""


def _config():
    # Imported here, not at module scope: config imports ApiError from this
    # module, and the CLI's one hard rule is that neither of them reaches for
    # `app`.
    from sql_agent import config

    return config


def _headers() -> dict[str, str]:
    token = _config().api_key()
    return {"authorization": f"Bearer {token}"} if token else {}


def _client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    # SQL_AGENT_URL carries the /v1 prefix and httpx preserves a base_url's
    # path when it merges a relative one, so every call site below writes the
    # path without it. Do not reach for urljoin — it eats the prefix.
    return httpx.AsyncClient(
        base_url=_config().base_url(), headers=_headers(), timeout=timeout
    )


def _raise_for_status(resp: httpx.Response, body: str | None = None) -> None:
    if resp.status_code < 400:
        return
    if resp.status_code == 401:
        raise ApiError(
            "401 unauthorized — SQL_AGENT_API_KEY does not match the API_TOKEN "
            "the server is running with"
        )
    detail = body if body is not None else resp.text
    try:
        detail = json.loads(detail)["detail"]
    except (ValueError, KeyError, TypeError):
        pass
    raise ApiError(f"{resp.status_code}: {str(detail)[:300]}")


def parse_sse(lines: list[str]) -> dict[str, Any] | None:
    """One SSE frame's `data:` payload, or None if the frame carried none.

    The payloads are the dicts the graph yielded; `app.events.sse` only
    serialised them.
    """
    data = "\n".join(
        line.removeprefix("data:").lstrip() for line in lines if line.startswith("data:")
    )
    if not data:
        return None
    return json.loads(data)


async def stream_events(path: str, payload: dict[str, Any]) -> AsyncIterator[dict]:
    """POST and yield each event as it arrives.

    Frames are separated by a blank line. `aiter_lines` already splits on line
    boundaries and never hands back a partial line, so accumulating until the
    blank is all the framing this needs.
    """
    seen = 0
    try:
        async with _client(STREAM_TIMEOUT) as c:
            async with c.stream("POST", path, json=payload) as resp:
                if resp.status_code >= 400:
                    _raise_for_status(resp, (await resp.aread()).decode())

                frame: list[str] = []
                async for line in resp.aiter_lines():
                    if line:
                        frame.append(line)
                        continue
                    ev = parse_sse(frame)
                    frame.clear()
                    if ev is not None:
                        seen += 1
                        yield ev
                # A stream cut mid-frame leaves no blank line behind it, and the
                # last event is usually the one saying what went wrong.
                ev = parse_sse(frame)
                if ev is not None:
                    seen += 1
                    yield ev
    except httpx.ConnectError as e:
        raise ApiError(_unreachable(e)) from e
    except httpx.TransportError as e:
        # A reload mid-turn, or a dropped connection. The count matters:
        # "nothing arrived" and "it died three minutes in" are different
        # problems. After ConnectError, which is itself a TransportError.
        raise ApiError(
            f"stream ended after {seen} events — the server closed the "
            f"connection ({type(e).__name__})"
        ) from e


async def request(method: str, path: str, **kw) -> Any:
    try:
        async with _client(REQUEST_TIMEOUT) as c:
            resp = await c.request(method, path, **kw)
    except httpx.ConnectError as e:
        raise ApiError(_unreachable(e)) from e
    except httpx.TransportError as e:
        # A server that accepts the connection and then dies is not a bug in
        # this client. After ConnectError, which is itself a TransportError.
        raise ApiError(
            f"no reply from {_config().base_url()} — the server accepted the "
            f"connection and then closed it ({type(e).__name__}). "
            "Check its logs: 'make logs-api'"
        ) from e
    _raise_for_status(resp)
    return resp.json()


async def get(path: str, params: dict[str, Any] | None = None) -> Any:
    return await request("GET", path, params=params)


async def post(path: str, json: dict[str, Any] | None = None, **kw) -> Any:
    return await request("POST", path, json=json, **kw)


async def patch(path: str, json: dict[str, Any] | None = None) -> Any:
    return await request("PATCH", path, json=json)


async def delete(path: str) -> Any:
    return await request("DELETE", path)


def _unreachable(e: Exception) -> str:
    # No "start it with `make up`" here: an installed CLI pointed at staging has
    # no Makefile, and a hint that does not apply is worse than none.
    return f"no API at {_config().base_url()} — is it running? ({type(e).__name__})"


def run(coro) -> None:
    """Entry point for every command: run it, and report ApiError as a message.

    A client that cannot reach the server should say so in one line; there is no
    bug to read in a traceback here.
    """
    import asyncio

    try:
        asyncio.run(coro)
    except ApiError as e:
        print(f"\033[31m{e}\033[0m", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
