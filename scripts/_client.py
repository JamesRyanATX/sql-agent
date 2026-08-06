"""The scripts' side of the API. The only new logic in `scripts/`.

Everything under `scripts/` renders; this module is what it renders *from*. The
business logic lives behind the endpoints in `app/api.py`, so a script here is a
terminal UI and nothing else.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.settings import settings

# A turn takes minutes — a T1 against a local model has been measured near 300s.
# httpx's default 5s read timeout would kill every one of them. Connect still
# fails fast, because a server that isn't listening should say so immediately
# rather than hang for the length of a turn.
STREAM_TIMEOUT = httpx.Timeout(None, connect=5.0)
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


class ApiError(Exception):
    """Anything the caller should see as a message rather than a traceback."""


def _headers() -> dict[str, str]:
    token = settings().api_token
    return {"authorization": f"Bearer {token}"} if token else {}


def _client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings().api_url, headers=_headers(), timeout=timeout
    )


def _raise_for_status(resp: httpx.Response, body: str | None = None) -> None:
    if resp.status_code < 400:
        return
    if resp.status_code == 401:
        raise ApiError(
            "401 unauthorized — set API_TOKEN in .env to the same value the "
            "server is running with"
        )
    detail = body if body is not None else resp.text
    raise ApiError(f"{resp.status_code} from {resp.request.url.path}: {detail[:300]}")


def parse_sse(lines: list[str]) -> dict[str, Any] | None:
    """One SSE frame's `data:` payload, or None if the frame carried none.

    The payloads are the same dicts the graph yielded — `app.events.sse` only
    serialised them — so a caller gets back exactly what the nodes emitted.
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
                        yield ev
                # A stream cut mid-frame leaves no blank line behind it. The
                # last event is worth having: it is usually the one that says
                # what went wrong.
                ev = parse_sse(frame)
                if ev is not None:
                    yield ev
    except httpx.ConnectError as e:
        raise ApiError(_unreachable(e)) from e


async def get(path: str, params: dict[str, Any] | None = None) -> Any:
    try:
        async with _client(REQUEST_TIMEOUT) as c:
            resp = await c.get(path, params=params)
    except httpx.ConnectError as e:
        raise ApiError(_unreachable(e)) from e
    _raise_for_status(resp)
    return resp.json()


async def delete(path: str) -> Any:
    try:
        async with _client(REQUEST_TIMEOUT) as c:
            resp = await c.delete(path)
    except httpx.ConnectError as e:
        raise ApiError(_unreachable(e)) from e
    _raise_for_status(resp)
    return resp.json()


def _unreachable(e: Exception) -> str:
    return (
        f"no API at {settings().api_url} — start it with 'make up' "
        f"({type(e).__name__})"
    )


def run(coro) -> None:
    """Entry point for every script: run it, and report ApiError as a message.

    A script that can't reach the server should say so in one line. A traceback
    here is noise — there is no bug to read in it.
    """
    import asyncio

    try:
        asyncio.run(coro)
    except ApiError as e:
        print(f"\033[31m{e}\033[0m", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
