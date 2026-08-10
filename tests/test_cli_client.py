"""SSE framing in sql_agent_cli/http.py.

The CLI renders whatever comes out of here, so a dropped or mis-split frame is a
turn that looks like it never finished. Driven through a mock transport rather
than a live turn: the framing is the thing under test, not the agent.

This file imports both sides on purpose — `app.events.sse` builds the fixtures
and `sql_agent_cli.http` parses them. It is the contract test for the split, and
the one place in the suite where the two halves are allowed to meet.
"""

import httpx
import pytest

from app.events import sse
from sql_agent_cli import http as _client


def stream_of(*chunks: str) -> httpx.MockTransport:
    """A transport that hands back exactly these byte chunks, in order.

    Chunk boundaries are the point — a real server splits wherever the network
    happens to, not on frame boundaries.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(b"".join(c.encode() for c in chunks)),
        )

    return httpx.MockTransport(handler)


async def collect(transport: httpx.MockTransport, monkeypatch) -> list[dict]:
    monkeypatch.setattr(
        _client, "_client", lambda timeout: httpx.AsyncClient(
            transport=transport, base_url="http://test"
        )
    )
    return [ev async for ev in _client.stream_events("/ask", {"question": "x"})]


# ---------------------------------------------------------------------- framing


async def test_events_round_trip_exactly_what_the_nodes_emitted(monkeypatch):
    """`sse()` only serialises; a client must get the same dicts back."""
    events = [
        {"type": "plan", "cache_entries": 0, "sufficient": False, "missing": []},
        {"type": "explore", "tool": "list_tables", "input": {}, "calls": 1},
        {"type": "answer", "text": "1,840 active customers.", "total_tokens": 371},
        {"type": "done"},
    ]
    got = await collect(stream_of(*(sse(e) for e in events)), monkeypatch)
    assert got == events


async def test_one_chunk_carrying_several_frames_is_split(monkeypatch):
    """Frames arrive batched far more often than one per read."""
    events = [{"type": "usage", "node": "explore", "tokens_in": i} for i in range(4)]
    got = await collect(stream_of("".join(sse(e) for e in events)), monkeypatch)
    assert got == events


async def test_a_frame_split_across_chunks_is_reassembled(monkeypatch):
    """The JSON payload of a findings event is long enough to straddle reads."""
    frame = sse({"type": "findings", "text": "customer.deleted_at is populated"})
    half = len(frame) // 2
    got = await collect(stream_of(frame[:half], frame[half:]), monkeypatch)
    assert got == [{"type": "findings", "text": "customer.deleted_at is populated"}]


async def test_a_stream_cut_mid_frame_still_yields_its_last_event(monkeypatch):
    """No trailing blank line. That last event is usually the one that says what
    went wrong, so losing it loses the diagnosis."""
    truncated = sse({"type": "error", "message": "ReadTimeout", "fatal": True}).rstrip()
    got = await collect(stream_of(truncated), monkeypatch)
    assert got == [{"type": "error", "message": "ReadTimeout", "fatal": True}]


def test_a_frame_with_no_data_line_yields_nothing():
    """Comments and bare keep-alives are not events."""
    assert _client.parse_sse([": keep-alive"]) is None
    assert _client.parse_sse([]) is None
    assert _client.parse_sse(["event: done"]) is None


def test_multiline_data_is_rejoined():
    """Per the SSE spec, consecutive data: lines are one payload."""
    assert _client.parse_sse(['data: {"type":', 'data: "done"}']) == {"type": "done"}


# ------------------------------------------------------------------ the plumbing


def test_the_ask_stream_has_no_read_timeout():
    """A T1 turn runs minutes. httpx's 5s default would kill every one of them —
    the same failure app/llm.py documents for the local backend."""
    assert _client.STREAM_TIMEOUT.read is None
    assert _client.STREAM_TIMEOUT.connect == 5.0


def test_the_token_is_sent_only_when_one_is_set(monkeypatch):
    monkeypatch.delenv("SQL_AGENT_API_KEY", raising=False)
    assert _client._headers() == {}

    monkeypatch.setenv("SQL_AGENT_API_KEY", "s3cret")
    assert _client._headers() == {"authorization": "Bearer s3cret"}


def test_the_base_url_keeps_its_v1_prefix(monkeypatch):
    """SQL_AGENT_URL carries /v1 and every call site writes the path without it.
    httpx preserves a base_url's path when merging; urljoin would eat it."""
    monkeypatch.setenv("SQL_AGENT_URL", "http://localhost:3000/v1")
    client = _client._client(_client.REQUEST_TIMEOUT)
    merged = client._merge_url("/connections/prod/cache")
    assert str(merged) == "http://localhost:3000/v1/connections/prod/cache"


class _DyingStream(httpx.AsyncByteStream):
    """Hands back `chunk`, then drops the connection the way a reload does."""

    def __init__(self, chunk: bytes, request: httpx.Request) -> None:
        self._chunk, self._request = chunk, request

    async def __aiter__(self):
        yield self._chunk
        raise httpx.RemoteProtocolError("peer closed", request=self._request)


async def test_a_stream_that_dies_mid_turn_is_a_message_not_a_traceback(monkeypatch):
    """A turn runs for minutes with no read timeout. The api container
    restarting under --reload raises out of aiter_lines, three minutes in, and a
    traceback there says nothing anyone can act on. The count distinguishes
    "nothing arrived" from "it died most of the way through"."""

    def die_after_two(request: httpx.Request) -> httpx.Response:
        two = (sse({"type": "usage", "node": "explore"}) * 2).encode()
        return httpx.Response(200, stream=_DyingStream(two, request))

    monkeypatch.setattr(
        _client, "_client", lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(die_after_two), base_url="http://test"
        )
    )
    with pytest.raises(_client.ApiError, match="stream ended after 2 events"):
        async for _ in _client.stream_events("/ask", {"question": "x"}):
            pass


async def test_a_server_that_is_not_running_is_a_message_not_a_traceback(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(
        _client, "_client", lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(refuse), base_url="http://test"
        )
    )
    with pytest.raises(_client.ApiError, match="no API at"):
        await _client.get("/cache")


async def test_a_401_says_what_to_do_about_it(monkeypatch):
    monkeypatch.setattr(
        _client, "_client", lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(401)),
            base_url="http://test",
        )
    )
    with pytest.raises(_client.ApiError, match="API_TOKEN"):
        await _client.get("/cache")


async def test_an_error_before_the_stream_starts_is_raised_not_streamed(monkeypatch):
    """A 500 on /ask would otherwise render as a turn that produced no events."""
    monkeypatch.setattr(
        _client, "_client", lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(500, text="boom")
            ),
            base_url="http://test",
        )
    )
    with pytest.raises(_client.ApiError, match="500"):
        async for _ in _client.stream_events("/ask", {"question": "x"}):
            pass


async def test_a_server_that_dies_mid_request_is_a_message_too(monkeypatch):
    """The non-streaming twin of the test above. Found the hard way: a container
    crashing on startup accepts the connection and then closes it, and every
    `sql-agent connections ls` printed forty lines of httpx traceback."""

    def die(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("server disconnected", request=request)

    monkeypatch.setattr(
        _client, "_client", lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(die), base_url="http://test"
        )
    )
    with pytest.raises(_client.ApiError, match="accepted the connection and then closed"):
        await _client.get("/connections")
