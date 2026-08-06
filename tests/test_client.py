"""SSE framing in scripts/_client.py.

The scripts render whatever comes out of here, so a dropped or mis-split frame
is a turn that looks like it never finished. Driven through a mock transport
rather than a live turn: the framing is the thing under test, not the agent.
"""

import httpx
import pytest

from app.events import sse
from app.settings import settings
from scripts import _client


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
    return [ev async for ev in _client.stream_events("/v1/ask", {"question": "x"})]


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
    assert _client._headers() == {}

    monkeypatch.setenv("API_TOKEN", "s3cret")
    settings.cache_clear()
    try:
        assert _client._headers() == {"authorization": "Bearer s3cret"}
    finally:
        settings.cache_clear()


# --------------------------------------------------------------------- failures


async def test_a_server_that_is_not_running_is_a_message_not_a_traceback(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(
        _client, "_client", lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(refuse), base_url="http://test"
        )
    )
    with pytest.raises(_client.ApiError, match="make up"):
        await _client.get("/v1/cache")


async def test_a_401_says_what_to_do_about_it(monkeypatch):
    monkeypatch.setattr(
        _client, "_client", lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(401)),
            base_url="http://test",
        )
    )
    with pytest.raises(_client.ApiError, match="API_TOKEN"):
        await _client.get("/v1/cache")


async def test_an_error_before_the_stream_starts_is_raised_not_streamed(monkeypatch):
    """A 500 on /v1/ask would otherwise render as a turn that produced no events."""
    monkeypatch.setattr(
        _client, "_client", lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(500, text="boom")
            ),
            base_url="http://test",
        )
    )
    with pytest.raises(_client.ApiError, match="500"):
        async for _ in _client.stream_events("/v1/ask", {"question": "x"}):
            pass
