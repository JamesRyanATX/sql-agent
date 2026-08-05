"""Phase 0 smoke tests: the plumbing is wired, not that the agent is smart.

Requires Postgres up (`make up`).
"""

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.events import to_events
from app.main import app


@pytest.fixture
async def client():
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.live
async def test_ask_streams_answer_then_done(client: AsyncClient):
    body = {
        "session_id": "33333333-3333-3333-3333-333333333333",
        "question": "how many customers do we have?",
    }
    async with client.stream("POST", "/ask", json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["x-accel-buffering"] == "no"
        text = "".join([chunk async for chunk in resp.aiter_text()])

    types = [
        line.removeprefix("event: ")
        for line in text.splitlines()
        if line.startswith("event: ")
    ]
    assert "answer" in types
    assert types[-1] == "done", "the stream must always terminate with done"


def test_custom_chunks_pass_through():
    ev = {"type": "explore", "tool": "list_tables"}
    assert to_events("custom", ev) == [ev]
    assert to_events("custom", {"no": "type key"}) == []


def test_updates_chunks_become_usage_events():
    chunk = {"answer": {"tokens_in": 120, "tokens_out": 40, "answer": "1,840"}}
    assert to_events("updates", chunk) == [
        {"type": "usage", "node": "answer", "tokens_in": 120, "tokens_out": 40}
    ]
    # A node that reports no tokens shouldn't move the counter at all.
    assert to_events("updates", {"load_cache": {"cache": []}}) == []
