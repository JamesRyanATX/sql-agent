"""Each read-only command, against the real app.

The mocked tests above pin dispatch, framing and rendering. This one pins the
thing splitting the CLI out of the server's import graph actually put at risk:
that `app/schemas.py` and the CLI's renderers still agree about field names. A
renamed key is a KeyError in production and nowhere else.

**These bypass CliRunner and await the command bodies directly**, which is not
laziness. Every command ends in `http.run(coro)` → `asyncio.run` → a fresh event
loop; drive `ASGITransport` through that and the app's lifespan ran in a
*different* loop, and psycopg fails in a way that reads as a CLI bug. Awaiting
the bodies keeps one loop, which is why each command is a thin click callback
around an `async def`.

Costs no tokens — nothing here asks a question.
"""

from __future__ import annotations

import re

import httpx
import pytest
from httpx import AsyncClient

from app import store
from sql_agent import config, connections, http, memory
from tests.conftest import DEFAULT_CONNECTION as CID


@pytest.fixture(autouse=True)
def through_the_app(client: AsyncClient, monkeypatch):
    """Point the CLI's client at the in-process app.

    `client` is the same ASGI fixture the API tests use, so this exercises the
    real routes, the real dependencies and the real serialisers.
    """
    monkeypatch.setattr(
        http,
        "_client",
        lambda timeout: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client._transport.app),
            base_url="http://test/v1",
            timeout=timeout,
        ),
    )


async def test_connections_ls_renders_the_registry(capsys):
    await connections._list()
    out = capsys.readouterr().out
    assert CID in out
    assert "rows)" in out  # the footer demo.tape waits on


async def test_connections_get_renders_one_and_hides_the_password(capsys):
    await connections._get(CID)
    out = capsys.readouterr().out
    # Matched without the padding: the label column widens whenever a longer
    # key is added, and a test that pins the spacing fails for the wrong reason.
    assert re.search(r"password\s+\((set|unset)\)", out)
    assert "reader" in out  # the username is fine to show
    assert "the environment" in out  # origin='env'
    assert "postgresql+psycopg" in out  # which engine it is
    assert "read-only  enforced" in out  # and how far that goes


async def test_connect_validates_before_it_writes(capsys):
    await connections._connect(CID)
    assert config.selected() == CID
    assert f"connected to {CID}" in capsys.readouterr().out


async def test_connect_to_a_typo_leaves_the_selection_alone(capsys):
    config.select(CID)
    with pytest.raises(http.ApiError, match="no connection named"):
        await connections._connect("wrehouse")
    assert config.selected() == CID


async def test_cache_renders_what_the_model_would_read(capsys, agent_conn):
    await store.write_entries(
        agent_conn,
        [
            store.CacheEntry(
                kind="recipe",
                name="spec:cli revenue",
                claim="revenue excludes cancelled orders",
                sql_fragment="WHERE status <> 'cancelled'",
                tables=["orders"],
                verified=True,
            )
        ],
        connection_id=CID,
    )
    try:
        await memory._cache(CID, None)
        out = capsys.readouterr().out
        assert "spec:cli revenue" in out
        assert "revenue excludes cancelled orders" in out
        assert "SQL: WHERE status <> 'cancelled'" in out
        assert "tables: orders" in out
    finally:
        await agent_conn.execute(
            "DELETE FROM cache_entry WHERE name = 'spec:cli revenue'"
        )


async def test_an_empty_cache_says_so_rather_than_printing_a_header(capsys, agent_conn):
    await agent_conn.execute("DELETE FROM cache_entry WHERE connection_id = %s", (CID,))
    await memory._cache(CID, None)
    assert "cache is empty" in capsys.readouterr().out


async def test_turns_renders_the_chart(capsys, agent_conn):
    """Two turns, not one: demo/demo.tape waits on the plural `rows)` footer to
    know the command has finished printing."""
    ids = []
    for explored, tin, tout in ((True, 11_021, 705), (False, 215, 190)):
        turn_id = await store.start_turn(
            agent_conn, connection_id=CID,
            session_id="77777777-7777-7777-7777-777777777777",
            question="how many customers do we have?",
        )
        await store.finish_turn(
            agent_conn, turn_id, answer="1,840", explored=explored, tool_calls=5,
            tokens_in=tin, tokens_out=tout, latency_ms=34_000,
        )
        ids.append(turn_id)
    try:
        await memory._turns(CID, 50, False)
        out = capsys.readouterr().out
        assert "how many customers do we have?" in out
        assert "11,726" in out  # in + out, as the endpoint precomputes it
        assert "405" in out  # and the cached turn beside it, which is the demo
        assert "(2 rows)" in out
    finally:
        await agent_conn.execute("DELETE FROM turn WHERE id = ANY(%s)", (ids,))


async def test_no_turns_yet_says_so(capsys, agent_conn):
    await agent_conn.execute("DELETE FROM turn WHERE connection_id = %s", (CID,))
    await memory._turns(CID, 50, False)
    assert "no turns yet" in capsys.readouterr().out


async def test_reset_reports_what_it_wiped(capsys, agent_conn):
    await store.write_entries(
        agent_conn,
        [store.CacheEntry(kind="recipe", name="spec:doomed", claim="x")],
        connection_id=CID,
    )
    await memory._reset(CID)
    out = capsys.readouterr().out
    assert "wiped cache_entry (1 rows)" in out
    assert "reset complete" in out

    await memory._reset(CID)
    assert "nothing to wipe" in capsys.readouterr().out
