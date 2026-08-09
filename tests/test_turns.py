"""The turn log, which used to be a psql query in the Makefile.

`make turns` ran raw SQL against agent-db and printed the demo chart. That
worked while there was one target and one client; it stopped being defensible
once a CLI needed the same numbers. The contract this file pins is that the
endpoint carries everything that query printed, so the Makefile target can be a
call rather than a second implementation of the predicate.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from psycopg import AsyncConnection

from app import store
from tests.conftest import DEFAULT_CONNECTION as A
from tests.conftest import OTHER_CONNECTION as B

# Exactly the columns Makefile's `turns` target selected.
WHAT_MAKE_TURNS_PRINTED = {
    "id", "question", "explored", "tool_calls", "cache_entries",
    "tokens_in", "tokens_out", "latency_ms",
}


@pytest.fixture(autouse=True)
async def clean(agent_conn: AsyncConnection):
    async def wipe():
        for cid in (A, B):
            await agent_conn.execute("DELETE FROM turn WHERE connection_id = %s", (cid,))

    await wipe()
    yield
    await wipe()


async def ask(conn: AsyncConnection, cid: str, question: str, **finish) -> int:
    turn_id = await store.start_turn(
        conn, connection_id=cid, session_id="99999999-9999-9999-9999-999999999999",
        question=question,
    )
    if finish:
        await store.finish_turn(conn, turn_id, **finish)
    return turn_id


async def test_turns_carries_everything_make_turns_printed(client: AsyncClient, agent_conn):
    await ask(
        agent_conn, A, "how many customers do we have?",
        answer="1,840", sql="SELECT 1", tool_calls=7, explored=True,
        tokens_in=10_800, tokens_out=705, latency_ms=34_000, cache_entries=0,
    )
    (row,) = (await client.get(f"/v1/connections/{A}/turns")).json()["turns"]

    assert WHAT_MAKE_TURNS_PRINTED <= set(row)
    assert row["question"] == "how many customers do we have?"
    assert row["explored"] is True and row["tool_calls"] == 7
    # in + out precomputed: it is the chart's y value, and every client would
    # otherwise add the two together itself.
    assert row["tokens"] == 11_505


async def test_turns_are_scoped_to_the_connection(client: AsyncClient, agent_conn):
    await ask(agent_conn, A, "about a", answer="x")
    await ask(agent_conn, B, "about b", answer="y")

    a = (await client.get(f"/v1/connections/{A}/turns")).json()["turns"]
    b = (await client.get(f"/v1/connections/{B}/turns")).json()["turns"]
    assert [t["question"] for t in a] == ["about a"]
    assert [t["question"] for t in b] == ["about b"]


async def test_turns_read_left_to_right_like_the_chart(client: AsyncClient, agent_conn):
    """Queried newest-first so a long-lived connection paginates; returned
    ascending, because the point of the log is watching the cost fall."""
    for n in range(1, 4):
        await ask(agent_conn, A, f"question {n}", answer="ok")

    rows = (await client.get(f"/v1/connections/{A}/turns")).json()["turns"]
    assert [t["question"] for t in rows] == ["question 1", "question 2", "question 3"]
    assert [t["id"] for t in rows] == sorted(t["id"] for t in rows)


async def test_unfinished_turns_are_excluded_by_default(client: AsyncClient, agent_conn):
    await ask(agent_conn, A, "answered", answer="done")
    await ask(agent_conn, A, "still running")

    default = (await client.get(f"/v1/connections/{A}/turns")).json()["turns"]
    assert [t["question"] for t in default] == ["answered"]

    everything = (
        await client.get(f"/v1/connections/{A}/turns", params={"finished": "false"})
    ).json()["turns"]
    assert [t["question"] for t in everything] == ["answered", "still running"]


async def test_limit_keeps_the_most_recent(client: AsyncClient, agent_conn):
    for n in range(1, 6):
        await ask(agent_conn, A, f"question {n}", answer="ok")

    rows = (
        await client.get(f"/v1/connections/{A}/turns", params={"limit": 2})
    ).json()["turns"]
    assert [t["question"] for t in rows] == ["question 4", "question 5"]


@pytest.mark.parametrize("limit", [0, 501, "many"])
async def test_a_nonsense_limit_is_422(client: AsyncClient, limit):
    resp = await client.get(f"/v1/connections/{A}/turns", params={"limit": limit})
    assert resp.status_code == 422
