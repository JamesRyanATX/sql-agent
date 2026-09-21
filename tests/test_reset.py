"""Forgetting.

`sql-agent reset` empties the one memory. What makes that more than a DELETE is
LangGraph's checkpoint tables: a checkpointed TurnState holds a `turn_id` and a
list of cache-entry ids, so leaving the checkpoints behind after wiping the rows
they name is wrong rather than untidy.

The second thing these guard is the list of table names in `store`. It is
written out by hand, and a LangGraph release that adds a table would otherwise
be missed in silence.
"""

from __future__ import annotations

import pytest
from psycopg import AsyncConnection

from app import store

SESSION = "aaaaaaaa-0000-0000-0000-000000000001"

LEARNED = {"cache_entry", "turn", "checkpoints", "checkpoint_blobs", "checkpoint_writes"}


async def table_names(conn: AsyncConnection) -> set[str]:
    cur = await conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    )
    return {r["table_name"] for r in await cur.fetchall()}


async def seed(conn: AsyncConnection) -> int:
    """One finished turn, one entry, and a checkpoint row for that thread."""
    turn_id = await store.start_turn(conn, session_id=SESSION, question="how many?")
    await store.finish_turn(conn, turn_id, answer="42", tokens_in=1, tokens_out=1)
    await store.write_entries(
        conn, [store.CacheEntry(kind="recipe", name="spec:customers", claim="x")]
    )
    # A checkpoint, as LangGraph writes them: keyed by thread_id, nothing else.
    await conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
        "type, checkpoint, metadata) VALUES (%s, '', 'cp-1', 'x', '{}', '{}') "
        "ON CONFLICT DO NOTHING",
        (SESSION,),
    )
    return turn_id


@pytest.fixture(autouse=True)
async def clean(agent_conn: AsyncConnection):
    """These tests empty tables other tests are also using, so they run against
    an empty memory and hand one back."""
    await store.reset_learned(agent_conn)
    yield
    await store.reset_learned(agent_conn)


async def test_reset_takes_the_cache_the_turns_and_the_checkpoints(agent_conn):
    await seed(agent_conn)

    wiped = await store.reset_learned(agent_conn)
    assert wiped["cache_entry"] == 1
    assert wiped["turn"] == 1
    assert wiped["checkpoints"] == 1

    assert await store.load_cache(agent_conn) == []
    assert await store.read_turns(agent_conn) == []


async def test_reset_leaves_checkpoint_migrations_alone(agent_conn):
    """It is LangGraph's schema-version table, not turn state. Emptying it makes
    the next setup() re-run every migration, two of which are CREATE INDEX
    CONCURRENTLY and cannot execute inside a transaction."""
    cur = await agent_conn.execute("SELECT count(*) AS n FROM checkpoint_migrations")
    before = (await cur.fetchone())["n"]

    wiped = await store.reset_learned(agent_conn)
    assert "checkpoint_migrations" not in wiped

    cur = await agent_conn.execute("SELECT count(*) AS n FROM checkpoint_migrations")
    assert (await cur.fetchone())["n"] == before


async def test_a_second_reset_is_a_no_op(agent_conn):
    await seed(agent_conn)
    await store.reset_learned(agent_conn)
    assert await store.reset_learned(agent_conn) == dict.fromkeys(LEARNED, 0)


async def test_the_global_reset_names_every_table_in_the_agent_database(agent_conn):
    """The longhand list in `reset_everything` has to stay complete. A LangGraph
    upgrade that adds a checkpoint table would otherwise leave it behind, and
    the reset would quietly stop being a reset."""
    wiped = await store.reset_everything(agent_conn)
    assert set(wiped) == await table_names(agent_conn)
