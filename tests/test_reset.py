"""Forgetting one connection, and forgetting everything.

The scoped reset is the awkward one. LangGraph's checkpoint tables are keyed by
`thread_id` — the session UUID — and know nothing about connections, so the
mapping runs through `turn`, which records both. That makes the delete order
load-bearing: the turn rows *are* the mapping, so they go last.
"""

from __future__ import annotations

import pytest
from psycopg import AsyncConnection

from app import store
from tests.conftest import DEFAULT_CONNECTION as A
from tests.conftest import OTHER_CONNECTION as B

SESSION_A = "aaaaaaaa-0000-0000-0000-000000000001"
SESSION_B = "bbbbbbbb-0000-0000-0000-000000000002"


async def table_names(conn: AsyncConnection) -> set[str]:
    cur = await conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    )
    return {r["table_name"] for r in await cur.fetchall()}


async def seed(conn: AsyncConnection, cid: str, session: str) -> int:
    """One turn and one entry on `cid`, plus a checkpoint row for its thread."""
    turn_id = await store.start_turn(
        conn, connection_id=cid, session_id=session, question=f"about {cid}?"
    )
    await store.finish_turn(conn, turn_id, answer="42", tokens_in=1, tokens_out=1)
    await store.write_entries(
        conn, [store.CacheEntry(kind="recipe", name=f"spec:{cid}", claim="x")],
        connection_id=cid,
    )
    # A checkpoint, as LangGraph writes them: keyed by thread_id, nothing else.
    await conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
        "type, checkpoint, metadata) VALUES (%s, '', %s, 'x', '{}', '{}') "
        "ON CONFLICT DO NOTHING",
        (session, f"cp-{cid}"),
    )
    return turn_id


@pytest.fixture(autouse=True)
async def clean(agent_conn: AsyncConnection):
    async def wipe():
        for cid in (A, B):
            await agent_conn.execute("DELETE FROM cache_entry WHERE connection_id = %s", (cid,))
            await agent_conn.execute("DELETE FROM turn WHERE connection_id = %s", (cid,))
        for s in (SESSION_A, SESSION_B):
            await agent_conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (s,))

    await wipe()
    yield
    await wipe()


async def test_a_scoped_reset_takes_only_this_connections_cache_and_turns(agent_conn):
    await seed(agent_conn, A, SESSION_A)
    await seed(agent_conn, B, SESSION_B)

    wiped = await store.reset_learned(agent_conn, connection_id=A)
    assert wiped["cache_entry"] == 1 and wiped["turn"] == 1

    assert await store.load_cache(agent_conn, connection_id=A) == []
    assert [e.name for e in await store.load_cache(agent_conn, connection_id=B)] == [f"spec:{B}"]
    assert len(await store.read_turns(agent_conn, connection_id=B)) == 1


async def test_a_scoped_reset_takes_the_checkpoints_of_its_own_sessions_only(agent_conn):
    """Leaving them would be wrong, not untidy: a checkpointed TurnState holds a
    turn_id and a list of entry ids, and after this both would dangle."""
    await seed(agent_conn, A, SESSION_A)
    await seed(agent_conn, B, SESSION_B)

    wiped = await store.reset_learned(agent_conn, connection_id=A)
    assert wiped["checkpoints"] == 1

    cur = await agent_conn.execute("SELECT thread_id FROM checkpoints")
    remaining = {r["thread_id"] for r in await cur.fetchall()}
    assert SESSION_A not in remaining
    assert SESSION_B in remaining


async def test_a_scoped_reset_leaves_checkpoint_migrations_alone(agent_conn):
    """It is LangGraph's schema-version table, not turn state. Emptying it makes
    the next setup() re-run every migration, two of which are CREATE INDEX
    CONCURRENTLY and cannot execute inside a transaction."""
    cur = await agent_conn.execute("SELECT count(*) AS n FROM checkpoint_migrations")
    before = (await cur.fetchone())["n"]

    wiped = await store.reset_learned(agent_conn, connection_id=A)
    assert "checkpoint_migrations" not in wiped

    cur = await agent_conn.execute("SELECT count(*) AS n FROM checkpoint_migrations")
    assert (await cur.fetchone())["n"] == before


async def test_the_delete_order_survives_the_mapping_it_depends_on(agent_conn):
    """The turn rows are what session→connection is made of. Deleting them first
    would leave every checkpoint behind, and nothing would say so."""
    await seed(agent_conn, A, SESSION_A)
    wiped = await store.reset_learned(agent_conn, connection_id=A)
    assert wiped["checkpoints"] == 1 and wiped["turn"] == 1


async def test_a_second_reset_is_a_no_op(agent_conn):
    await seed(agent_conn, A, SESSION_A)
    await store.reset_learned(agent_conn, connection_id=A)
    assert await store.reset_learned(agent_conn, connection_id=A) == {
        "cache_entry": 0, "turn": 0, "checkpoints": 0,
        "checkpoint_blobs": 0, "checkpoint_writes": 0,
    }


async def test_the_global_reset_still_names_every_table_it_must(agent_conn):
    """The longhand list in `reset_everything` has to stay complete. A LangGraph
    upgrade that adds a checkpoint table would otherwise leave it behind, and
    the reset would quietly stop being a reset.

    `connection` is the deliberate exclusion: the registry is configuration, not
    learned state, and wiping it would delete the address of every warehouse
    somebody registered.
    """
    wiped = await store.reset_everything(agent_conn)
    assert set(wiped) == await table_names(agent_conn) - {"connection"}


async def test_the_global_reset_keeps_the_registry(agent_conn):
    before = {c.id for c in await store.list_connections(agent_conn)}
    assert {A, B} <= before

    await store.reset_everything(agent_conn)

    assert {c.id for c in await store.list_connections(agent_conn)} == before
