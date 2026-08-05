"""Phase 2: the cache round-trips, and fingerprints track schema shape.

Requires Postgres up and migrated (`make up && make migrate`).
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from app.settings import settings
from app.store import (
    CacheEntry,
    bump_hits,
    finish_turn,
    load_cache,
    schema_fingerprint,
    start_turn,
    write_entries,
)

FP_TABLE = "fp_probe"


@pytest.fixture
async def conn() -> AsyncIterator[AsyncConnection]:
    """A connection whose writes are rolled back, so tests don't leak rows."""
    async with await psycopg.AsyncConnection.connect(
        settings().database_url, row_factory=dict_row
    ) as c:
        yield c
        await c.rollback()


@pytest.fixture
async def probe_table(conn: AsyncConnection) -> AsyncIterator[str]:
    """A throwaway table to fingerprint. DDL is transactional in Postgres, so
    the rollback in the `conn` fixture drops it again."""
    await conn.execute(
        f"CREATE TABLE {FP_TABLE} (id bigint, label text, created timestamptz)"
    )
    yield FP_TABLE


# ---------------------------------------------------------------- fingerprint


async def test_fingerprint_is_stable_across_calls(conn, probe_table):
    assert await schema_fingerprint(conn, [probe_table]) == await schema_fingerprint(
        conn, [probe_table]
    )


async def test_fingerprint_changes_when_a_column_is_renamed(conn, probe_table):
    before = await schema_fingerprint(conn, [probe_table])
    # The `created` vs `created_at` trap, as a schema change.
    await conn.execute(f"ALTER TABLE {probe_table} RENAME COLUMN created TO created_at")
    assert await schema_fingerprint(conn, [probe_table]) != before


async def test_fingerprint_changes_on_add_drop_and_type_change(conn, probe_table):
    before = await schema_fingerprint(conn, [probe_table])

    await conn.execute(f"ALTER TABLE {probe_table} ADD COLUMN deleted_at timestamptz")
    added = await schema_fingerprint(conn, [probe_table])
    assert added != before

    await conn.execute(f"ALTER TABLE {probe_table} DROP COLUMN deleted_at")
    assert await schema_fingerprint(conn, [probe_table]) == before

    await conn.execute(f"ALTER TABLE {probe_table} ALTER COLUMN label TYPE varchar(64)")
    assert await schema_fingerprint(conn, [probe_table]) != before


async def test_fingerprint_ignores_unrelated_tables(conn, probe_table):
    before = await schema_fingerprint(conn, [probe_table])
    await conn.execute("CREATE TABLE fp_unrelated (x int)")
    assert await schema_fingerprint(conn, [probe_table]) == before


# --------------------------------------------------------------------- cache


async def test_write_and_load_round_trip(conn, probe_table):
    turn_id = await start_turn(conn, session_id=uuid4(), question="how many customers?")
    entry = CacheEntry(
        kind="recipe",
        name="spec:store roundtrip",
        claim="An active customer is one whose deleted_at is null.",
        sql_fragment="customer WHERE deleted_at IS NULL",
        tables=[probe_table],
        verified=True,
    )
    written = await write_entries(conn, [entry], turn_id=turn_id)
    assert len(written) == 1

    loaded = {e.name: e for e in await load_cache(conn)}
    got = loaded["spec:store roundtrip"]
    assert got.kind == "recipe"
    assert got.sql_fragment == "customer WHERE deleted_at IS NULL"
    assert got.tables == [probe_table]
    assert got.verified is True
    assert got.origin == "learned"
    assert got.created_turn == turn_id
    # Stamped at write time, so drift is detectable later.
    assert got.schema_fp == await schema_fingerprint(conn, [probe_table])


async def test_named_entries_upsert_rather_than_duplicate(conn):
    first = CacheEntry(kind="recipe", name="spec:revenue", claim="qty * price", tables=[])
    await write_entries(conn, [first])
    refined = CacheEntry(
        kind="recipe",
        name="spec:revenue",
        claim="qty * price, excluding cancelled orders",
        tables=[],
    )
    await write_entries(conn, [refined])

    matches = [e for e in await load_cache(conn) if e.name == "spec:revenue"]
    assert len(matches) == 1
    assert matches[0].claim == "qty * price, excluding cancelled orders"


async def test_a_humans_pinned_entry_is_never_overwritten(conn):
    """The correction someone made in /admin/cache survives the next extraction.

    An extraction quietly reverting a human's fix is the worst bug this system
    can have — every future question that composes the recipe inherits it.
    """
    human = CacheEntry(
        kind="recipe",
        name="spec:revenue",
        claim="SUM(oi.qty * oi.price) excluding cancelled orders",
        origin="human",
        pinned=True,
        tables=[],
    )
    await write_entries(conn, [human])

    relearned = CacheEntry(
        kind="recipe", name="spec:revenue", claim="SUM(oi.qty * p.unit_price)", tables=[]
    )
    written = await write_entries(conn, [relearned])

    assert written == [], "the pinned row must not be updated"
    survivor = next(e for e in await load_cache(conn) if e.name == "spec:revenue")
    assert survivor.claim == "SUM(oi.qty * oi.price) excluding cancelled orders"
    assert survivor.origin == "human"


async def test_load_cache_orders_by_hits_and_skips_disabled(conn):
    await write_entries(
        conn,
        [
            CacheEntry(kind="schema_fact", name="spec:cold", claim="rarely used", tables=[]),
            CacheEntry(kind="schema_fact", name="spec:hot", claim="always used", tables=[]),
            CacheEntry(
                kind="schema_fact",
                name="spec:off",
                claim="switched off",
                tables=[],
                disabled=True,
            ),
        ],
    )
    cache = {e.name: e for e in await load_cache(conn)}
    assert "spec:off" not in cache

    await bump_hits(conn, [cache["spec:hot"].id], turn_id=7)
    reloaded = await load_cache(conn)
    assert reloaded[0].name == "spec:hot"
    assert reloaded[0].hits == 1
    assert reloaded[0].last_used_turn == 7


async def test_tombstones_stay_visible_to_the_model(conn):
    """A tombstone is content, not a deletion — it's the negative constraint
    that stops exploration relearning the same wrong thing (§5)."""
    await write_entries(
        conn,
        [
            CacheEntry(
                kind="recipe",
                name="spec:revenue excludes cancelled",
                claim="revenue does not include cancelled orders",
                tables=[],
                tombstone=True,
            )
        ],
    )
    loaded = await load_cache(conn)
    assert any(e.tombstone for e in loaded)


# --------------------------------------------------------------------- turns


async def test_turn_records_what_it_cost(conn):
    session = uuid4()
    turn_id = await start_turn(conn, session_id=session, question="how many customers?")
    await finish_turn(
        conn,
        turn_id,
        sql="SELECT count(*) FROM customer WHERE deleted_at IS NULL",
        answer="1,840 customers (excluding soft-deleted).",
        tool_calls=11,
        explored=True,
        tokens_in=7800,
        tokens_out=600,
        latency_ms=14200,
        cache_entries=0,
    )

    cur = await conn.execute("SELECT * FROM turn WHERE id = %s", (turn_id,))
    row = await cur.fetchone()
    assert row["explored"] is True
    assert row["tokens_in"] + row["tokens_out"] == 8400  # the chart's y value
    assert row["cache_entries"] == 0
    assert str(row["session_id"]) == str(session)
