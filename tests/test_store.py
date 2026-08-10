"""Phase 2: the cache round-trips, and fingerprints track schema shape.

**Two connections, and they are not interchangeable.** `conn` is the agent's own
database, where entries live. `fp` is the demo database, whose shape they
describe — every fingerprint call takes that one. An entry is agent state about
target structure, and these tests are where that seam is checked.

Requires `make up && make migrate && make seed`.
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection as TargetConnection
from sqlalchemy.ext.asyncio import create_async_engine

from app import store
from app.settings import settings
from app.store import (
    CacheEntry,
    bump_hits,
    finish_turn,
    fingerprint_entries,
    load_cache,
    schema_fingerprint,
    start_turn,
    write_entries,
)
from tests.conftest import DEFAULT_CONNECTION as CID

FP_TABLE = "fp_probe"


async def _rollback_conn(url: str) -> AsyncIterator[AsyncConnection]:
    """A connection whose writes are rolled back, so tests don't leak rows."""
    async with await psycopg.AsyncConnection.connect(
        url, row_factory=dict_row
    ) as c:
        yield c
        await c.rollback()


@pytest.fixture
async def conn() -> AsyncIterator[AsyncConnection]:
    """The agent's memory."""
    async for c in _rollback_conn(settings().agent_database_url):
        yield c


@pytest.fixture
async def fp() -> AsyncIterator[TargetConnection]:
    """The demo database, as its owner — these tests need DDL to move a schema
    under an entry and watch the fingerprint change.

    A **SQLAlchemy** connection, because that is what the fingerprint functions
    take now. Deliberately not AUTOCOMMIT: the DDL and the reflection have to
    share one transaction, both so the reflection sees the uncommitted DDL and
    so the rollback at the end drops it again. DDL is transactional in Postgres,
    which is what makes that work.
    """
    engine = create_async_engine(
        store.connection_from_url(settings().target_admin_url, id="_fp").url()
    )
    try:
        async with engine.connect() as c:
            yield c
            await c.rollback()
    finally:
        await engine.dispose()


async def ddl(conn: TargetConnection, statement: str) -> None:
    """Raw DDL through SQLAlchemy. `text()` is fine here — unlike the model's
    SQL, these strings are ours and contain no colons."""
    await conn.execute(text(statement))


@pytest.fixture
async def probe_table(fp: TargetConnection) -> AsyncIterator[str]:
    """A throwaway table to fingerprint, on the *demo* server — that is where
    the schemas entries describe actually live."""
    await ddl(fp, f"CREATE TABLE {FP_TABLE} (id bigint, label text, created timestamptz)")
    yield FP_TABLE


# ---------------------------------------------------------------- fingerprint


async def test_fingerprint_is_stable_across_calls(fp, probe_table):
    assert await schema_fingerprint(fp, [probe_table]) == await schema_fingerprint(
        fp, [probe_table]
    )


async def test_fingerprint_changes_when_a_column_is_renamed(fp, probe_table):
    before = await schema_fingerprint(fp, [probe_table])
    # The `created` vs `created_at` trap, as a schema change.
    await ddl(fp, f"ALTER TABLE {probe_table} RENAME COLUMN created TO created_at")
    assert await schema_fingerprint(fp, [probe_table]) != before


async def test_fingerprint_changes_on_add_drop_and_type_change(fp, probe_table):
    before = await schema_fingerprint(fp, [probe_table])

    await ddl(fp, f"ALTER TABLE {probe_table} ADD COLUMN deleted_at timestamptz")
    added = await schema_fingerprint(fp, [probe_table])
    assert added != before

    await ddl(fp, f"ALTER TABLE {probe_table} DROP COLUMN deleted_at")
    assert await schema_fingerprint(fp, [probe_table]) == before

    await ddl(fp, f"ALTER TABLE {probe_table} ALTER COLUMN label TYPE varchar(64)")
    assert await schema_fingerprint(fp, [probe_table]) != before


async def test_fingerprint_ignores_unrelated_tables(fp, probe_table):
    before = await schema_fingerprint(fp, [probe_table])
    await ddl(fp, "CREATE TABLE fp_unrelated (x int)")
    assert await schema_fingerprint(fp, [probe_table]) == before


# --------------------------------------------------------------------- cache


async def test_write_and_load_round_trip(conn, fp, probe_table):
    """The full seam: fingerprint on the demo server, write on the agent's."""
    turn_id = await start_turn(conn, connection_id=CID, session_id=uuid4(), question="how many customers?")
    entry = CacheEntry(
        kind="recipe",
        name="spec:store roundtrip",
        claim="An active customer is one whose deleted_at is null.",
        sql_fragment="customer WHERE deleted_at IS NULL",
        tables=[probe_table],
        verified=True,
    )
    await fingerprint_entries(fp, [entry])
    written = await write_entries(conn, [entry], connection_id=CID, turn_id=turn_id)
    assert len(written) == 1

    loaded = {e.name: e for e in await load_cache(conn, connection_id=CID)}
    got = loaded["spec:store roundtrip"]
    assert got.kind == "recipe"
    assert got.sql_fragment == "customer WHERE deleted_at IS NULL"
    assert got.tables == [probe_table]
    assert got.verified is True
    assert got.origin == "learned"
    assert got.created_turn == turn_id
    # Stamped before the write, so drift is detectable later.
    assert got.schema_fp == await schema_fingerprint(fp, [probe_table])


async def test_writing_without_fingerprinting_leaves_it_unstamped(conn, probe_table):
    """`write_entries` no longer computes fingerprints, and must not pretend to.

    It used to fall back to hashing on its own connection. Once the databases
    split that became actively wrong — it would have fingerprinted the *agent*
    schema and stamped the result onto an entry describing business tables,
    permanently stale from the moment it was written. Unstamped is the honest
    answer, and `stale_ids` reads it as "unknown" rather than "fine".
    """
    entry = CacheEntry(
        kind="schema_fact",
        name="spec:unstamped",
        claim="written without a fingerprint",
        tables=[probe_table],
    )
    await write_entries(conn, [entry], connection_id=CID)
    got = next(e for e in await load_cache(conn, connection_id=CID) if e.name == "spec:unstamped")
    assert got.schema_fp is None


async def test_fingerprinting_skips_entries_with_no_tables(conn, fp):
    """Nothing to hash, so nothing is claimed."""
    entry = CacheEntry(kind="recipe", name="spec:tableless", claim="…", tables=[])
    await fingerprint_entries(fp, [entry])
    assert entry.schema_fp is None


async def test_named_entries_upsert_rather_than_duplicate(conn):
    first = CacheEntry(kind="recipe", name="spec:revenue", claim="qty * price", tables=[])
    await write_entries(conn, [first], connection_id=CID)
    refined = CacheEntry(
        kind="recipe",
        name="spec:revenue",
        claim="qty * price, excluding cancelled orders",
        tables=[],
    )
    await write_entries(conn, [refined], connection_id=CID)

    matches = [e for e in await load_cache(conn, connection_id=CID) if e.name == "spec:revenue"]
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
    await write_entries(conn, [human], connection_id=CID)

    relearned = CacheEntry(
        kind="recipe", name="spec:revenue", claim="SUM(oi.qty * p.unit_price)", tables=[]
    )
    written = await write_entries(conn, [relearned], connection_id=CID)

    assert written == [], "the pinned row must not be updated"
    survivor = next(e for e in await load_cache(conn, connection_id=CID) if e.name == "spec:revenue")
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
        connection_id=CID,
    )
    cache = {e.name: e for e in await load_cache(conn, connection_id=CID)}
    assert "spec:off" not in cache

    await bump_hits(conn, [cache["spec:hot"].id], connection_id=CID, turn_id=7)
    reloaded = await load_cache(conn, connection_id=CID)
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
        connection_id=CID,
    )
    loaded = await load_cache(conn, connection_id=CID)
    assert any(e.tombstone for e in loaded)


# --------------------------------------------------------------------- turns


async def test_turn_records_what_it_cost(conn):
    session = uuid4()
    turn_id = await start_turn(conn, connection_id=CID, session_id=session, question="how many customers?")
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
