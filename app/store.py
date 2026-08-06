"""Reads and writes for the cache and the turn log.

The cache is the product (PLAN.md §6.2), so this layer holds the two rules that
protect it: a human's correction is never silently overwritten, and every entry
records a fingerprint of the schema it was learned against.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection, sql

_COLUMNS = """
    id, kind, name, claim, sql_fragment, tables, origin, pinned, disabled,
    tombstone, verified, hits, schema_fp, created_turn, last_used_turn
"""

# Everything the agent has learned, as opposed to the business it answers
# questions about. Two consumers with opposite needs, and one list because they
# have to agree: `tools` hides these from introspection so the agent can't
# explore its own memory, and `reset_learned` wipes exactly this set.
#
# Ordered rather than a set: `reset_learned` reports per-table counts, and a
# stable order keeps that output the same every run.
AGENT_TABLES: tuple[str, ...] = (
    "cache_entry",
    "turn",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
)


@dataclass(slots=True)
class CacheEntry:
    kind: str  # schema_fact | recipe
    claim: str
    tables: list[str] = field(default_factory=list)
    name: str | None = None
    sql_fragment: str | None = None
    origin: str = "learned"
    verified: bool = False
    tombstone: bool = False
    pinned: bool = False
    disabled: bool = False
    hits: int = 0
    schema_fp: str | None = None
    created_turn: int | None = None
    last_used_turn: int | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> CacheEntry:
        return cls(**{k: row[k] for k in row if k in cls.__slots__})


# ---------------------------------------------------------------- fingerprint


async def schema_fingerprint(conn: AsyncConnection, tables: Sequence[str]) -> str:
    """Hash the live shape of `tables`.

    Stored on an entry at write time; recomputed on load. A mismatch means the
    schema moved under a recipe that was learned against the old shape, so the
    entry can no longer be trusted (§5, wired up in phase 6).

    Catches renames, drops, additions and type changes. Does not catch a column
    keeping its name and changing its meaning — that one is in §10 for a reason.
    """
    cur = await conn.execute(
        """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = ANY(%s)
        ORDER BY table_name, ordinal_position
        """,
        (list(tables),),
    )
    rows = await cur.fetchall()
    payload = "\n".join(
        f"{r['table_name']}.{r['column_name']}:{r['data_type']}:{r['is_nullable']}"
        for r in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------- cache


async def load_cache(conn: AsyncConnection) -> list[CacheEntry]:
    """Everything not disabled, ordered by hits.

    All of it, every turn — it fits in context, and retrieval would only add a
    way to miss the entry you needed (§4).

    Tombstones are included deliberately. A tombstone *is* the useful content:
    a visible negative constraint that stops exploration rediscovering the same
    wrong thing next session (§5).
    """
    cur = await conn.execute(
        f"SELECT {_COLUMNS} FROM cache_entry "
        "WHERE NOT disabled ORDER BY hits DESC, id ASC"
    )
    return [CacheEntry.from_row(r) for r in await cur.fetchall()]


async def count_disabled(conn: AsyncConnection) -> int:
    """How many entries `load_cache` filtered out.

    The one number a caller cannot derive from the entries themselves, and the
    cache listing shows it so a disabled entry never goes quietly missing.
    """
    cur = await conn.execute("SELECT count(*) AS n FROM cache_entry WHERE disabled")
    row = await cur.fetchone()
    return row["n"] if row else 0


async def stale_ids(
    conn: AsyncConnection, entries: Sequence[CacheEntry]
) -> set[int]:
    """Which of these entries were learned against a schema that has since moved?

    Recomputes `schema_fp` per entry and compares. An entry with no fingerprint
    or no tables can't be checked, so it is never reported stale — silence here
    means "unknown", not "fine", which is why `infer_tables` in the graph works
    so hard to keep `tables` populated.

    Reporting only, for now. §5's invalidation is phase 6 and will use this.
    """
    stale: set[int] = set()
    for e in entries:
        if e.id is None or not e.schema_fp or not e.tables:
            continue
        if await schema_fingerprint(conn, e.tables) != e.schema_fp:
            stale.add(e.id)
    return stale


async def write_entries(
    conn: AsyncConnection,
    entries: Sequence[CacheEntry],
    *,
    turn_id: int | None = None,
) -> list[int]:
    """Insert or refresh learned entries. Returns the ids actually written.

    Named entries upsert, so re-learning `revenue` refines one row instead of
    accumulating near-duplicates for compaction to clean up later.

    **A human's pinned entry is never overwritten.** That is the whole point of
    the admin surface: correcting a recipe fixes every future question that
    composes it (§6.2), and an extraction quietly reverting that correction on
    the next turn is the worst bug this system can have. Such an entry is
    skipped and its id is absent from the return value.
    """
    written: list[int] = []
    for e in entries:
        fp = e.schema_fp or await schema_fingerprint(conn, e.tables)
        cur = await conn.execute(
            """
            INSERT INTO cache_entry (
                kind, name, claim, sql_fragment, tables, origin,
                pinned, disabled, tombstone, verified, schema_fp,
                created_turn, last_used_turn
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) WHERE name IS NOT NULL
            DO UPDATE SET
                kind         = EXCLUDED.kind,
                claim        = EXCLUDED.claim,
                sql_fragment = EXCLUDED.sql_fragment,
                tables       = EXCLUDED.tables,
                origin       = EXCLUDED.origin,
                tombstone    = EXCLUDED.tombstone,
                verified     = EXCLUDED.verified,
                schema_fp    = EXCLUDED.schema_fp,
                updated_at   = now()
            WHERE cache_entry.origin <> 'human' OR NOT cache_entry.pinned
            RETURNING id
            """,
            (
                e.kind,
                e.name,
                e.claim,
                e.sql_fragment,
                list(e.tables),
                e.origin,
                e.pinned,
                e.disabled,
                e.tombstone,
                e.verified,
                fp,
                e.created_turn if e.created_turn is not None else turn_id,
                e.last_used_turn,
            ),
        )
        row = await cur.fetchone()
        if row is not None:
            e.id = row["id"]
            written.append(row["id"])
    return written


async def bump_hits(
    conn: AsyncConnection, ids: Sequence[int], *, turn_id: int | None = None
) -> None:
    """Mark the entries a turn actually used.

    `hits` orders the cache and shows blast radius in the admin list;
    `last_used_turn` is what lets compaction drop entries nothing has needed.
    """
    if not ids:
        return
    await conn.execute(
        """
        UPDATE cache_entry
        SET hits = hits + 1,
            last_used_turn = COALESCE(%s, last_used_turn),
            updated_at = now()
        WHERE id = ANY(%s)
        """,
        (turn_id, list(ids)),
    )


# --------------------------------------------------------------------- turns


async def reset_learned(conn: AsyncConnection) -> dict[str, int]:
    """Wipe every trace of what the agent has learned. Returns rows-per-table.

    The stage recovery button (PLAN.md §9), and the only destructive operation
    the API exposes. It takes the cache, the turn log *and* LangGraph's
    checkpoints together: an empty cache beside a turn log that says the
    questions were already asked is a state nothing knows how to read.

    The business schema is not touched — that is `scripts/seed.py`'s to own.

    Tolerant of tables that don't exist, so it stays usable on a database the
    migrations haven't reached yet.
    """
    wiped: dict[str, int] = {}
    for table in AGENT_TABLES:
        cur = await conn.execute("SELECT to_regclass(%s) AS oid", (f"public.{table}",))
        row = await cur.fetchone()
        if row is None or row["oid"] is None:
            continue
        cur = await conn.execute(
            sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(table))
        )
        row = await cur.fetchone()
        wiped[table] = row["n"] if row else 0
        await conn.execute(
            sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
                sql.Identifier(table)
            )
        )
    return wiped


async def start_turn(
    conn: AsyncConnection, *, session_id: str | UUID, question: str
) -> int:
    """Open the turn row and return its id.

    Split from finish_turn because entries written by `extract` mid-turn need a
    `created_turn` to point at, and the turn's results aren't known until it
    ends.
    """
    cur = await conn.execute(
        "INSERT INTO turn (session_id, question) VALUES (%s, %s) RETURNING id",
        (str(session_id), question),
    )
    row = await cur.fetchone()
    assert row is not None
    return row["id"]


async def fail_open_turn(
    conn: AsyncConnection, session_id: str | UUID, message: str
) -> int | None:
    """Close the most recent unfinished turn for a session.

    A turn row is opened before any model call, so anything that throws between
    then and `answer` would otherwise leave it open forever — invisible in
    /stats, and indistinguishable from a turn still in flight.
    """
    cur = await conn.execute(
        """
        UPDATE turn SET answer = %s
        WHERE id = (
            SELECT id FROM turn
            WHERE session_id = %s AND answer IS NULL
            ORDER BY id DESC LIMIT 1
        )
        RETURNING id
        """,
        (message, str(session_id)),
    )
    row = await cur.fetchone()
    return row["id"] if row else None


async def finish_turn(
    conn: AsyncConnection,
    turn_id: int,
    *,
    sql: str | None = None,
    answer: str | None = None,
    tool_calls: int = 0,
    explored: bool = False,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: int | None = None,
    cache_entries: int = 0,
) -> None:
    """Record what the turn cost. This is the demo chart."""
    await conn.execute(
        """
        UPDATE turn SET
            sql = %s, answer = %s, tool_calls = %s, explored = %s,
            tokens_in = %s, tokens_out = %s, latency_ms = %s, cache_entries = %s
        WHERE id = %s
        """,
        (
            sql,
            answer,
            tool_calls,
            explored,
            tokens_in,
            tokens_out,
            latency_ms,
            cache_entries,
            turn_id,
        ),
    )
