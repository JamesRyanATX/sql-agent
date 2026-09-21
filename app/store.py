"""Reads and writes for the cache and the turn log.

The cache is the product (PLAN.md §6.2), so this layer holds the two rules that
protect it: a human's correction is never silently overwritten, and every entry
records a fingerprint of the schema it was learned against.

**Two servers, and the functions here are not interchangeable about which.**
Everything takes a `psycopg.AsyncConnection` to the agent's own database except
`reflect_columns`, `schema_fingerprint`, `fingerprint_entries` and `stale_ids`,
which take a **SQLAlchemy** connection to the target.

One memory, for the one database `TARGET_DATABASE_URL` names. There was a
registry of databases once and the cache was partitioned by it; migrations/006
undoes that, and the idea with it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection, sql
from sqlalchemy.ext.asyncio import AsyncConnection as TargetConnection

_COLUMNS = """
    id, kind, name, claim, sql_fragment, tables, origin, pinned, disabled,
    tombstone, verified, hits, schema_fp, created_turn, last_used_turn
"""

_CONNECTION_COLUMNS = """
    id, label, origin, driver, host, port, database, username, password,
    sslmode, options, created_at, updated_at
"""


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


async def reflect_columns(
    conn: TargetConnection, tables: Sequence[str]
) -> dict[str, Any]:
    """One batched reflection pass over `tables`. **Takes a target connection.**"""
    from sqlalchemy import inspect

    def work(sync_conn: Any) -> dict[str, Any]:
        inspector = inspect(sync_conn)
        wanted = sorted({t for t in tables})
        present = set(inspector.get_table_names())
        multi = inspector.get_multi_columns(
            filter_names=[t for t in wanted if t in present]
        )
        shapes: dict[str, Any] = {
            key[1]: [
                (c["name"], str(c["type"]), bool(c["nullable"])) for c in cols
            ]
            for key, cols in multi.items()
        }
        shapes["__dialect__"] = sync_conn.dialect.name
        return shapes

    return await conn.run_sync(work)


def fingerprint(shapes: dict[str, Any], tables: Sequence[str]) -> str:
    """Hash `tables`' shape out of an already-reflected map. Pure."""
    lines = [
        # The dialect leads, so an entry fingerprinted against Postgres cannot
        # compare equal to the same table on MySQL — which catches the case the
        # PATCH refusing a driver change cannot see: the address stayed the same
        # and the database behind it did not.
        f"dialect:{shapes.get('__dialect__', '')}"
    ]
    for t in sorted(tables):
        for name, type_name, nullable in shapes.get(t, []):
            lines.append(f"{t}.{name}:{type_name}:{nullable}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


async def schema_fingerprint(
    conn: TargetConnection, tables: Sequence[str]
) -> str:
    """Hash the live shape of `tables`. **Takes a target connection.**

    Stored at write time and recomputed on load; a mismatch means the schema
    moved under a recipe learned against the old shape (§5). The type strings
    come from the dialect's own reflection, so a fingerprint from one engine is
    not comparable with one from another.
    """
    return fingerprint(await reflect_columns(conn, tables), tables)


async def fingerprint_entries(
    conn: TargetConnection, entries: Sequence[CacheEntry]
) -> None:
    """Stamp each entry with the shape of the tables it describes, in place.

    **Takes a target connection**, so it must run before `write_entries`. An
    entry naming no tables gets no fingerprint.
    """
    pending = [e for e in entries if e.schema_fp is None and e.tables]
    if not pending:
        return
    shapes = await reflect_columns(conn, [t for e in pending for t in e.tables])
    for e in pending:
        e.schema_fp = fingerprint(shapes, e.tables)


# --------------------------------------------------------------------- cache


async def load_cache(conn: AsyncConnection) -> list[CacheEntry]:
    """Everything learned and not disabled, ordered by hits.

    All of it, every turn — it fits in context, and retrieval would only add a
    way to miss the entry you needed (§4). Tombstones included: a visible
    negative constraint stops exploration rediscovering the same wrong thing (§5).
    """
    cur = await conn.execute(
        f"SELECT {_COLUMNS} FROM cache_entry "
        "WHERE NOT disabled ORDER BY hits DESC, id ASC"
    )
    return [CacheEntry.from_row(r) for r in await cur.fetchall()]


async def count_disabled(conn: AsyncConnection) -> int:
    """How many entries `load_cache` filtered out."""
    cur = await conn.execute(
        "SELECT count(*) AS n FROM cache_entry WHERE disabled"
    )
    row = await cur.fetchone()
    return row["n"] if row else 0


async def stale_ids(
    conn: TargetConnection, entries: Sequence[CacheEntry]
) -> set[int]:
    """Which entries were learned against a schema that has since moved?

    **Takes a target connection.** An entry with no fingerprint or no tables is never reported stale, so
    silence means "unknown" rather than "fine". Reporting only, for now (§5).
    """
    checkable = [e for e in entries if e.id is not None and e.schema_fp and e.tables]
    if not checkable:
        return set()
    shapes = await reflect_columns(conn, [t for e in checkable for t in e.tables])
    return {
        e.id
        for e in checkable
        if fingerprint(shapes, e.tables) != e.schema_fp
        if e.id is not None
    }


async def write_entries(
    conn: AsyncConnection,
    entries: Sequence[CacheEntry],
    *,
    turn_id: int | None = None,
) -> list[int]:
    """Insert or refresh learned entries. Returns the ids actually written.

    **Takes an agent connection**, and does not compute fingerprints — that is
    `fingerprint_entries`, which needs the target.

    Named entries upsert, so learning more about `revenue` refines that entry
    rather than filing a second one. **A human's pinned entry is never
    overwritten**: it is skipped, and its id is absent from the return value.
    """
    written: list[int] = []
    for e in entries:
        cur = await conn.execute(
            """
            INSERT INTO cache_entry (
                kind, name, claim, sql_fragment, tables, origin,
                pinned, disabled, tombstone, verified, schema_fp,
                created_turn, last_used_turn
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            -- Postgres infers cache_entry_name_key from the column plus the
            -- matching predicate.
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
                e.schema_fp,
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
    conn: AsyncConnection,
    ids: Sequence[int],
    *,
    turn_id: int | None = None,
) -> None:
    """Mark the entries a turn actually used.

    `hits` orders the cache; `last_used_turn` is what lets compaction drop
    entries nothing has needed.
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


async def _wipe(conn: AsyncConnection, table: str) -> int:
    """Empty one table, returning how many rows it held."""
    cur = await conn.execute(
        sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(table))
    )
    row = await cur.fetchone()
    await conn.execute(
        sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
            sql.Identifier(table)
        )
    )
    return row["n"] if row else 0


async def reset_learned(conn: AsyncConnection) -> dict[str, int]:
    """Forget everything the agent learned. Rows-per-table.

    The stage recovery button (PLAN.md §9), and what `sql-agent reset` calls.
    Cache, turn log and LangGraph's checkpoints go together, because a
    checkpointed TurnState holds a `turn_id` and cache-entry ids that would
    otherwise dangle.

    `checkpoint_migrations` is excluded: it is a schema version, not turn state,
    and deleting it makes the next checkpointer start-up rebuild its tables.
    """
    wiped = {
        table: await _wipe(conn, table)
        for table in ("cache_entry", "turn", "checkpoints",
                      "checkpoint_blobs", "checkpoint_writes")
    }
    return wiped


async def reset_everything(conn: AsyncConnection) -> dict[str, int]:
    """Empty the agent's database, `checkpoint_migrations` included.

    **Not exposed on the API** — this is for the test suite. The difference from
    `reset_learned` is that one table: dropping the checkpointer's schema
    version makes it rebuild its tables on the next start-up, which a running
    server does not want done underneath it.

    A `langgraph-checkpoint-postgres` release adding a seventh table has to be
    added here by hand, and the failure is quiet, so check after an upgrade.
    """
    return {
        "cache_entry": await _wipe(conn, "cache_entry"),
        "turn": await _wipe(conn, "turn"),
        "checkpoints": await _wipe(conn, "checkpoints"),
        "checkpoint_blobs": await _wipe(conn, "checkpoint_blobs"),
        "checkpoint_writes": await _wipe(conn, "checkpoint_writes"),
        "checkpoint_migrations": await _wipe(conn, "checkpoint_migrations"),
    }


async def start_turn(
    conn: AsyncConnection, *, session_id: str | UUID, question: str
) -> int:
    """Open the turn row and return its id. Split from `finish_turn` because
    `extract` writes entries mid-turn that need a `created_turn` to point at.
    """
    cur = await conn.execute(
        "INSERT INTO turn (session_id, question) "
        "VALUES (%s, %s) RETURNING id",
        (str(session_id), question),
    )
    row = await cur.fetchone()
    assert row is not None
    return row["id"]


async def fail_open_turn(
    conn: AsyncConnection,
    session_id: str | UUID,
    message: str,
) -> int | None:
    """Close the most recent unfinished turn for a session.

    The row is opened before any model call, so without this anything that
    throws leaves it indistinguishable from a turn still in flight.
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


async def read_turns(
    conn: AsyncConnection, *, limit: int = 50, finished: bool = True
) -> list[dict[str, Any]]:
    """The demo chart, as rows: what each turn asked and what it cost.

    Queried newest-first with a LIMIT so a long-lived install paginates, and
    returned **ascending** so it reads left to right. `finished=False` also
    shows the turns still in flight and the ones that failed.
    """
    cur = await conn.execute(
        f"""
        SELECT id, question, sql, answer, tool_calls, explored,
               tokens_in, tokens_out, latency_ms, cache_entries, created_at,
               trace_id, cost
        FROM turn
        {"WHERE answer IS NOT NULL" if finished else ""}
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,),
    )
    return list(reversed(await cur.fetchall()))


async def get_turn(conn: AsyncConnection, turn_id: int) -> dict[str, Any] | None:
    """One turn, or None. The caller turns None into a 404."""
    cur = await conn.execute(
        "SELECT id, question, answer, trace_id FROM turn "
        "WHERE id = %s",
        (turn_id,),
    )
    return await cur.fetchone()


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
    trace_id: str | None = None,
    cost: float | None = None,
) -> None:
    """Record what the turn cost, in tokens and in money. This is the demo chart.

    `cost` is None wherever the backend did not itemise the charge, which is
    every backend but OpenRouter today. None and 0 are different answers.
    """
    await conn.execute(
        """
        UPDATE turn SET
            sql = %s, answer = %s, tool_calls = %s, explored = %s,
            tokens_in = %s, tokens_out = %s, latency_ms = %s, cache_entries = %s,
            trace_id = %s, cost = %s
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
            trace_id,
            cost,
            turn_id,
        ),
    )
