"""Reads and writes for the connection registry, the cache and the turn log.

The cache is the product (PLAN.md §6.2), so this layer holds the two rules that
protect it: a human's correction is never silently overwritten, and every entry
records a fingerprint of the schema it was learned against.

**Two servers, and the functions here are not interchangeable about which.**
Everything takes a `psycopg.AsyncConnection` to the agent's own database except
`reflect_columns`, `schema_fingerprint`, `fingerprint_entries` and `stale_ids`,
which take a **SQLAlchemy** connection to a target — and it must be *the entry's
own* connection's target, or every entry reports stale, or coincidentally not.

**Everything touching learned state is scoped to one `connection_id`, and none
of these functions has a default for it.** A default is how one warehouse's
cache answers another warehouse's question: the answer looks right, the SQL
looks right, and the numbers come from the wrong database.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection, sql
from psycopg.types.json import Jsonb
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncConnection as TargetConnection

from app.secrets import seal, unseal

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


# ------------------------------------------------------------------- registry
#
# A registered database the agent can be pointed at. `app/db.py` uses the word
# "connection" ~20 times to mean a driver connection, so never bind a bare
# `connection` variable to one of these rows — it is `connection_id: str` or
# `registered: store.Connection`, always.


# Async-capable drivers only: a sync one would put a thread pool under the
# asyncio graph. Mirrored by a CHECK in migrations/003, since a row edited by
# hand at psql never sees a validator.
DRIVERS = ("postgresql+psycopg", "mysql+asyncmy", "sqlite+aiosqlite")

# What a user may reasonably type. `postgresql://` alone resolves to psycopg2 in
# SQLAlchemy, which is not installed.
_ALIASES = {
    "postgres": "postgresql+psycopg",
    "postgresql": "postgresql+psycopg",
    "mysql": "mysql+asyncmy",
    "mariadb": "mysql+asyncmy",
    "mariadb+asyncmy": "mysql+asyncmy",
    "sqlite": "sqlite+aiosqlite",
}
_DEFAULT_PORT = {"postgresql": 5432, "mysql": 3306}


def normalise_driver(name: str) -> str:
    """Resolve a driver name, or say what the choices are."""
    resolved = _ALIASES.get(name, name)
    if resolved not in DRIVERS:
        raise ValueError(
            f"unsupported driver {name!r} — this agent speaks {', '.join(DRIVERS)}"
        )
    return resolved


@dataclass(slots=True)
class Connection:
    id: str
    origin: str = "api"  # api | env — who owns the address
    driver: str = "postgresql+psycopg"
    label: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None  # the file path, when the driver is sqlite
    username: str | None = None
    # **Plaintext**, unsealed on read, and only ever handed to a driver. The wire
    # model in app/schemas.py has no password field at all.
    password: str | None = None
    sslmode: str = "prefer"  # postgres only; ignored elsewhere
    options: dict[str, Any] = field(default_factory=dict)
    created_at: Any = None
    updated_at: Any = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Connection:
        return cls(**{**{k: row[k] for k in row if k in cls.__slots__},
                      "password": unseal(row.get("password"))})

    @property
    def dialect(self) -> str:
        """`postgresql` | `mysql` | `sqlite` — the half of `driver` that changes
        the SQL. Nothing above app/db.py should care about the other half."""
        return self.driver.split("+", 1)[0]

    def _url(self, *, password: str | None, query: bool = True) -> URL:
        """The address as SQLAlchemy sees it.

        `URL.create`, never an f-string: a password containing `@`, `/` or `%`
        formats into a wrong URL whose authentication failure blames the
        password rather than the quoting.
        """
        if self.dialect == "sqlite":
            # host/port/username are NULL for sqlite by construction (the CHECK
            # in 003), and passing them renders an authority section aiosqlite
            # treats as part of the path.
            return URL.create(self.driver, database=self.database)
        return URL.create(
            self.driver,
            username=self.username,
            password=password,
            host=self.host,
            port=self.port,
            database=self.database,
            query=self._query() if query else {},
        )

    def _query(self) -> dict[str, str]:
        if self.dialect == "postgresql":
            return {"sslmode": self.sslmode}
        if self.dialect == "mysql":
            # asyncmy negotiates latin1_swedish_ci by default, which is
            # case-insensitive — a comparison under the connection's collation
            # then folds `west`/`West`/`WEST` together.
            return {"charset": "utf8mb4"}
        return {}

    def url(self) -> URL:
        """What create_async_engine is handed. Carries the password."""
        return self._url(password=self.password)

    def conninfo(self) -> str:
        """A libpq URL, for dialing this address with psycopg rather than
        SQLAlchemy. Postgres only.
        """
        assert self.dialect == "postgresql", (
            f"{self.id!r} is {self.driver} — psycopg cannot dial it."
        )
        return self._url(password=self.password).set(
            drivername="postgresql"
        ).render_as_string(hide_password=False)

    def safe_dsn(self) -> str:
        """The address, renderable. Never carries the password.

        Built from a URL with no password rather than masking a real one, which
        is one flipped keyword from leaking. The query string is dropped too.
        """
        return self._url(password=None, query=False).render_as_string(
            hide_password=False
        )


def connection_from_url(url: str, *, id: str, origin: str = "api") -> Connection:
    """Parse a URL into a registry row. URLs only — libpq keyword form
    (`host=x dbname=y`) fails here rather than being reparsed.
    """
    parsed = make_url(url)
    driver = normalise_driver(parsed.drivername)
    dialect = driver.split("+", 1)[0]
    return Connection(
        id=id,
        origin=origin,
        driver=driver,
        host=parsed.host or (None if dialect == "sqlite" else "localhost"),
        port=parsed.port or _DEFAULT_PORT.get(dialect),
        database=parsed.database,
        username=parsed.username,
        password=parsed.password,
        sslmode=parsed.query.get("sslmode") or "prefer",
    )


async def get_connection(conn: AsyncConnection, connection_id: str) -> Connection | None:
    cur = await conn.execute(
        f"SELECT {_CONNECTION_COLUMNS} FROM connection WHERE id = %s",
        (connection_id,),
    )
    row = await cur.fetchone()
    return Connection.from_row(row) if row else None


async def list_connections(conn: AsyncConnection) -> list[Connection]:
    cur = await conn.execute(
        f"SELECT {_CONNECTION_COLUMNS} FROM connection ORDER BY id"
    )
    return [Connection.from_row(r) for r in await cur.fetchall()]


async def connection_stats(conn: AsyncConnection) -> dict[str, dict[str, int]]:
    """Cache and turn counts per connection, in one query."""
    cur = await conn.execute(
        """
        SELECT c.id,
               (SELECT count(*) FROM cache_entry e WHERE e.connection_id = c.id)
                   AS cache_entries,
               (SELECT count(*) FROM turn t WHERE t.connection_id = c.id)
                   AS turns
        FROM connection c
        """
    )
    return {
        r["id"]: {"cache_entries": r["cache_entries"], "turns": r["turns"]}
        for r in await cur.fetchall()
    }


async def create_connection(conn: AsyncConnection, row: Connection) -> Connection:
    """Register a database. Raises `psycopg.errors.UniqueViolation` on a reused id."""
    cur = await conn.execute(
        f"""
        INSERT INTO connection (
            id, label, origin, driver, host, port, database, username,
            password, sslmode, options
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING {_CONNECTION_COLUMNS}
        """,
        (
            row.id,
            row.label,
            row.origin,
            row.driver,
            row.host,
            row.port,
            row.database,
            row.username,
            seal(row.password),
            row.sslmode,
            Jsonb(row.options or {}),
        ),
    )
    written = await cur.fetchone()
    assert written is not None
    return Connection.from_row(written)


async def update_connection(
    conn: AsyncConnection, connection_id: str, **fields: Any
) -> Connection | None:
    """Change the named fields and nothing else. None if there is no such row.

    An absent field is never confused with one set to NULL. `password` is sealed.
    """
    # No `driver` — see the 409 in app/api.py.
    allowed = ("label", "host", "port", "database", "username", "password",
               "sslmode", "options")
    unknown = set(fields) - set(allowed)
    assert not unknown, f"not a connection field: {sorted(unknown)}"
    if not fields:
        return await get_connection(conn, connection_id)

    if "password" in fields:
        fields = {**fields, "password": seal(fields["password"])}
    assignments = sql.SQL(", ").join(
        sql.SQL("{} = {}").format(sql.Identifier(k), sql.Placeholder()) for k in fields
    )
    cur = await conn.execute(
        sql.SQL(
            "UPDATE connection SET {}, updated_at = now() WHERE id = {} "
            "RETURNING " + _CONNECTION_COLUMNS
        ).format(assignments, sql.Placeholder()),
        (*fields.values(), connection_id),
    )
    row = await cur.fetchone()
    return Connection.from_row(row) if row else None


async def delete_connection(conn: AsyncConnection, connection_id: str) -> dict[str, int]:
    """Remove a connection and everything learned about it.

    Counted deletes rather than the foreign keys' cascade, so the caller can
    report what it destroyed.
    """
    wiped = await reset_learned(conn, connection_id=connection_id)
    await conn.execute("DELETE FROM connection WHERE id = %s", (connection_id,))
    return wiped


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
    come from the dialect's own reflection, so a fingerprint is comparable only
    within one connection.
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


async def load_cache(
    conn: AsyncConnection, *, connection_id: str
) -> list[CacheEntry]:
    """One connection's entries, not disabled, ordered by hits.

    All of it, every turn — it fits in context, and retrieval would only add a
    way to miss the entry you needed (§4). Tombstones included: a visible
    negative constraint stops exploration rediscovering the same wrong thing (§5).
    """
    cur = await conn.execute(
        f"SELECT {_COLUMNS} FROM cache_entry "
        "WHERE connection_id = %s AND NOT disabled ORDER BY hits DESC, id ASC",
        (connection_id,),
    )
    return [CacheEntry.from_row(r) for r in await cur.fetchall()]


async def count_disabled(conn: AsyncConnection, *, connection_id: str) -> int:
    """How many of this connection's entries `load_cache` filtered out."""
    cur = await conn.execute(
        "SELECT count(*) AS n FROM cache_entry "
        "WHERE connection_id = %s AND disabled",
        (connection_id,),
    )
    row = await cur.fetchone()
    return row["n"] if row else 0


async def stale_ids(
    conn: TargetConnection, entries: Sequence[CacheEntry]
) -> set[int]:
    """Which entries were learned against a schema that has since moved?

    **Takes a target connection**, and it must be *this entry's* connection's.
    An entry with no fingerprint or no tables is never reported stale, so
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
    connection_id: str,
    turn_id: int | None = None,
) -> list[int]:
    """Insert or refresh learned entries. Returns the ids actually written.

    **Takes an agent connection**, and does not compute fingerprints — that is
    `fingerprint_entries`, which needs the target.

    Named entries upsert **within a connection**, so `revenue` on another
    warehouse stays a separate entry. **A human's pinned entry is never
    overwritten**: it is skipped, and its id is absent from the return value.
    """
    written: list[int] = []
    for e in entries:
        cur = await conn.execute(
            """
            INSERT INTO cache_entry (
                connection_id, kind, name, claim, sql_fragment, tables, origin,
                pinned, disabled, tombstone, verified, schema_fp,
                created_turn, last_used_turn
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            -- Postgres infers cache_entry_conn_name_key from the columns plus
            -- the matching predicate.
            ON CONFLICT (connection_id, name) WHERE name IS NOT NULL
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
                connection_id,
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
    connection_id: str,
    turn_id: int | None = None,
) -> None:
    """Mark the entries a turn actually used.

    `hits` orders the cache; `last_used_turn` is what lets compaction drop
    entries nothing has needed. The `connection_id` clause guards the resumed
    path, where a checkpointed TurnState carries another connection's entry ids.
    """
    if not ids:
        return
    await conn.execute(
        """
        UPDATE cache_entry
        SET hits = hits + 1,
            last_used_turn = COALESCE(%s, last_used_turn),
            updated_at = now()
        WHERE id = ANY(%s) AND connection_id = %s
        """,
        (turn_id, list(ids), connection_id),
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


async def reset_learned(
    conn: AsyncConnection, *, connection_id: str
) -> dict[str, int]:
    """Forget what the agent learned about *one* connection. Rows-per-table.

    The stage recovery button (PLAN.md §9). Cache, turn log and LangGraph's
    checkpoints go together, because a checkpointed TurnState holds a `turn_id`
    and cache-entry ids that would otherwise dangle.

    `turn` is the only mapping from a connection to LangGraph's `thread_id`, so
    **the order below is load-bearing** — those rows go last.
    `checkpoint_migrations` is excluded: it is a schema version, not turn state.
    """
    threads = "SELECT DISTINCT session_id::text FROM turn WHERE connection_id = %s"
    wiped = {}
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
        cur = await conn.execute(
            sql.SQL("DELETE FROM {} WHERE thread_id IN (" + threads + ")").format(
                sql.Identifier(table)
            ),
            (connection_id,),
        )
        wiped[table] = cur.rowcount
    for table in ("turn", "cache_entry"):
        cur = await conn.execute(
            sql.SQL("DELETE FROM {} WHERE connection_id = %s").format(
                sql.Identifier(table)
            ),
            (connection_id,),
        )
        wiped[table] = cur.rowcount
    # Ordered as a reader expects to see it, not as the deletes had to run.
    return {k: wiped[k] for k in ("cache_entry", "turn", "checkpoints",
                                  "checkpoint_blobs", "checkpoint_writes")}


async def reset_everything(conn: AsyncConnection) -> dict[str, int]:
    """Empty the agent's database. Every connection, every turn, every thread.

    **Not exposed on the API** — this is for the test suite and a `make` target.
    `connection` is excluded: the registry is configuration, not learned state.

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
    conn: AsyncConnection, *, connection_id: str, session_id: str | UUID, question: str
) -> int:
    """Open the turn row and return its id. Split from `finish_turn` because
    `extract` writes entries mid-turn that need a `created_turn` to point at.
    """
    cur = await conn.execute(
        "INSERT INTO turn (connection_id, session_id, question) "
        "VALUES (%s, %s, %s) RETURNING id",
        (connection_id, str(session_id), question),
    )
    row = await cur.fetchone()
    assert row is not None
    return row["id"]


async def fail_open_turn(
    conn: AsyncConnection,
    session_id: str | UUID,
    message: str,
    *,
    connection_id: str,
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
            WHERE session_id = %s AND connection_id = %s AND answer IS NULL
            ORDER BY id DESC LIMIT 1
        )
        RETURNING id
        """,
        (message, str(session_id), connection_id),
    )
    row = await cur.fetchone()
    return row["id"] if row else None


async def session_connection(
    conn: AsyncConnection, session_id: str | UUID
) -> str | None:
    """Which connection this session has been asking about, if any.

    How the caller's 409 finds out: a thread's history is checkpointed, so
    reusing a session id would hand the model another warehouse's conversation.
    """
    cur = await conn.execute(
        "SELECT connection_id FROM turn WHERE session_id = %s "
        "ORDER BY id DESC LIMIT 1",
        (str(session_id),),
    )
    row = await cur.fetchone()
    return row["connection_id"] if row else None


async def read_turns(
    conn: AsyncConnection,
    *,
    connection_id: str,
    limit: int = 50,
    finished: bool = True,
) -> list[dict[str, Any]]:
    """The demo chart, as rows: what each turn asked and what it cost.

    Queried newest-first with a LIMIT so a long-lived connection paginates, and
    returned **ascending** so it reads left to right. `finished=False` also
    shows the turns still in flight and the ones that failed.
    """
    cur = await conn.execute(
        f"""
        SELECT id, question, sql, answer, tool_calls, explored,
               tokens_in, tokens_out, latency_ms, cache_entries, created_at,
               trace_id
        FROM turn
        WHERE connection_id = %s {"AND answer IS NOT NULL" if finished else ""}
        ORDER BY id DESC
        LIMIT %s
        """,
        (connection_id, limit),
    )
    return list(reversed(await cur.fetchall()))


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
) -> None:
    """Record what the turn cost. This is the demo chart."""
    await conn.execute(
        """
        UPDATE turn SET
            sql = %s, answer = %s, tool_calls = %s, explored = %s,
            tokens_in = %s, tokens_out = %s, latency_ms = %s, cache_entries = %s,
            trace_id = %s
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
            turn_id,
        ),
    )
