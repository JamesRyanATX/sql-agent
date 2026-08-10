"""Reads and writes for the connection registry, the cache and the turn log.

The cache is the product (PLAN.md §6.2), so this layer holds the two rules that
protect it: a human's correction is never silently overwritten, and every entry
records a fingerprint of the schema it was learned against.

**Two servers, and the functions here are not interchangeable about which** —
and now they are not even the same *type*. Everything takes a
`psycopg.AsyncConnection` against the agent's own database except
`reflect_columns`, `schema_fingerprint`, `fingerprint_entries` and `stale_ids`,
which take a **SQLAlchemy** connection to a target. An entry is agent state
describing target shape, and those are the only places the two meet. Handing one
kind where the other belongs used to be a quiet bug; it is now an
`AttributeError` on the first line.

**And now N targets.** Everything that reads or writes what the agent has
learned is scoped to one `connection_id`, and **none of these functions has a
default for it**. A default is how one warehouse's cache ends up answering
another warehouse's question, and nothing about that failure is loud: the
answer looks right, the SQL looks right, and the numbers come from the wrong
database. The three fingerprint functions keep their signatures, but their
caller now owes them a target connection belonging to *the entry's own*
connection.
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
# A registered database the agent can be pointed at. Note the word: `app/db.py`
# uses "connection" ~20 times to mean a psycopg connection, and these are not
# that. Never bind a bare `connection` variable to one of these rows — it is
# `connection_id: str` or `registered: store.Connection`, always.


# The drivers a connection may name. Async-capable only: a sync driver would
# need every target call to cross into a thread pool, and a thread pool under an
# asyncio graph is how a demo hangs on stage. Mirrored by a CHECK in
# migrations/003 — a row edited by hand at psql never sees a validator.
DRIVERS = ("postgresql+psycopg", "mysql+asyncmy", "sqlite+aiosqlite")

# What a user may reasonably type, and what it means. `postgresql://` on its own
# resolves to psycopg2 in SQLAlchemy, which is not installed and not wanted.
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
    # **Plaintext**, unsealed on read. It exists to be handed to a driver and
    # nothing else; the wire model in app/schemas.py has no password field at
    # all, so this can never be leaked by serialising the wrong object.
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
        the SQL. The other half only changes who dials the socket, and nothing
        above app/db.py should care which."""
        return self.driver.split("+", 1)[0]

    def _url(self, *, password: str | None, query: bool = True) -> URL:
        """The address as SQLAlchemy sees it.

        `URL.create`, never an f-string — the same reason this was never an
        f-string when it built a libpq DSN: a password containing `@`, `/` or
        `%` produces a wrong-and-confusing URL under string formatting, and the
        resulting authentication failure points at the password rather than at
        the quoting. `URL.create` takes the parts as objects and percent-encodes
        on render, so there is nothing to get wrong.
        """
        if self.dialect == "sqlite":
            # host/port/username are NULL for sqlite by construction (the CHECK
            # in 003). Passing them anyway renders a URL whose authority section
            # aiosqlite quietly treats as part of the path.
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
            # asyncmy negotiates latin1_swedish_ci by default, and that is not
            # cosmetic: a CAST or a comparison then runs under the *connection's*
            # collation rather than the column's, which is case-insensitive and
            # folds `west`/`West`/`WEST` together. utf8mb4 makes the connection
            # agree with any server built this decade.
            return {"charset": "utf8mb4"}
        return {}

    def url(self) -> URL:
        """What create_async_engine is handed. Carries the password."""
        return self._url(password=self.password)

    def conninfo(self) -> str:
        """A libpq URL, for the psycopg call sites that have not moved yet.

        Temporary: `app/db.py`'s target pools and `app/api.py::_probe` still
        dial with psycopg, and they are replaced together in the phase that
        makes `db.target()` return a SQLAlchemy connection. Deleted with them.
        """
        assert self.dialect == "postgresql", (
            f"{self.id!r} is {self.driver} — psycopg cannot dial it. "
            "This path exists only until db.target() moves to SQLAlchemy."
        )
        return self._url(password=self.password).set(
            drivername="postgresql"
        ).render_as_string(hide_password=False)

    def safe_dsn(self) -> str:
        """The address, renderable. Never carries the password.

        Rendered from a URL built *without* one rather than from a real URL with
        `hide_password=True`. SQLAlchemy 2.0's masking is sound — `__repr__` and
        `__str__` both hide it — but masking is the weaker promise: it is one
        flipped keyword from leaking, and this string goes to the API, the CLI
        and error messages. A password that was never put in the object cannot
        come out of it.

        The query string is dropped too — `sslmode` has its own line in
        `sql-agent connections get`, and an address a human reads should not
        repeat it.

        What `GET /v1/connections` shows.
        """
        return self._url(password=None, query=False).render_as_string(
            hide_password=False
        )


def connection_from_url(url: str, *, id: str, origin: str = "api") -> Connection:
    """Parse a URL into a registry row.

    Used for the env-owned `default` row, whose address is TARGET_DATABASE_URL,
    and by the test harness — which has to agree with the lifespan about how a
    URL becomes a row, or the two disagree in exactly the way that is hardest
    to see.

    Note this reads URLs only. It used to go through `psycopg.conninfo`, which
    also accepted libpq keyword form (`host=x dbname=y`); a TARGET_DATABASE_URL
    in that spelling now fails at startup rather than being silently reparsed.
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
    """Cache and turn counts per connection, in one query.

    Kept off `Connection` deliberately: the dataclass is the registry row, and
    these are facts about other tables that happen to point at it.
    """
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

    Partial by construction: a caller that passes no fields gets the row back
    untouched, and a field absent from `fields` is never confused with a field
    set to NULL. `password` is sealed on the way in.
    """
    # `driver` is deliberately absent — see the 409 in app/api.py. A cached
    # recipe is SQL in a dialect, so changing the driver under one silently
    # invalidates every entry, and schema_fp would not catch it.
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

    The counted deletes run here rather than being left to the foreign keys'
    `ON DELETE CASCADE`, so the caller can report what it destroyed. The cascade
    stays as a backstop for a `DELETE FROM connection` typed at psql, where
    there is nobody to report to.
    """
    wiped = await reset_learned(conn, connection_id=connection_id)
    await conn.execute("DELETE FROM connection WHERE id = %s", (connection_id,))
    return wiped


# ---------------------------------------------------------------- fingerprint


async def reflect_columns(
    conn: TargetConnection, tables: Sequence[str]
) -> dict[str, Any]:
    """One reflection pass over `tables`. **Takes a target connection.**

    Batched deliberately. This used to be one `information_schema` query per
    entry, which was merely wasteful; through reflection it would be one whole
    table reflection per entry, on somebody else's warehouse, per page load of
    `GET /connections/{id}/cache`.
    """
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
        # compare equal to the same table on MySQL. The PATCH that refuses a
        # driver change is what actually closes that door; this catches the case
        # it cannot see — the address stayed the same and the database behind it
        # did not.
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

    Stored on an entry at write time; recomputed on load. A mismatch means the
    schema moved under a recipe that was learned against the old shape, so the
    entry can no longer be trusted (§5).

    Catches renames, drops, additions and type changes. Does not catch a column
    keeping its name and changing its meaning — that one is in §10 for a reason.

    The type strings come from the dialect's own reflection, so a fingerprint is
    comparable only within one connection. That was always true and is now
    visibly so; `migrations/003` NULLs every fingerprint written before the port
    for exactly this reason.
    """
    return fingerprint(await reflect_columns(conn, tables), tables)


async def fingerprint_entries(
    conn: TargetConnection, entries: Sequence[CacheEntry]
) -> None:
    """Stamp each entry with the shape of the tables it describes, in place.

    **Takes a target connection**, and has to be called before `write_entries`,
    which takes an agent one. Splitting them is what the two servers force: the
    fingerprint is a fact about the business schema, the entry it lands on is
    the agent's own memory, and no single connection can reach both.

    An entry naming no tables gets no fingerprint. That is honest — there is
    nothing to hash — and `stale_ids` reads a missing fingerprint as "unknown"
    rather than "fine".
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
    way to miss the entry you needed (§4). "All of it" now means all of *this
    connection's*: what the agent knows about one warehouse is not evidence
    about another, and handing the model both is how a recipe crosses over.

    Tombstones are included deliberately. A tombstone *is* the useful content:
    a visible negative constraint that stops exploration rediscovering the same
    wrong thing next session (§5).
    """
    cur = await conn.execute(
        f"SELECT {_COLUMNS} FROM cache_entry "
        "WHERE connection_id = %s AND NOT disabled ORDER BY hits DESC, id ASC",
        (connection_id,),
    )
    return [CacheEntry.from_row(r) for r in await cur.fetchall()]


async def count_disabled(conn: AsyncConnection, *, connection_id: str) -> int:
    """How many of this connection's entries `load_cache` filtered out.

    The one number a caller cannot derive from the entries themselves, and the
    cache listing shows it so a disabled entry never goes quietly missing.
    """
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

    **Takes a target connection**, and it must be *this entry's* connection's
    target — asking a different one reports every entry stale, or worse,
    coincidentally not stale.

    An entry with no fingerprint or no tables can't be checked, so it is never
    reported stale — silence here means "unknown", not "fine", which is why
    `infer_tables` in the graph works so hard to keep `tables` populated.

    Reporting only, for now. §5's invalidation is phase 6 and will use this.
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
    `fingerprint_entries`, which needs the target. This function used to fall
    back to hashing on its own connection, which quietly became wrong the day
    the two databases split: it would have fingerprinted the *agent* schema and
    stamped the answer onto an entry describing business tables, permanently
    stale from the moment it was written.

    Named entries upsert **within a connection**, so re-learning `revenue`
    refines one row instead of accumulating near-duplicates for compaction to
    clean up later — while `revenue` on another warehouse stays a separate
    entry, because it is a separate fact.

    **A human's pinned entry is never overwritten.** That is the whole point of
    the admin surface: correcting a recipe fixes every future question that
    composes it (§6.2), and an extraction quietly reverting that correction on
    the next turn is the worst bug this system can have. Such an entry is
    skipped and its id is absent from the return value.
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

    `hits` orders the cache and shows blast radius in the admin list;
    `last_used_turn` is what lets compaction drop entries nothing has needed.

    The `connection_id` clause is belt and braces — the ids come from an
    already-scoped `load_cache`. It earns its place on the resumed-checkpoint
    path: a checkpointed TurnState carries entry ids, and a thread replayed
    against another connection would otherwise silently credit that
    connection's entries to this one's turn.
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

    The stage recovery button (PLAN.md §9), and the only destructive operation
    the API exposes. It takes the cache, the turn log *and* LangGraph's
    checkpoints together: an empty cache beside a turn log that says the
    questions were already asked is a state nothing knows how to read.

    The checkpoint tables are LangGraph's and are keyed by `thread_id` — the
    session UUID — with no idea a connection exists. `turn` records both, so it
    is the mapping, and **the order below is load-bearing**: the turn rows are
    what the mapping is made of, so they go last.

    Leaving the checkpoints alone would be wrong rather than untidy. A
    checkpointed TurnState holds a `turn_id` and a list of cache-entry ids;
    after this runs, both dangle, and the next question on that thread resumes
    into a state describing rows that no longer exist.

    `checkpoint_migrations` is deliberately absent — it is LangGraph's schema
    version table, not turn state.

    The business data is on another server and this connection cannot reach it.
    That used to be a promise kept by a list of table names; now it is a promise
    kept by the network.
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

    **Not exposed on the API** — the destructive operation a user gets is scoped
    to a connection they named. This exists for the test suite and for a `make`
    target, where "start over" means all of it.

    `connection` is deliberately excluded: the registry is configuration, not
    learned state, and wiping it would delete the address of every warehouse
    somebody registered. It also cannot be truncated alongside these anyway —
    the foreign keys cascade the other way.

    Written out rather than driven from a list: these six tables *are* the
    agent's database. The cost is that a `langgraph-checkpoint-postgres` release
    adding a seventh has to be added here by hand — and the failure mode is
    quiet, `make reset-all` no longer resetting, so it is worth checking after
    an upgrade.
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
    """Open the turn row and return its id.

    Split from finish_turn because entries written by `extract` mid-turn need a
    `created_turn` to point at, and the turn's results aren't known until it
    ends.
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

    A turn row is opened before any model call, so anything that throws between
    then and `answer` would otherwise leave it open forever — invisible in
    /stats, and indistinguishable from a turn still in flight.
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

    A thread's history is checkpointed, so reusing a session id against a second
    warehouse would hand the model warehouse A's conversation while it answers
    about B. The caller refuses that; this is how it finds out.
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
    returned **ascending** so it reads left-to-right like the chart. The
    reversal is deliberate — this endpoint exists to replace `make turns`, which
    ordered by id.

    `finished` defaults to true for the same reason: `make turns` filtered
    `answer IS NOT NULL`, so a turn still in flight never appeared. Passing
    false shows those, and the failed ones, which the old query could not.
    """
    cur = await conn.execute(
        f"""
        SELECT id, question, sql, answer, tool_calls, explored,
               tokens_in, tokens_out, latency_ms, cache_entries, created_at
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
