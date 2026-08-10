"""One pool for the agent's memory, and one engine per registered target.

The agent's memory (`cache_entry`, `turn`, the checkpoints) and the data it
answers questions about live on separate servers. They can't see each other
because they aren't in the same place — not because application code remembers
to filter.

**The two halves speak different libraries, and that is the point.**

    db.agent()        -> psycopg.AsyncConnection
    db.target(cid)    -> sqlalchemy.ext.asyncio.AsyncConnection

They share no method that matters: `execute()` returns an `AsyncCursor` on one
and a `CursorResult` on the other, and `fetchone()` is awaitable on one and not
the other. Handing an agent connection to `store.schema_fingerprint` used to be
a quiet bug — a fingerprint of the wrong schema, stamped onto an entry that
would be permanently stale. It is now an `AttributeError` on the first line.

The agent's own database is psycopg and Postgres and stays that way: LangGraph's
`AsyncPostgresSaver` uses psycopg3 pipeline mode and has no SQLAlchemy seam, and
`migrations/` leans on `text[]`, partial unique indexes and a conditional
`ON CONFLICT` that implements the pinned-entry rule. Targets are SQLAlchemy
because targets are somebody else's database and may not be Postgres at all.

**The read-only guarantee moved twice.** It used to rest on the credentials —
`demo/demo.sql` builds a role holding SELECT and nothing else. Registered
connections made that false, so it moved into a session-level setting applied on
every connection. Supporting three dialects made *that* only partly true, so it
is now a capability: see `app/dialects.py` for what each engine can actually
promise, and `ConnectionTestOut.warnings` for how a user is told.
"""

import asyncio
import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncConnection as TargetConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# `store` for the registry lookup. This is a layering inversion — db is the
# lower layer — but it is acyclic (store imports psycopg and app.secrets, never
# app.db) and checked by the import. An injected resolver would buy nothing but
# a place for the lookup to hide.
from app import dialects, store
from app.settings import settings

_agent_pool: AsyncConnectionPool | None = None
_target_engines: dict[str, AsyncEngine] = {}
_registry_lock = asyncio.Lock()

DEFAULT_CONNECTION = "default"


class UnknownConnection(LookupError):
    """No connection is registered under that id."""


class TargetUnreachable(RuntimeError):
    """Registered, but we could not open a pool against it.

    Carries the exception *type* and not its message: psycopg's connection
    errors quote the conninfo they failed on, and that is the one string in this
    module that must never reach a caller.
    """

    def __init__(self, connection_id: str, cause: str) -> None:
        super().__init__(f"cannot reach connection {connection_id!r} ({cause})")
        self.connection_id = connection_id


def cause_of(e: BaseException) -> str:
    """The exception type, and only the type.

    Two layers now quote things they should not. A driver's connection error
    quotes the address it failed on, which carries the password — that is why
    this exists. SQLAlchemy adds a second: it wraps the driver's exception, so
    the outer `type(e).__name__` is `OperationalError` for everything and says
    nothing, while `str()` on a `StatementError` appends the whole SQL and a
    https://sqlalche.me/e/ link. Unwrap for the name, stop before the message.
    """
    orig = getattr(e, "orig", None)
    return type(orig if orig is not None else e).__name__


def _make_pool(url: str, **kw) -> AsyncConnectionPool:
    """The agent pool. Targets are SQLAlchemy engines — see `target_engine`."""
    return AsyncConnectionPool(
        url,
        open=False,
        # autocommit + dict_row + prepare_threshold=0 are what
        # AsyncPostgresSaver requires of a pool it's handed.
        kwargs={
            "row_factory": dict_row,
            "autocommit": True,
            "prepare_threshold": 0,
        },
        **kw,
    )


async def open_pools() -> AsyncConnectionPool:
    """Open the agent pool. Called once on app startup.

    Target engines are built on first use, not here: at startup we do not yet
    know which of the registered warehouses anyone will ask about, and dialing
    all of them would make a customer's database being down into a failure to
    boot.
    """
    global _agent_pool
    if _agent_pool is None:
        _agent_pool = _make_pool(settings().agent_database_url, min_size=1, max_size=10)
        await _agent_pool.open(wait=True, timeout=10)
    return _agent_pool


async def close_pools() -> None:
    global _agent_pool
    engines = list(_target_engines.values())
    pool, _agent_pool = _agent_pool, None
    _target_engines.clear()
    if pool is not None:
        await pool.close()
    for engine in engines:
        await engine.dispose()


def agent_pool() -> AsyncConnectionPool:
    if _agent_pool is None:
        raise RuntimeError("pools not open — call open_pools() during startup")
    return _agent_pool


async def resolve(connection_id: str) -> store.Connection:
    """The registry row, with the env-owned one's address filled in.

    A row with `origin = 'env'` is the connection the agent had before it had a
    name, and its address is `TARGET_DATABASE_URL`. The migration cannot write
    that address down — it is applied by psql, which cannot see the environment
    — so it is substituted here, and `ensure_default_connection()` writes a copy
    back so a listing reads true.
    """
    async with agent() as conn:
        row = await store.get_connection(conn, connection_id)
    if row is None:
        raise UnknownConnection(connection_id)
    if row.origin == "env":
        env = store.connection_from_url(
            settings().target_database_url, id=row.id, origin="env"
        )
        env.label = row.label
        return env
    return row


async def ensure_default_connection() -> None:
    """Copy TARGET_DATABASE_URL's address onto the `default` row.

    Best-effort, and called from the lifespan. `make up` starts the API before
    `make migrate` has ever run, so on a fresh clone the table does not exist
    yet; that is a warning, not a failed boot. The password is *not* copied —
    the env row's credentials come from the environment every time it resolves,
    and writing them into agent-db would create a second place they live.
    """
    env = store.connection_from_url(
        settings().target_database_url, id=DEFAULT_CONNECTION, origin="env"
    )
    async with agent() as conn:
        await store.update_connection(
            conn,
            DEFAULT_CONNECTION,
            host=env.host,
            port=env.port,
            database=env.database,
            username=env.username,
            sslmode=env.sslmode,
        )


def _engine_kwargs(dialect: str) -> dict[str, Any]:
    """Pool sizing, translated rather than copied.

    The psycopg pool was `min_size=0, max_idle=300` — hold nothing at rest, hold
    up to five during a burst, let them go after five minutes. **SQLAlchemy's
    QueuePool has no `max_idle`**: `pool_recycle` only replaces a connection on
    checkout if it is too old, and never proactively closes an idle one. So the
    obvious mapping (`pool_size=5, pool_recycle=300`) silently inverts the
    invariant, and ten registered warehouses each asked one question hold fifty
    idle sockets open on ten customers' databases forever.

    `pool_size=1` plus overflow is the closest QueuePool has: a connection
    returned when the pool is already at `pool_size` is **closed**, not kept, so
    a burst opens up to five and drops back to one. `pool_size=0` would mean
    *unlimited*, which is the opposite of what it looks like.
    """
    s = settings()
    if dialect == "sqlite":
        # A file database is one process's file. Pool sizing is a socket concept
        # and there is no socket — and SQLAlchemy swaps the pool class for
        # :memory:, where passing pool_size is a TypeError at construction.
        return {"connect_args": {"timeout": s.target_connect_timeout}}
    return {
        "pool_size": 1,
        "max_overflow": max(0, s.target_pool_max - 1),
        # The last socket does survive at rest. pool_recycle bounds how stale it
        # can get, and pool_pre_ping is what turns a warehouse restart into a
        # reconnect rather than a mid-turn error.
        "pool_recycle": int(s.target_pool_max_idle),
        "pool_pre_ping": True,
    }


def _guard_sqlite_path(registered: store.Connection) -> None:
    """A typo'd SQLite path is not an error — sqlite3 creates the file.

    Registering `/data/shop.dp` would succeed, probe green, and hand the agent
    an empty database it would explore, learn nothing from, and cache the
    nothing. Refuse a path that does not already exist; we are a reader.
    """
    if registered.dialect != "sqlite":
        return
    if not registered.database or not pathlib.Path(registered.database).is_file():
        raise TargetUnreachable(registered.id, "FileNotFoundError")


async def target_engine(connection_id: str) -> AsyncEngine:
    """The engine for one registered target, built on first use.

    Double-checked under a lock so a burst of concurrent turns against a cold
    connection builds one engine between them rather than one each.
    """
    engine = _target_engines.get(connection_id)
    if engine is not None:
        return engine

    async with _registry_lock:
        if (engine := _target_engines.get(connection_id)) is not None:
            return engine
        registered = await resolve(connection_id)
        _guard_sqlite_path(registered)
        engine = create_async_engine(
            registered.url(),
            # AUTOCOMMIT, matching the psycopg pool's autocommit=True. Not a
            # style choice: `explore` holds one connection across up to 24 tool
            # calls with model round trips between them, and a SQLAlchemy
            # connection begins a transaction implicitly on its first
            # statement. Without this a T1 turn holds a snapshot open on a
            # customer's production database for minutes while a model thinks —
            # pinning the xmin horizon and blocking autovacuum. Nothing fails.
            # Their DBA notices. `target_readonly` opts back in; it is the one
            # place that wants a transaction.
            isolation_level="AUTOCOMMIT",
            **_engine_kwargs(registered.dialect),
        )
        dialects.install(engine, registered.dialect, settings().statement_timeout_ms)
        try:
            # Take a connection. The reason changed with the port but the
            # conclusion did not: `create_async_engine` never dials anything —
            # it is a factory, and it succeeds against a host that is not
            # listening. Without this the failure surfaces later as an error
            # event inside a 200 SSE response instead of the 502 it is.
            async with asyncio.timeout(settings().target_connect_timeout):
                async with engine.connect():
                    pass
        except Exception as e:
            await engine.dispose()
            raise TargetUnreachable(connection_id, cause_of(e)) from e
        _target_engines[connection_id] = engine
        return engine


async def evict(connection_id: str) -> None:
    """Drop a target engine, so the next turn re-resolves its address.

    Called after a connection is updated or deleted. Honest about what it is
    not, unchanged from the pool version: `dispose()` closes what is idle and
    leaves what is checked out running, so a PATCH landing mid-turn is
    last-writer-wins for that turn. The registry is not transactional with
    respect to a running turn and cannot be, because TurnState is checkpointed
    and cannot hold an engine.
    """
    engine = _target_engines.pop(connection_id, None)
    if engine is not None:
        await engine.dispose()


@asynccontextmanager
async def agent() -> AsyncIterator[AsyncConnection]:
    """Read/write against the agent's own memory."""
    async with agent_pool().connection() as conn:
        yield conn


@asynccontextmanager
async def target(connection_id: str) -> AsyncIterator[TargetConnection]:
    """A registered target, for introspection and fingerprinting.

    Read-only because `dialects.install` put the session into that state when
    the connection was opened — not because of the credentials, which we did not
    choose, and to whatever extent the dialect can promise it.

    The engine runs AUTOCOMMIT, so this holds no transaction open across the
    explore loop's model round trips.
    """
    engine = await target_engine(connection_id)
    async with engine.connect() as conn:
        yield conn


@asynccontextmanager
async def target_readonly(connection_id: str) -> AsyncIterator[TargetConnection]:
    """The transaction the agent's generated SQL runs inside.

    The one place that wants a real transaction: the engine is AUTOCOMMIT so
    `explore` does not pin a snapshot, and a transaction-scoped setting means
    nothing without a transaction to scope it to.

    What runs here is per-dialect and may be empty — Postgres re-applies
    `transaction_read_only` and `statement_timeout` (redundant with the session
    settings, and kept because defence that exists in one place is one edit from
    not existing); MySQL and SQLite have no transaction-scoped equivalent of
    either, and that emptiness is `app/dialects.py` doing its job rather than a
    gap. See `Capability.gaps` for what a user is told about it.
    """
    engine = await target_engine(connection_id)
    dialect = engine.sync_engine.dialect
    cap = dialects.for_dialect(dialect.name)
    async with engine.connect() as conn:
        # Back out of the engine-level AUTOCOMMIT, to whatever this dialect's
        # own default is — `READ COMMITTED` on Postgres, `SERIALIZABLE` on
        # SQLite, which rejects the former outright. Asking the dialect beats
        # a per-dialect constant: there is nothing here to keep in step.
        conn = await conn.execution_options(
            isolation_level=dialect.default_isolation_level
        )
        async with conn.begin():
            for statement in cap.transaction(dialect, settings().statement_timeout_ms):
                await conn.exec_driver_sql(statement)
            yield conn
