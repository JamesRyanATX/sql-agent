"""One pool for the agent's memory, and one engine per registered target.

    db.agent()        -> psycopg.AsyncConnection
    db.target(cid)    -> sqlalchemy.ext.asyncio.AsyncConnection

The agent's memory is always Postgres via psycopg; a target is SQLAlchemy and
may be Postgres, MySQL or SQLite. The two types share no method that matters, so
passing one where the other belongs raises rather than querying the wrong server.

Read-only is enforced per dialect — see `app/dialects.py`.
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

# `store` for the registry lookup. Acyclic: store never imports app.db.
from app import dialects, store
from app.config import config
from app.settings import settings

_agent_pool: AsyncConnectionPool | None = None
_target_engines: dict[str, AsyncEngine] = {}
_registry_lock = asyncio.Lock()

DEFAULT_CONNECTION = "default"


class UnknownConnection(LookupError):
    """No connection is registered under that id."""


class TargetUnreachable(RuntimeError):
    """Registered, but we could not open a pool against it."""

    def __init__(self, connection_id: str, cause: str) -> None:
        super().__init__(f"cannot reach connection {connection_id!r} ({cause})")
        self.connection_id = connection_id


def cause_of(e: BaseException) -> str:
    """The exception type, never its message — a driver quotes the DSN it failed on.

    Unwrapped through `.orig` first: SQLAlchemy's own type name is
    `OperationalError` for everything.
    """
    orig = getattr(e, "orig", None)
    return type(orig if orig is not None else e).__name__


def _make_pool(url: str, **kw) -> AsyncConnectionPool:
    """The agent pool. Targets are SQLAlchemy engines — see `target_engine`."""
    return AsyncConnectionPool(
        url,
        open=False,
        # What AsyncPostgresSaver requires of a pool it is handed.
        kwargs={
            "row_factory": dict_row,
            "autocommit": True,
            "prepare_threshold": 0,
        },
        **kw,
    )


async def open_pools() -> AsyncConnectionPool:
    """Open the agent pool. Called once on app startup.

    Target engines are built on first use instead, so one unreachable warehouse
    is not a failure to boot.
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
    """The registry row. A row with `origin = 'env'` gets its address from
    `TARGET_DATABASE_URL`, which the psql-applied migration cannot write down.
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
    """Copy TARGET_DATABASE_URL's address onto the `default` row, so a listing
    reads true. The password is not copied — it stays in the environment.
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
    """Pool sizing: one socket at rest, up to `target_pool_max` in a burst.

    QueuePool closes a connection returned above `pool_size` rather than keeping
    it, so overflow is what shrinks back. `pool_size=0` would mean unlimited.
    """
    s = settings()
    if dialect == "sqlite":
        # No socket to size, and SQLAlchemy swaps the pool class for :memory:,
        # where passing pool_size is a TypeError at construction.
        return {"connect_args": {"timeout": s.target_connect_timeout}}
    return {
        "pool_size": 1,
        "max_overflow": max(0, s.target_pool_max - 1),
        # pool_pre_ping turns a warehouse restart into a reconnect rather than a
        # mid-turn error.
        "pool_recycle": int(s.target_pool_max_idle),
        "pool_pre_ping": True,
    }


def _guard_sqlite_path(registered: store.Connection) -> None:
    """Refuse a SQLite path that does not exist. sqlite3 would create the file,
    so a typo registers green and hands the agent an empty database.
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
            # `explore` holds one connection across up to 24 tool calls with
            # model round trips between them, and SQLAlchemy would otherwise
            # open a transaction on the first statement — holding a snapshot on
            # a customer's database for minutes. `target_readonly` opts back in.
            isolation_level="AUTOCOMMIT",
            **_engine_kwargs(registered.dialect),
        )
        dialects.install(engine, registered.dialect, config().statement_timeout_ms)
        try:
            # `create_async_engine` is a factory and never dials, so it succeeds
            # against a host that is not listening. Without this the failure
            # surfaces as an error event inside a 200 SSE response, not a 502.
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

    `dispose()` leaves checked-out connections running, so a PATCH landing
    mid-turn is last-writer-wins for that turn.
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

    Read-only because `dialects.install` set the session that way, and AUTOCOMMIT
    so the explore loop holds no transaction open across model round trips.
    """
    engine = await target_engine(connection_id)
    async with engine.connect() as conn:
        yield conn


@asynccontextmanager
async def target_readonly(connection_id: str) -> AsyncIterator[TargetConnection]:
    """The transaction the agent's generated SQL runs inside.

    The only place that wants a real transaction. What runs inside it is
    per-dialect and may be empty — see `app/dialects.py`.
    """
    engine = await target_engine(connection_id)
    dialect = engine.sync_engine.dialect
    cap = dialects.for_dialect(dialect.name)
    async with engine.connect() as conn:
        # Back out of the engine-level AUTOCOMMIT to this dialect's own default.
        # Asking the dialect, because SQLite rejects READ COMMITTED outright.
        conn = await conn.execution_options(
            isolation_level=dialect.default_isolation_level
        )
        async with conn.begin():
            for statement in cap.transaction(dialect, config().statement_timeout_ms):
                await conn.exec_driver_sql(statement)
            yield conn
