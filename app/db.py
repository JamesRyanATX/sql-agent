"""One pool for the agent's memory, and one engine for the database it queries.

    db.agent()    -> psycopg.AsyncConnection
    db.target()   -> sqlalchemy.ext.asyncio.AsyncConnection

Two addresses, both from the environment: `AGENT_DATABASE_URL` and
`TARGET_DATABASE_URL`. The agent's memory is always Postgres via psycopg; the
target is SQLAlchemy and may be Postgres, MySQL or SQLite. The two types share no method that matters, so
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
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncConnection as TargetConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app import dialects
from app.config import config
from app.settings import settings

# What a user may reasonably write in TARGET_DATABASE_URL, mapped to the driver
# that is actually installed. `postgresql://` alone resolves to psycopg2 in
# SQLAlchemy, which is not here.
_DRIVERS = {
    "postgres": "postgresql+psycopg",
    "postgresql": "postgresql+psycopg",
    "mysql": "mysql+asyncmy",
    "mariadb": "mysql+asyncmy",
    "mariadb+asyncmy": "mysql+asyncmy",
    "sqlite": "sqlite+aiosqlite",
}

_agent_pool: AsyncConnectionPool | None = None
_target: AsyncEngine | None = None
_target_lock = asyncio.Lock()


class TargetUnreachable(RuntimeError):
    """We could not open a connection to TARGET_DATABASE_URL."""

    def __init__(self, cause: str) -> None:
        super().__init__(f"cannot reach the target database ({cause})")


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
    global _agent_pool, _target
    pool, _agent_pool = _agent_pool, None
    engine, _target = _target, None
    if pool is not None:
        await pool.close()
    if engine is not None:
        await engine.dispose()


def agent_pool() -> AsyncConnectionPool:
    if _agent_pool is None:
        raise RuntimeError("pools not open — call open_pools() during startup")
    return _agent_pool


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


def _guard_sqlite_path(url: URL) -> None:
    """Refuse a SQLite path that does not exist. sqlite3 would create the file,
    so a typo starts green and hands the agent an empty database.
    """
    if url.get_backend_name() != "sqlite":
        return
    if not url.database or not pathlib.Path(url.database).is_file():
        raise TargetUnreachable("FileNotFoundError")


def target_url() -> URL:
    """TARGET_DATABASE_URL, with the driver spelled out.

    A bare `postgresql://` resolves to psycopg2 in SQLAlchemy, which is not
    installed, so the scheme is normalised here rather than in everyone's .env.
    """
    url = make_url(settings().target_database_url)
    return url.set(drivername=_DRIVERS.get(url.drivername, url.drivername))


def target_dialect() -> str:
    """`postgresql` | `mysql` | `sqlite` — which SQL the model should write."""
    return target_url().get_backend_name()


async def target_engine() -> AsyncEngine:
    """The engine for the target database, built on first use.

    Double-checked under a lock so a burst of concurrent turns against a cold
    start builds one engine between them rather than one each.
    """
    global _target
    if _target is not None:
        return _target

    async with _target_lock:
        if _target is not None:
            return _target
        url = target_url()
        _guard_sqlite_path(url)
        engine = create_async_engine(
            url,
            # `explore` holds one connection across up to 24 tool calls with
            # model round trips between them, and SQLAlchemy would otherwise
            # open a transaction on the first statement — holding a snapshot on
            # a customer's database for minutes. `target_readonly` opts back in.
            isolation_level="AUTOCOMMIT",
            **_engine_kwargs(url.get_backend_name()),
        )
        dialects.install(engine, url.get_backend_name(), config().statement_timeout_ms)
        try:
            # `create_async_engine` is a factory and never dials, so it succeeds
            # against a host that is not listening. Without this the failure
            # surfaces as an error event inside a 200 SSE response, not a 502.
            async with asyncio.timeout(settings().target_connect_timeout):
                async with engine.connect():
                    pass
        except Exception as e:
            await engine.dispose()
            raise TargetUnreachable(cause_of(e)) from e
        _target = engine
        return engine


@asynccontextmanager
async def agent() -> AsyncIterator[AsyncConnection]:
    """Read/write against the agent's own memory."""
    async with agent_pool().connection() as conn:
        yield conn


@asynccontextmanager
async def target() -> AsyncIterator[TargetConnection]:
    """The target database, for introspection and fingerprinting.

    Read-only because `dialects.install` set the session that way, and AUTOCOMMIT
    so the explore loop holds no transaction open across model round trips.
    """
    engine = await target_engine()
    async with engine.connect() as conn:
        yield conn


@asynccontextmanager
async def target_readonly() -> AsyncIterator[TargetConnection]:
    """The transaction the agent's generated SQL runs inside.

    The only place that wants a real transaction. What runs inside it is
    per-dialect and may be empty — see `app/dialects.py`.
    """
    async with readonly(await target_engine()) as conn:
        yield conn


@asynccontextmanager
async def readonly(engine: AsyncEngine) -> AsyncIterator[TargetConnection]:
    """The read-only transaction, on any engine.

    Split out from `target_readonly` so the capability tests can apply it to an
    engine on another dialect: the app dials one database, and the claims in
    `app/dialects.py` are about all three.
    """
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
