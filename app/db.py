"""Two pools, two servers.

The agent's memory (`cache_entry`, `turn`, the checkpoints) and the data it
answers questions about live on separate Postgres servers. They can't see each
other because they aren't in the same place — not because application code
remembers to filter.

The connection to the business data is named for the role it plays, `target`,
rather than for the demo fixture that fills it locally. The container is called
`demo-db` because that is what it is; this is what it does.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.settings import settings

_agent_pool: AsyncConnectionPool | None = None
_target_pool: AsyncConnectionPool | None = None


def _make_pool(url: str) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        url,
        min_size=1,
        max_size=10,
        open=False,
        # autocommit + dict_row + prepare_threshold=0 are what
        # AsyncPostgresSaver requires of a pool it's handed.
        kwargs={
            "row_factory": dict_row,
            "autocommit": True,
            "prepare_threshold": 0,
        },
    )


async def open_pools() -> tuple[AsyncConnectionPool, AsyncConnectionPool]:
    """Open both pools. Called once on app startup. Returns (agent, target)."""
    global _agent_pool, _target_pool
    if _agent_pool is None:
        _agent_pool = _make_pool(settings().agent_database_url)
        await _agent_pool.open(wait=True, timeout=10)
    if _target_pool is None:
        _target_pool = _make_pool(settings().target_database_url)
        await _target_pool.open(wait=True, timeout=10)
    return _agent_pool, _target_pool


async def close_pools() -> None:
    global _agent_pool, _target_pool
    for pool in (_agent_pool, _target_pool):
        if pool is not None:
            await pool.close()
    _agent_pool = _target_pool = None


def agent_pool() -> AsyncConnectionPool:
    if _agent_pool is None:
        raise RuntimeError("pools not open — call open_pools() during startup")
    return _agent_pool


def target_pool() -> AsyncConnectionPool:
    if _target_pool is None:
        raise RuntimeError("pools not open — call open_pools() during startup")
    return _target_pool


@asynccontextmanager
async def agent() -> AsyncIterator[AsyncConnection]:
    """Read/write against the agent's own memory."""
    async with agent_pool().connection() as conn:
        yield conn


@asynccontextmanager
async def target() -> AsyncIterator[AsyncConnection]:
    """The business data, for introspection and fingerprinting.

    Read-only by role — the credentials here hold SELECT and nothing else — so
    this needs no transaction guard of its own.
    """
    async with target_pool().connection() as conn:
        yield conn


@asynccontextmanager
async def target_readonly() -> AsyncIterator[AsyncConnection]:
    """A transaction the agent's generated SQL runs inside.

    Two guards, both of which have to be inside an explicit transaction for
    SET LOCAL to mean anything: the session can't write, and a runaway query
    dies rather than hanging the demo.

    The read-only half is now the *second* line of defence — the role can't
    write either. Kept because defence that only exists in one place is one
    edit from not existing, and because the statement timeout has no
    equivalent at the role level.
    """
    async with target_pool().connection() as conn:
        await conn.set_autocommit(False)
        try:
            async with conn.transaction():
                # set_config(..., is_local => true) rather than SET LOCAL:
                # SET takes no bind parameters, so the timeout would have to be
                # string-interpolated. Both are transaction-scoped either way.
                await conn.execute(
                    "SELECT set_config('transaction_read_only', 'on', true)"
                )
                await conn.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (settings().statement_timeout,),
                )
                yield conn
        finally:
            await conn.set_autocommit(True)
