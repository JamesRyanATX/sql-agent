from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.settings import settings

_pool: AsyncConnectionPool | None = None


async def open_pool() -> AsyncConnectionPool:
    """Open the shared pool. Called once on app startup."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            settings().database_url,
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
        await _pool.open(wait=True, timeout=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("pool not open — call open_pool() during startup")
    return _pool


@asynccontextmanager
async def connection() -> AsyncIterator[AsyncConnection]:
    async with pool().connection() as conn:
        yield conn


@asynccontextmanager
async def readonly() -> AsyncIterator[AsyncConnection]:
    """A transaction the agent's generated SQL runs inside.

    Two guards, both of which have to be inside an explicit transaction for
    SET LOCAL to mean anything: the session can't write, and a runaway query
    dies rather than hanging the demo.
    """
    async with pool().connection() as conn:
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
