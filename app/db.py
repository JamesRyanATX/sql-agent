"""One pool for the agent's memory, and one per registered target.

The agent's memory (`cache_entry`, `turn`, the checkpoints) and the data it
answers questions about live on separate Postgres servers. They can't see each
other because they aren't in the same place — not because application code
remembers to filter.

The connections to the business data are named for the role they play, `target`,
rather than for the demo fixture that fills one of them locally. The container is
called `demo-db` because that is what it is; this is what it does.

**The target is no longer a single address.** `target(cid)` and
`target_readonly(cid)` take a required connection id and resolve it through the
registry in `app.store`. There is no default: a default is how a mis-scoped node
queries the demo warehouse, looks correct on stage, and answers a customer's
question against somebody else's data.

**The read-only guarantee had to move.** It used to rest on the credentials —
`demo/demo.sql` builds a role holding SELECT and nothing else, so the transaction
guard was a second line of defence. With registered connections the credentials
are whatever the user handed us, and that sentence stopped being true. So every
connection out of every target pool is now put into a read-only session by
`_readonly_session` below, and `target_readonly()`'s explicit transaction sits on
top of it carrying the statement timeout. `transaction_read_only` is a property
of the transaction, not a privilege, so it holds for any role — including a
superuser.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# `store` for the registry lookup. This is a layering inversion — db is the
# lower layer — but it is acyclic (store imports psycopg and app.secrets, never
# app.db) and checked by the import. An injected resolver would buy nothing but
# a place for the lookup to hide.
from app import store
from app.settings import settings

_agent_pool: AsyncConnectionPool | None = None
_target_pools: dict[str, AsyncConnectionPool] = {}
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


def _make_pool(url: str, **kw) -> AsyncConnectionPool:
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


async def _readonly_session(conn: AsyncConnection) -> None:
    """Put a freshly-opened target connection into a read-only session.

    Runs on every connection from every target pool, which is the point: the
    agent reaches a registered warehouse with credentials it did not choose, and
    `db.target()` — used by `explore`, `infer_tables` and `extract`'s
    fingerprinting — has no transaction of its own to guard.

    SET SESSION CHARACTERISTICS rather than `options=-c ...` in the conninfo,
    because pgbouncer in transaction mode rejects the latter.
    """
    await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")


async def open_pools() -> AsyncConnectionPool:
    """Open the agent pool. Called once on app startup.

    Target pools are opened on first use, not here: at startup we do not yet
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
    pools = [_agent_pool, *_target_pools.values()]
    _target_pools.clear()
    _agent_pool = None
    for pool in pools:
        if pool is not None:
            await pool.close()


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


async def target_pool(connection_id: str) -> AsyncConnectionPool:
    """The pool for one registered target, opened on first use.

    Double-checked under a lock so a burst of concurrent turns against a
    cold connection opens one pool between them rather than one each.
    """
    pool = _target_pools.get(connection_id)
    if pool is not None:
        return pool

    async with _registry_lock:
        if (pool := _target_pools.get(connection_id)) is not None:
            return pool
        registered = await resolve(connection_id)
        pool = _make_pool(
            registered.conninfo(),
            # min_size=0, unlike the agent pool: a registry of ten warehouses
            # must not hold ten idle sockets open against ten customers'
            # databases because somebody once asked a question. max_idle lets
            # it shrink back to nothing between turns.
            min_size=0,
            max_size=settings().target_pool_max,
            max_idle=settings().target_pool_max_idle,
            configure=_readonly_session,
        )
        try:
            await pool.open()
            # Actually take a connection, rather than trusting `open(wait=True)`.
            # `wait` waits for `min_size` connections to be ready, and min_size
            # is 0 here — so opening a pool against a warehouse that is not
            # listening succeeds, and the failure surfaces later as an error
            # event inside a 200 SSE response instead of the 502 it is.
            async with pool.connection(timeout=settings().target_connect_timeout):
                pass
        except Exception as e:
            await pool.close()
            raise TargetUnreachable(connection_id, type(e).__name__) from e
        _target_pools[connection_id] = pool
        return pool


async def evict(connection_id: str) -> None:
    """Drop a target pool, so the next turn re-resolves its address.

    Called after a connection is updated or deleted. Honest about what it is
    not: `close()` does not wait for checked-out connections, so a PATCH landing
    mid-turn is last-writer-wins for that turn — `explore` may have run against
    the old address and `execute` against the new. The registry is not
    transactional with respect to a running turn and cannot be, because
    TurnState is checkpointed and cannot hold a pool.
    """
    pool = _target_pools.pop(connection_id, None)
    if pool is not None:
        await pool.close()


@asynccontextmanager
async def agent() -> AsyncIterator[AsyncConnection]:
    """Read/write against the agent's own memory."""
    async with agent_pool().connection() as conn:
        yield conn


@asynccontextmanager
async def target(connection_id: str) -> AsyncIterator[AsyncConnection]:
    """A registered target, for introspection and fingerprinting.

    Read-only because `_readonly_session` put the session into that state when
    the connection was opened — not because of the credentials, which we did not
    choose.
    """
    pool = await target_pool(connection_id)
    async with pool.connection() as conn:
        yield conn


@asynccontextmanager
async def target_readonly(connection_id: str) -> AsyncIterator[AsyncConnection]:
    """A transaction the agent's generated SQL runs inside.

    Two guards, both of which have to be inside an explicit transaction for
    SET LOCAL to mean anything: the transaction can't write, and a runaway query
    dies rather than hanging the demo.

    The read-only half is now redundant with the session-level setting above,
    and stays anyway: defence that only exists in one place is one edit from not
    existing, and the statement timeout has no session-level equivalent to hang
    off.
    """
    pool = await target_pool(connection_id)
    async with pool.connection() as conn:
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
