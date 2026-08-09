import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app import db, store
from app.api import router as v1
from app.graph import build_graph
from app.settings import settings

log = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    agent_pool = await db.open_pools()
    # Checkpoints are the agent's own state, so they belong on the agent's
    # server — never on the database it is answering questions about.
    checkpointer = AsyncPostgresSaver(agent_pool)
    await checkpointer.setup()  # idempotent; creates the checkpoint tables

    # Built once and shared: the graph, the pool and the checkpointer exist in
    # this process only, which is the whole reason the clients go through HTTP.
    app.state.graph = build_graph(checkpointer)

    # Copy TARGET_DATABASE_URL's address onto the `default` registry row so a
    # listing shows something true. Best-effort: on a fresh clone `make up`
    # runs before `make migrate` has ever created the table.
    try:
        await db.ensure_default_connection()
    except Exception as e:
        log.warning("could not resolve the 'default' connection (%s) — run 'make migrate'", type(e).__name__)

    if not settings().api_token:
        # An unauthenticated DELETE on a connection's cache wipes everything the
        # agent has learned about it, so this should never be a surprise.
        log.warning("API_TOKEN is unset — /v1 is open and unauthenticated")

    if not settings().connection_secret:
        # The same bargain: an insecure default is fine when it is loud.
        log.warning(
            "CONNECTION_SECRET is unset — registered warehouse passwords are "
            "stored in plaintext on agent-db"
        )

    try:
        yield
    finally:
        await db.close_pools()


app = FastAPI(title="sql-agent", lifespan=lifespan)
app.include_router(v1)


@app.get("/health")
async def health() -> dict[str, str | int]:
    """Unversioned and unauthenticated, so a load balancer can reach it.

    The agent's own database, and deliberately not the registered targets. It
    used to ping the target too, back when there was exactly one — but a load
    balancer asking "should I send traffic here?" gets a *wrong* answer if one
    customer's warehouse is down: this process can still list connections, serve
    every other connection's questions, and read every cache. Pinging all of
    them would also make the check as slow as the slowest warehouse.

    Per-target reachability moved to POST /v1/connections/{id}/test, where it is
    a fact about a connection rather than a fact about the process.
    """
    async with db.agent() as conn:
        await (await conn.execute("SELECT 1 AS ok")).fetchone()
        registered = len(await store.list_connections(conn))
    return {"status": "ok", "agent": "ok", "connections": registered}
