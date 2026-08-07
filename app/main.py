import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app import db
from app.api import router as v1
from app.graph import build_graph
from app.settings import settings

log = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    agent_pool, _ = await db.open_pools()
    # Checkpoints are the agent's own state, so they belong on the agent's
    # server — never on the database it is answering questions about.
    checkpointer = AsyncPostgresSaver(agent_pool)
    await checkpointer.setup()  # idempotent; creates the checkpoint tables

    # Built once and shared: the graph, the pool and the checkpointer exist in
    # this process only, which is the whole reason the scripts go through HTTP.
    app.state.graph = build_graph(checkpointer)

    if not settings().api_token:
        # An unauthenticated DELETE /v1/cache wipes everything the agent has
        # learned, so this should never be a surprise.
        log.warning("API_TOKEN is unset — /v1 is open and unauthenticated")

    try:
        yield
    finally:
        await db.close_pools()


app = FastAPI(title="sql-agent", lifespan=lifespan)
app.include_router(v1)


@app.get("/health")
async def health() -> dict[str, str]:
    """Unversioned and unauthenticated, so a load balancer can reach it.

    Both servers, because a turn needs both: the agent can be perfectly able to
    read its cache and still unable to answer anything.
    """
    async with db.agent() as conn:
        await (await conn.execute("SELECT 1 AS ok")).fetchone()
    async with db.target() as conn:
        await (await conn.execute("SELECT 1 AS ok")).fetchone()
    return {"status": "ok", "agent": "ok", "target": "ok"}
