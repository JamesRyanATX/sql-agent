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
    pool = await db.open_pool()
    checkpointer = AsyncPostgresSaver(pool)
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
        await db.close_pool()


app = FastAPI(title="sql-agent", lifespan=lifespan)
app.include_router(v1)


@app.get("/health")
async def health() -> dict[str, str]:
    """Unversioned and unauthenticated, so a load balancer can reach it."""
    async with db.connection() as conn:
        cur = await conn.execute("SELECT 1 AS ok")
        await cur.fetchone()
    return {"status": "ok"}
