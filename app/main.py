import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field

from app import db
from app.events import sse
from app.graph import build_graph, stream_turn


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pool = await db.open_pool()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # idempotent; creates the checkpoint tables
    app.state.graph = build_graph(checkpointer)
    try:
        yield
    finally:
        await db.close_pool()


app = FastAPI(title="sql-agent", lifespan=lifespan)


class AskBody(BaseModel):
    session_id: str = Field(..., description="stable per conversation")
    question: str


@app.get("/health")
async def health() -> dict[str, str]:
    async with db.connection() as conn:
        cur = await conn.execute("SELECT 1 AS ok")
        await cur.fetchone()
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: Request, body: AskBody) -> StreamingResponse:
    graph = req.app.state.graph

    async def gen():
        async for ev in stream_turn(graph, body.session_id, body.question):
            # Without this a closed tab leaves the graph running and burning
            # tokens with nobody watching.
            if await req.is_disconnected():
                break
            yield sse(ev)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Without this nginx buffers the whole stream and the live demo
            # looks frozen until the turn finishes.
            "X-Accel-Buffering": "no",
        },
    )
