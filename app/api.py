"""The versioned API. Every route the outside world touches lives here.

`scripts/*` are HTTP clients of this module, not a second implementation — the
graph, the pool and the checkpointer exist in one process, and the code path the
demo exercises is the code path a user gets.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app import db, store
from app.events import sse
from app.graph import stream_turn
from app.schemas import (
    AskBody,
    CacheEntryOut,
    CacheListOut,
    CacheSummary,
    Kind,
    ResetOut,
)
from app.settings import settings


async def require_token(authorization: str = Header(default="")) -> None:
    """Bearer auth for everything under /v1.

    Read at request time, not import time: `settings()` is `lru_cache`d and the
    test suite clears it, so a module-level snapshot would freeze whatever the
    first import happened to see.

    An empty `API_TOKEN` disables enforcement — that is what tests and a first
    `make up` run on. The server says so at startup rather than leaving it
    silent (see `app.main`).
    """
    expected = settings().api_token
    if not expected:
        return

    scheme, _, presented = authorization.partition(" ")
    # compare_digest, so a wrong token can't be found a character at a time.
    if scheme.lower() != "bearer" or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Applied to the router rather than to each route: a new endpoint is
# authenticated by default, instead of by remembering.
router = APIRouter(prefix="/v1", dependencies=[Depends(require_token)])


@router.post("/ask")
async def ask(req: Request, body: AskBody) -> StreamingResponse:
    """One turn, streamed as it happens.

    Streaming is the product: the point is watching T1's exploration scroll past
    and then *not* happen on T2.
    """
    graph = req.app.state.graph
    session_id = str(body.session_id)

    async def gen():
        async for ev in stream_turn(graph, session_id, body.question):
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
            # Tells a caller that didn't supply one which thread it just used,
            # so the next question can continue the same conversation. Headers
            # flush before the body, so this arrives immediately.
            "X-Session-Id": session_id,
        },
    )


@router.get("/cache", response_model=CacheListOut)
async def read_cache(kind: Kind | None = Query(default=None)) -> CacheListOut:
    """What the agent has learned.

    Served through `store.load_cache()`, so this is exactly what the model reads
    on the next turn — same entries, same order, tombstones included. That
    equivalence is the point: the cache is the product (PLAN.md §6.2).

    `kind` filters the listing only. It never changes what the model would see.
    """
    async with db.connection() as conn:
        entries = await store.load_cache(conn)
        stale = await store.stale_ids(conn, entries)
        disabled = await store.count_disabled(conn)

    shown = [e for e in entries if kind is None or e.kind == kind]
    return CacheListOut(
        # Counted over everything loaded, not over `shown` — a filtered view
        # that also reported filtered totals would misreport the cache's size.
        summary=CacheSummary(
            total=len(entries),
            verified=sum(1 for e in entries if e.verified),
            stale=len(stale),
            disabled=disabled,
        ),
        entries=[
            CacheEntryOut(
                id=e.id,
                kind=e.kind,
                name=e.name,
                claim=e.claim,
                sql_fragment=e.sql_fragment,
                tables=e.tables,
                origin=e.origin,
                pinned=e.pinned,
                tombstone=e.tombstone,
                verified=e.verified,
                hits=e.hits,
                stale=e.id in stale,
            )
            for e in shown
            if e.id is not None
        ],
    )


@router.delete("/cache", response_model=ResetOut)
async def reset_cache() -> ResetOut:
    """Forget everything. The stage recovery button (PLAN.md §9).

    Takes the turn log and the checkpoints with it — see `store.reset_learned`.
    The business schema is untouched; reseeding it is `scripts/seed.py`'s job.
    """
    async with db.connection() as conn:
        wiped = await store.reset_learned(conn)
    return ResetOut(wiped=wiped)
