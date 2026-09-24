"""The versioned API. Every route the outside world touches lives here.

`cli/sql_agent/` is an HTTP client of this module, not a second implementation —
the graph, the pool and the checkpointer exist in one process, and the code path
the demo exercises is the code path a user gets.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.engine import URL
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app import db, dialects, store, tracing
from app.config import config, overlay, overrides
from app.events import sse
from app.graph import stream_turn
from app.schemas import (
    AskBody,
    CacheEntryOut,
    CacheListOut,
    CacheSummary,
    ConfigOut,
    FeedbackBody,
    FeedbackOut,
    Kind,
    ResetOut,
    TargetTestOut,
    TurnListOut,
    TurnOut,
)
from app.settings import settings

# The names a verdict and a reference query are filed under, so a harvest has
# two things to filter on. Here rather than in `tracing`, which does not know
# what is being scored. `tools/gepa/harvest.py` reads both back.
SCORE = "correct"
REFERENCE = "reference"


async def require_token(authorization: str = Header(default="")) -> None:
    """Bearer auth for everything under /v1.

    Read at request time, not import time, because the suite clears `settings()`.
    An empty `API_TOKEN` disables enforcement, and `app.main` warns at startup.
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


async def _probe(url: URL | None = None) -> TargetTestOut:
    """Connect and report what we found. Never raises.

    `url` defaults to the target database. The capability tests pass another, so
    the claims in `app/dialects.py` are checked on every engine while the app
    itself is pointed at one.

    Not through `db.target_engine` (hence `NullPool`): a probe should leave no
    cached engine behind, and one run after the address changed must dial the
    new one.

    **No read-only hooks on this engine** — the question is what these
    credentials *could* do, and a read-only session makes every database look so.
    """
    started = time.monotonic()
    url = url or db.target_url()
    cap = dialects.for_dialect(url.get_backend_name())
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with asyncio.timeout(settings().target_connect_timeout):
            async with engine.connect() as conn:

                def reflect(sync_conn):
                    inspector = sa_inspect(sync_conn)
                    return (
                        len(inspector.get_table_names()),
                        inspector.default_schema_name,
                        sync_conn.dialect.server_version_info,
                    )

                tables, schema, version = await conn.run_sync(reflect)
                username, writable, superuser = await _PRIVILEGE[
                    url.get_backend_name()
                ](conn, url)
    except Exception as e:
        return TargetTestOut(ok=False, driver=url.drivername, error=_sanitise(e))
    finally:
        await engine.dispose()

    warnings = list(cap.gaps)
    if superuser:
        warnings.append("connects as a superuser")
    if writable:
        warnings.append("these credentials can write to this database")
    if superuser or writable:
        warnings.append(
            "the agent only ever reads, and the session it opens is read-only "
            f"as far as {url.get_backend_name()} allows — but a role holding SELECT "
            "and nothing else is the better answer"
        )
    return TargetTestOut(
        ok=True,
        driver=url.drivername,
        readonly_tier=cap.tier,
        latency_ms=int((time.monotonic() - started) * 1000),
        server_version=_version(url.get_backend_name(), version),
        username=username,
        default_schema=schema,
        tables=tables,
        # None where there is nothing to judge — on SQLite there are no
        # credentials, and reporting True would be a lie a user would act on.
        read_only=None if writable is None else not (writable or superuser),
        warnings=warnings,
    )


def _version(dialect: str, info: tuple | None) -> str:
    label = {"postgresql": "PostgreSQL", "mysql": "MySQL", "sqlite": "SQLite"}[dialect]
    return f"{label} {'.'.join(str(p) for p in info)}" if info else label


async def _pg_privileges(conn, url) -> tuple[str | None, bool | None, bool]:
    row = (
        await conn.exec_driver_sql(
            """
            SELECT current_user AS username,
                   current_setting('is_superuser') = 'on' AS superuser,
                   (SELECT count(*) FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = current_schema() AND c.relkind = 'r'
                      AND has_table_privilege(c.oid, 'INSERT')) AS writable
            """
        )
    ).mappings().one()
    return row["username"], bool(row["writable"]), bool(row["superuser"])


async def _mysql_privileges(conn, url) -> tuple[str | None, bool | None, bool]:
    """`SHOW GRANTS`, not `information_schema.table_privileges` — that view lists
    table-level grants only, so `GRANT ALL ON db.*` reads as read-only: true.
    """
    username = (await conn.exec_driver_sql("SELECT CURRENT_USER()")).scalar_one()
    try:
        grants = (await conn.exec_driver_sql("SHOW GRANTS FOR CURRENT_USER()")).all()
    except Exception:
        # Refused. "We could not find out" is a real answer and has a value.
        return username, None, False
    text = " ".join(str(g[0]).upper() for g in grants)
    writable = any(
        word in text
        for word in ("INSERT", "UPDATE", "DELETE", "ALL PRIVILEGES", "CREATE", "DROP")
    )
    return username, writable, "SUPER" in text or "ALL PRIVILEGES ON *.*" in text


async def _sqlite_privileges(conn, url) -> tuple[str | None, bool | None, bool]:
    """A file has no users, so the only question is whether it is writable."""
    path = url.database or ""
    return None, os.access(path, os.W_OK), False


_PRIVILEGE = {
    "postgresql": _pg_privileges,
    "mysql": _mysql_privileges,
    "sqlite": _sqlite_privileges,
}


def _sanitise(e: Exception) -> str:
    """A connection failure a caller can act on, with no credentials in it.

    Unwrapped through `.orig`, because SQLAlchemy's wrapper is
    `OperationalError` for everything. The password check is the backstop.
    """
    orig = getattr(e, "orig", None) or e
    detail = str(orig).strip().splitlines()[0] if str(orig).strip() else ""
    password = db.target_url().password
    if password and password in detail:
        detail = ""
    return f"{type(orig).__name__}: {detail}" if detail else type(orig).__name__


# --------------------------------------------------------------------- server


@router.get("/config", response_model=ConfigOut)
async def read_config() -> ConfigOut:
    """The model configuration this process is running.

    config.yaml with config.local.yaml merged over it, and which keys the
    overlay decided. `model.url` may be an internal address, which is why this
    sits behind the token with everything else under /v1.

    `tracing` rides along although it is environment rather than file: it is
    what a client asks before a run that only pays off if the turns are
    recorded.
    """
    local = overlay()
    return ConfigOut(
        overlay=str(local) if local is not None else None,
        overridden=list(overrides()),
        config=config(),
        tracing=tracing.enabled(),
    )


# ------------------------------------------------------------------- registry


@router.post("/test", response_model=TargetTestOut)
async def test_target() -> TargetTestOut:
    """Can the agent reach the database, and what can those credentials do?

    `ok: false` is a successful diagnostic, not an error — the body carries the
    reason, so a 200 with a false in it is the honest shape.
    """
    return await _probe()


@router.post("/ask")
async def ask(req: Request, body: AskBody) -> StreamingResponse:
    """One turn, streamed as it happens — watching T1's exploration scroll past
    and then *not* happen on T2 is the product.

    The database is dialled here rather than lazily inside the graph: once the
    generator reaches Starlette a 200 is on the wire, and an unreachable
    database would arrive as an error event inside it instead of a 502.

    `memory: false` asks the turn to ignore what has been learned and to keep
    nothing it learns — an optimisation run measuring a prompt, not the cache.
    """
    graph = req.app.state.graph
    session_id = str(body.session_id)

    try:
        await db.target_engine()
    except db.TargetUnreachable as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(e)) from None

    async def gen():
        async for ev in stream_turn(
            graph, session_id, body.question, memory=body.memory
        ):
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
            # Which thread a caller that supplied none just used, so the next
            # question can continue it. Headers flush before the body.
            "X-Session-Id": session_id,
        },
    )


@router.get("/cache", response_model=CacheListOut)
async def read_cache(kind: Kind | None = Query(default=None)) -> CacheListOut:
    """What the agent has learned.

    Served through `store.load_cache()`, so this is exactly what the model reads
    on the next turn — same entries, same order, tombstones included (§6.2).
    `kind` filters the listing only; it never changes what the model sees.
    """
    async with db.agent() as conn:
        entries = await store.load_cache(conn)
        disabled = await store.count_disabled(conn)
    # Staleness is a question about the business schema, so it is asked on the
    # other server.
    async with db.target() as conn:
        stale = await store.stale_ids(conn, entries)

    shown = [e for e in entries if kind is None or e.kind == kind]
    return CacheListOut(
        # Counted over everything loaded, not over `shown`: a filtered view
        # reporting filtered totals would misreport the cache's size.
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
    """Forget everything learned. The stage recovery button (PLAN.md §9).

    Takes the turn log and the checkpoints with it — see `store.reset_learned`.
    "Forget what you learned", not "forget the database": the address is in the
    environment and this cannot touch it.
    """
    async with db.agent() as conn:
        wiped = await store.reset_learned(conn)
    return ResetOut(wiped=wiped)


@router.get("/turns", response_model=TurnListOut)
async def read_turns(
    limit: int = Query(default=50, ge=1, le=500),
    finished: bool = Query(default=True),
) -> TurnListOut:
    """The turn log as rows: what was asked, and what it cost. Ascending, so it
    reads left to right; `finished=false` adds the unfinished and failed turns.
    """
    async with db.agent() as conn:
        rows = await store.read_turns(conn, limit=limit, finished=finished)
    return TurnListOut(
        turns=[
            TurnOut(**r, tokens=r["tokens_in"] + r["tokens_out"]) for r in rows
        ]
    )


@router.post(
    "/turns/{turn_id}/feedback",
    response_model=FeedbackOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def leave_feedback(
    body: FeedbackBody, turn_id: int = Path(...)
) -> FeedbackOut:
    """What a person thought of one answer, onto that turn's trace.

    The verdict belongs with the inputs that produced it, and only the trace
    store has those: `make reset` empties the turn log by design, so a column
    here would hold labels that outlive nothing. A later harvest reads the score
    and the SQL the turn ran from the same trace.

    202, not 200: the client buffers the score and flushes it later, so what
    this route can honestly report is that the verdict was accepted.
    """
    async with db.agent() as conn:
        turn = await store.get_turn(conn, turn_id)
    if turn is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"no turn {turn_id}"
        ) from None

    trace_id = turn["trace_id"] or ""
    # Both halves of "there is nowhere to put this": the turn ran with tracing
    # off, or it is off now. A 202 in either case would be a lie a user only
    # discovers when the corpus comes back empty.
    nowhere = HTTPException(
        status.HTTP_409_CONFLICT,
        detail=(
            f"turn {turn_id} has no trace to score — tracing was off when it "
            "ran, or is off now. Set both Langfuse keys and ask again."
        ),
    )
    if not trace_id:
        raise nowhere

    filed: list[str] = []
    if body.correct is not None:
        value = 1.0 if body.correct else 0.0
        if not tracing.score(
            trace_id=trace_id, name=SCORE, value=value, comment=body.comment
        ):
            raise nowhere
        filed.append(SCORE)
    if body.reference_sql:
        # The query the answer should have come from, with the reading as its
        # comment: what makes this turn a case for the tools and config
        # searches. A TEXT score, so the query is the value and not prose in a
        # comment somebody would have to parse back out.
        if not tracing.score_text(
            trace_id=trace_id, name=REFERENCE, value=body.reference_sql,
            comment=body.reading,
        ):
            raise nowhere
        filed.append(REFERENCE)
    return FeedbackOut(trace_id=trace_id, filed=filed)

