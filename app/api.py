"""The versioned API. Every route the outside world touches lives here.

`scripts/*` are HTTP clients of this module, not a second implementation — the
graph, the pool and the checkpointer exist in one process, and the code path the
demo exercises is the code path a user gets.
"""

from __future__ import annotations

import secrets
import time

import psycopg
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, status
from fastapi.responses import Response, StreamingResponse

from app import db, store
from app.events import sse
from app.graph import stream_turn
from app.schemas import (
    AskBody,
    CacheEntryOut,
    CacheListOut,
    CacheSummary,
    ConnectionCreate,
    ConnectionCreatedOut,
    ConnectionListOut,
    ConnectionOut,
    ConnectionPatch,
    ConnectionTestOut,
    Kind,
    ResetOut,
    TurnListOut,
    TurnOut,
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


async def connection_dep(cid: str = Path(...)) -> store.Connection:
    """Resolve the connection a scoped route is about, or 404.

    Hung off the sub-router below rather than repeated in each handler, for the
    same reason `require_token` is hung off `router`: a route added later is
    scoped by default instead of by somebody remembering. The path segment is
    what makes it unforgettable — there is no unscoped route to reach by
    accident, because it doesn't exist.
    """
    try:
        return await db.resolve(cid)
    except db.UnknownConnection:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"no connection named {cid!r}"
        ) from None


# Everything about one registered database hangs here.
scoped = APIRouter(
    prefix="/connections/{cid}", dependencies=[Depends(connection_dep)]
)


def _out(registered: store.Connection, stats: dict[str, int] | None = None) -> ConnectionOut:
    stats = stats or {"cache_entries": 0, "turns": 0}
    return ConnectionOut(
        id=registered.id,
        label=registered.label,
        origin=registered.origin,
        host=registered.host,
        port=registered.port,
        database=registered.database,
        username=registered.username,
        sslmode=registered.sslmode,
        has_password=registered.password is not None,
        dsn=registered.safe_dsn(),
        cache_entries=stats["cache_entries"],
        turns=stats["turns"],
        created_at=registered.created_at,
        updated_at=registered.updated_at,
    )


async def _probe(registered: store.Connection) -> ConnectionTestOut:
    """Connect, and report what we found. Never raises.

    Deliberately does not go through `db.target_pool` — a probe of a connection
    that may not work should not leave a cached pool behind, and a probe run
    right after a PATCH has to dial the new address rather than a pooled old
    one.
    """
    started = time.monotonic()
    try:
        conn = await psycopg.AsyncConnection.connect(
            registered.conninfo(),
            connect_timeout=int(settings().target_connect_timeout),
            row_factory=psycopg.rows.dict_row,
            autocommit=True,
        )
    except Exception as e:
        # The exception's message can quote the conninfo, and the conninfo
        # carries the password. Type and nothing else.
        return ConnectionTestOut(ok=False, error=_sanitise(e, registered))
    try:
        cur = await conn.execute(
            """
            SELECT current_user AS username,
                   version() AS server_version,
                   current_setting('is_superuser') = 'on' AS superuser,
                   (SELECT count(*) FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE')
                       AS tables,
                   (SELECT count(*) FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public' AND c.relkind = 'r'
                      AND has_table_privilege(c.oid, 'INSERT')) AS writable
            """
        )
        row = await cur.fetchone()
    except Exception as e:
        return ConnectionTestOut(ok=False, error=_sanitise(e, registered))
    finally:
        await conn.close()

    assert row is not None
    warnings = []
    if row["superuser"]:
        warnings.append("connects as a superuser")
    if row["writable"]:
        warnings.append(f"can INSERT into {row['writable']} table(s)")
    if warnings:
        warnings.append(
            "the agent only ever reads, and the session it opens is read-only — "
            "but a role holding SELECT and nothing else is the better answer"
        )
    return ConnectionTestOut(
        ok=True,
        latency_ms=int((time.monotonic() - started) * 1000),
        server_version=row["server_version"].split(" on ")[0],
        username=row["username"],
        tables=row["tables"],
        read_only=not row["superuser"] and not row["writable"],
        warnings=warnings,
    )


def _sanitise(e: Exception, registered: store.Connection) -> str:
    """A connection failure a caller can act on, with no credentials in it."""
    detail = str(e).strip().splitlines()[0] if str(e).strip() else ""
    if registered.password and registered.password in detail:
        detail = ""
    return f"{type(e).__name__}: {detail}" if detail else type(e).__name__


# ------------------------------------------------------------------- registry


@router.get("/connections", response_model=ConnectionListOut)
async def list_connections() -> ConnectionListOut:
    """Every registered database. Never carries a password — see ConnectionOut."""
    async with db.agent() as conn:
        rows = await store.list_connections(conn)
        stats = await store.connection_stats(conn)
    return ConnectionListOut(
        connections=[_out(r, stats.get(r.id)) for r in rows]
    )


@router.post(
    "/connections",
    response_model=ConnectionCreatedOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_connection(
    body: ConnectionCreate,
    response: Response,
    probe: bool = Query(default=True),
) -> ConnectionCreatedOut:
    """Register a database.

    The probe runs by default and its result comes back *inside* the 201, but a
    failed probe does not block the create: a typo should be visible now, while
    the user is still looking at it, whereas refusing to save a connection
    because the warehouse is down for maintenance is worse than saving it with a
    warning.

    Give the agent a role holding SELECT and nothing else. We cannot make one
    for you, and with a registered connection the read-only session is what
    stands in for it.
    """
    registered = store.Connection(
        id=body.id,
        origin="api",
        label=body.label,
        host=body.host,
        port=body.port,
        database=body.database,
        username=body.username,
        password=body.password,
        sslmode=body.sslmode,
    )
    try:
        async with db.agent() as conn:
            written = await store.create_connection(conn, registered)
    except psycopg.errors.UniqueViolation:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"connection {body.id!r} already exists",
        ) from None

    response.headers["Location"] = f"/v1/connections/{written.id}"
    return ConnectionCreatedOut(
        connection=_out(written),
        test=await _probe(written) if probe else None,
    )


@scoped.get("", response_model=ConnectionOut)
async def read_connection(
    registered: store.Connection = Depends(connection_dep),
) -> ConnectionOut:
    async with db.agent() as conn:
        stats = await store.connection_stats(conn)
    return _out(registered, stats.get(registered.id))


@scoped.patch("", response_model=ConnectionOut)
async def patch_connection(
    body: ConnectionPatch,
    registered: store.Connection = Depends(connection_dep),
) -> ConnectionOut:
    """Change some fields. Only what you send is touched.

    An omitted `password` keeps the stored one — clearing it takes an explicit
    empty string — so changing a port does not mean re-sending a secret.
    """
    _refuse_env(registered)
    fields = body.model_dump(exclude_unset=True)
    async with db.agent() as conn:
        updated = await store.update_connection(conn, registered.id, **fields)
        stats = await store.connection_stats(conn)
    assert updated is not None  # connection_dep already resolved it
    # The pooled connections point at the old address.
    await db.evict(registered.id)
    return _out(updated, stats.get(registered.id))


@scoped.delete("", response_model=ResetOut)
async def delete_connection(
    registered: store.Connection = Depends(connection_dep),
) -> ResetOut:
    """Forget a database, and everything the agent learned about it."""
    _refuse_env(registered)
    async with db.agent() as conn:
        wiped = await store.delete_connection(conn, registered.id)
    await db.evict(registered.id)
    return ResetOut(wiped=wiped)


@scoped.post("/test", response_model=ConnectionTestOut)
async def test_connection(
    registered: store.Connection = Depends(connection_dep),
) -> ConnectionTestOut:
    """Can we reach it, and what are we?

    **200 even when the probe fails**, with `ok: false` and a sanitised error. A
    failed diagnostic is a successful diagnostic: a non-2xx would make a client
    print "502 from /v1/…" where the useful sentence is "password
    authentication failed for user 'analytics'".
    """
    return await _probe(registered)


def _refuse_env(registered: store.Connection) -> None:
    if registered.origin == "env":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"{registered.id!r} is owned by the environment — its address is "
                "TARGET_DATABASE_URL. Change it there, not here."
            ),
        )


# ----------------------------------------------------------------- one turn


@scoped.post("/ask")
async def ask(
    req: Request,
    body: AskBody,
    registered: store.Connection = Depends(connection_dep),
) -> StreamingResponse:
    """One turn, streamed as it happens.

    Streaming is the product: the point is watching T1's exploration scroll past
    and then *not* happen on T2.

    The connection is opened here, before the response is returned, and not
    lazily inside the graph. Once the generator is handed to Starlette a 200 is
    already on the wire, and an unreachable warehouse would arrive as an error
    event inside a successful response instead of the 502 it is.
    """
    graph = req.app.state.graph
    session_id = str(body.session_id)

    async with db.agent() as conn:
        bound = await store.session_connection(conn, session_id)
    if bound is not None and bound != registered.id:
        # The thread's history is checkpointed. Reusing it against a second
        # warehouse would hand the model the first one's conversation while it
        # answers about the second.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"session {session_id} has been asking about {bound!r} — "
                "start a new session to ask about another connection"
            ),
        )

    try:
        await db.target_pool(registered.id)
    except db.TargetUnreachable as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(e)) from None

    async def gen():
        async for ev in stream_turn(graph, session_id, body.question, registered.id):
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


@scoped.get("/cache", response_model=CacheListOut)
async def read_cache(
    kind: Kind | None = Query(default=None),
    registered: store.Connection = Depends(connection_dep),
) -> CacheListOut:
    """What the agent has learned about this connection.

    Served through `store.load_cache()`, so this is exactly what the model reads
    on the next turn — same entries, same order, tombstones included. That
    equivalence is the point: the cache is the product (PLAN.md §6.2).

    `kind` filters the listing only. It never changes what the model would see.
    """
    async with db.agent() as conn:
        entries = await store.load_cache(conn, connection_id=registered.id)
        disabled = await store.count_disabled(conn, connection_id=registered.id)
    # Staleness is a question about the *business* schema — has it moved since
    # these entries were learned? — so it is asked on the other server, and on
    # **this entry's** other server. Asking a different connection's target
    # would report every entry stale, or worse, coincidentally not stale.
    async with db.target(registered.id) as conn:
        stale = await store.stale_ids(conn, entries)

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


@scoped.delete("/cache", response_model=ResetOut)
async def reset_cache(
    registered: store.Connection = Depends(connection_dep),
) -> ResetOut:
    """Forget everything learned about this connection. The stage recovery
    button (PLAN.md §9).

    Takes this connection's turn log and the checkpoints of its sessions with it
    — see `store.reset_learned`. Another connection's cache is untouched, and
    the registry row itself survives: this is "forget what you learned", not
    "forget the database exists".

    The business data is on another server that this connection cannot reach,
    so "leaves the business schema alone" is now a property of the wiring rather
    than a promise.
    """
    async with db.agent() as conn:
        wiped = await store.reset_learned(conn, connection_id=registered.id)
    return ResetOut(wiped=wiped)


@scoped.get("/turns", response_model=TurnListOut)
async def read_turns(
    limit: int = Query(default=50, ge=1, le=500),
    finished: bool = Query(default=True),
    registered: store.Connection = Depends(connection_dep),
) -> TurnListOut:
    """The turn log: what was asked, and what it cost.

    This is the demo chart as rows, and it is the whole reason `make turns`
    could stop being a psql query. Ascending, so it reads left to right.

    `finished` defaults to true, matching what `make turns` showed. False also
    shows turns still in flight and the ones that failed, which that query had
    no way to ask for.
    """
    async with db.agent() as conn:
        rows = await store.read_turns(
            conn, connection_id=registered.id, limit=limit, finished=finished
        )
    return TurnListOut(
        turns=[
            TurnOut(**r, tokens=r["tokens_in"] + r["tokens_out"]) for r in rows
        ]
    )


# Registered last so every scoped route carries `connection_dep`.
router.include_router(scoped)
