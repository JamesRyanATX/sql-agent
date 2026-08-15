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

import psycopg
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, status
from fastapi.responses import Response, StreamingResponse

from app import db, dialects, store
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


async def connection_dep(cid: str = Path(...)) -> store.Connection:
    """Resolve the connection a scoped route is about, or 404.

    Hung off the sub-router rather than repeated per handler, so a route added
    later is scoped by default and the unscoped call is unrepresentable.
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
        driver=registered.driver,
        host=registered.host,
        port=registered.port,
        database=registered.database,
        username=registered.username,
        sslmode=registered.sslmode,
        has_password=registered.password is not None,
        dsn=registered.safe_dsn(),
        readonly_tier=dialects.for_dialect(registered.dialect).tier,
        cache_entries=stats["cache_entries"],
        turns=stats["turns"],
        created_at=registered.created_at,
        updated_at=registered.updated_at,
    )


async def _probe(registered: store.Connection) -> ConnectionTestOut:
    """Connect, and report what we found. Never raises.

    Not through `db.target_engine` (hence `NullPool`): a probe should leave no
    cached engine behind, and one run after a PATCH must dial the new address.

    **No read-only hooks on this engine** — the question is what these
    credentials *could* do, and a read-only session makes every warehouse look so.
    """
    started = time.monotonic()
    cap = dialects.for_dialect(registered.dialect)
    engine = create_async_engine(registered.url(), poolclass=NullPool)
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
                username, writable, superuser = await _PRIVILEGE[registered.dialect](
                    conn, registered
                )
    except Exception as e:
        return ConnectionTestOut(
            ok=False, driver=registered.driver, error=_sanitise(e, registered)
        )
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
            f"as far as {registered.dialect} allows — but a role holding SELECT "
            "and nothing else is the better answer"
        )
    return ConnectionTestOut(
        ok=True,
        driver=registered.driver,
        readonly_tier=cap.tier,
        latency_ms=int((time.monotonic() - started) * 1000),
        server_version=_version(registered.dialect, version),
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


async def _pg_privileges(conn, registered) -> tuple[str | None, bool | None, bool]:
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


async def _mysql_privileges(conn, registered) -> tuple[str | None, bool | None, bool]:
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


async def _sqlite_privileges(conn, registered) -> tuple[str | None, bool | None, bool]:
    """A file has no users, so the only question is whether it is writable."""
    path = registered.database or ""
    return None, os.access(path, os.W_OK), False


_PRIVILEGE = {
    "postgresql": _pg_privileges,
    "mysql": _mysql_privileges,
    "sqlite": _sqlite_privileges,
}


def _sanitise(e: Exception, registered: store.Connection) -> str:
    """A connection failure a caller can act on, with no credentials in it.

    Unwrapped through `.orig`, because SQLAlchemy's wrapper is
    `OperationalError` for everything. The password check is the backstop.
    """
    orig = getattr(e, "orig", None) or e
    detail = str(orig).strip().splitlines()[0] if str(orig).strip() else ""
    if registered.password and registered.password in detail:
        detail = ""
    return f"{type(orig).__name__}: {detail}" if detail else type(orig).__name__


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

    The probe runs by default and comes back *inside* the 201, but a failed one
    does not block the create — a warehouse down for maintenance should still
    register, with a warning. Give the agent a role holding SELECT and nothing
    else; the read-only session is only what stands in for one.
    """
    registered = store.Connection(
        id=body.id,
        origin="api",
        driver=body.driver,
        label=body.label,
        host=body.host,
        port=body.port,
        database=body.database,
        username=body.username,
        password=body.password,
        sslmode=body.sslmode,
        options=dict(body.options),
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
    if "driver" in fields:
        driver = fields.pop("driver")
        if driver != registered.driver:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail=(
                    f"{registered.id!r} is registered as {registered.driver} — a "
                    "connection's driver cannot be changed, because every recipe "
                    "cached against it is SQL in that dialect. Delete it and "
                    "register it again; that forgets what was learned, which is "
                    "the point."
                ),
            )
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

    **200 even when the probe fails**, with `ok: false` and a sanitised error: a
    non-2xx would make a client print "502 from /v1/…" where the useful sentence
    is "password authentication failed for user 'analytics'".
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
    """One turn, streamed as it happens — watching T1's exploration scroll past
    and then *not* happen on T2 is the product.

    The connection is opened here rather than lazily inside the graph: once the
    generator reaches Starlette a 200 is on the wire, and an unreachable
    warehouse would arrive as an error event inside it instead of a 502.
    """
    graph = req.app.state.graph
    session_id = str(body.session_id)

    async with db.agent() as conn:
        bound = await store.session_connection(conn, session_id)
    if bound is not None and bound != registered.id:
        # The thread's history is checkpointed, so reusing it against a second
        # warehouse hands the model the first one's conversation.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"session {session_id} has been asking about {bound!r} — "
                "start a new session to ask about another connection"
            ),
        )

    try:
        await db.target_engine(registered.id)
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
            # Which thread a caller that supplied none just used, so the next
            # question can continue it. Headers flush before the body.
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
    on the next turn — same entries, same order, tombstones included (§6.2).
    `kind` filters the listing only; it never changes what the model sees.
    """
    async with db.agent() as conn:
        entries = await store.load_cache(conn, connection_id=registered.id)
        disabled = await store.count_disabled(conn, connection_id=registered.id)
    # Staleness is a question about the business schema, so it is asked on the
    # other server — and on **this entry's** other server.
    async with db.target(registered.id) as conn:
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


@scoped.delete("/cache", response_model=ResetOut)
async def reset_cache(
    registered: store.Connection = Depends(connection_dep),
) -> ResetOut:
    """Forget everything learned about this connection. The stage recovery
    button (PLAN.md §9).

    Takes this connection's turn log and its sessions' checkpoints with it — see
    `store.reset_learned`. Another connection's cache is untouched and the
    registry row survives: "forget what you learned", not "forget the database".
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
    """The turn log as rows: what was asked, and what it cost. Ascending, so it
    reads left to right; `finished=false` adds the unfinished and failed turns.
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
