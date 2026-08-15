import contextlib
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app import config as config_module
from app import db, prompts, store, tracing
from app.api import router as v1
from app.config import config
from app.graph import build_graph
from app.settings import settings

log = logging.getLogger(__name__)

# Names that configure nothing; they live in config/config.yaml. Listed rather
# than dropped, because `extra="ignore"` makes a retired name and a typo
# identical — the value you set is silently not the one that applies.
RETIRED = {
    "PROVIDER",
    "MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_TIMEOUT",
    "OPENAI_MAX_TOKENS",
    "OPENAI_REASONING_EFFORT",
    "EFFORT_PLAN",
    "EFFORT_EXPLORE",
    "EFFORT_SQL",
    "EFFORT_EXTRACT",
    "MAX_TOOL_CALLS",
    "MAX_FIX_ATTEMPTS",
    "MAX_ROWS",
    "STATEMENT_TIMEOUT",
    "STATEMENT_TIMEOUT_MS",
    "PROMPT_DIR",
}


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # First and unguarded, so a bad config fails at boot rather than on the
    # first model call of a demo. Both are memoised for the life of the process.
    resolved = config()
    prompts.fingerprint()
    if (local := config_module.overlay()) is not None:
        # WARNING, like the four below: uvicorn configures handlers for its own
        # loggers only, so anything lower from here is dropped. The overlay is
        # untracked, so this server is not running what the repository says.
        log.warning(
            "%s is overlaying config.yaml — running %s/%s",
            local,
            resolved.model.provider,
            resolved.model.model,
        )

    agent_pool = await db.open_pools()
    # Checkpoints are the agent's own state, so they go on the agent's server —
    # never on the database it is answering questions about.
    checkpointer = AsyncPostgresSaver(agent_pool)
    await checkpointer.setup()  # idempotent; creates the checkpoint tables

    # Built once and shared. The graph, the pool and the checkpointer exist in
    # this process only, which is why the clients go through HTTP.
    app.state.graph = build_graph(checkpointer)

    # Best-effort: on a fresh clone `make up` runs before `make migrate` has
    # created the table.
    try:
        await db.ensure_default_connection()
    except Exception as e:
        log.warning("could not resolve the 'default' connection (%s) — run 'make migrate'", type(e).__name__)

    if not settings().api_token:
        # An unauthenticated DELETE on a connection's cache wipes everything the
        # agent has learned about it.
        log.warning("API_TOKEN is unset — /v1 is open and unauthenticated")

    retired = sorted(RETIRED & os.environ.keys())
    if retired:
        log.warning(
            "%s set and no longer read — these moved to %s. See its comments "
            "for the new key names; `statement_timeout_ms` is still an integer "
            "count of milliseconds.",
            ", ".join(retired),
            Path(settings().config_dir) / "config.yaml",
        )

    if not settings().connection_secret:
        log.warning(
            "CONNECTION_SECRET is unset — registered warehouse passwords are "
            "stored in plaintext on agent-db"
        )

    if tracing.enabled():
        # Turning this on copies every question, prompt and result row into
        # another datastore, which is the same kind of fact as "/v1 is open".
        log.warning(
            "tracing to Langfuse at %s — questions, SQL and result rows are captured",
            settings().langfuse_host,
        )
    elif settings().langfuse_public_key or settings().langfuse_secret_key:
        # Without this, a mistyped key name looks exactly like unconfigured.
        log.warning(
            "only one of LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY is set — "
            "tracing needs both and is off"
        )

    try:
        yield
    finally:
        # Before the pools: shutdown() flushes what is buffered.
        tracing.shutdown()
        await db.close_pools()


app = FastAPI(title="sql-agent", lifespan=lifespan)
app.include_router(v1)


@app.get("/health")
async def health() -> dict[str, str | int]:
    """Unversioned and unauthenticated, so a load balancer can reach it.

    Checks the agent's own database and not the registered targets: one
    warehouse being down does not stop this process serving every other one.
    Per-target reachability is POST /v1/connections/{id}/test.
    """
    async with db.agent() as conn:
        await (await conn.execute("SELECT 1 AS ok")).fetchone()
        registered = len(await store.list_connections(conn))
    return {"status": "ok", "agent": "ok", "connections": registered}
