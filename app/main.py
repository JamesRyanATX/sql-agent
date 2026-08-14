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

# Environment variables that used to configure something and now do not. They
# moved into config/config.yaml when "which model, how hard, within what
# bounds" stopped being one flat namespace with the database URLs and the
# secrets. Kept as a list rather than deleted quietly, because `extra="ignore"`
# makes a retired name and a typo'd name look identical from in here.
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
    # First, and deliberately unguarded: read the config and the prompts. Both
    # are memoised for the life of the process, so the alternative to failing
    # here is failing on the first model call of the first turn — which on a
    # demo is the worst possible moment to discover that CONFIG_DIR is wrong or
    # that a prompt file is empty. Boot is where a config error belongs.
    resolved = config()
    prompts.fingerprint()
    if (local := config_module.overlay()) is not None:
        # A warning rather than an info, for the reason the tracing line below
        # is one: uvicorn configures handlers for its own loggers only, so
        # anything below WARNING from here is dropped, and a startup line nobody
        # sees is not information. And this one earns it — the overlay is
        # untracked, so the model actually answering questions is not the one
        # config/config.yaml says ships, and that should never be a surprise.
        log.warning(
            "%s is overlaying config.yaml — running %s/%s",
            local,
            resolved.model.provider,
            resolved.model.model,
        )

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

    # `Settings` has extra="ignore", so a retired name is not an error — it is
    # silently dropped and the default quietly applies. Every one of these used
    # to configure something, and a value that stopped being the one you set
    # should say so. `config/config.yaml` is where they went; the two API keys
    # stayed in the environment, because a tracked file must not hold a secret.
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
        # The same bargain: an insecure default is fine when it is loud.
        log.warning(
            "CONNECTION_SECRET is unset — registered warehouse passwords are "
            "stored in plaintext on agent-db"
        )

    if tracing.enabled():
        # A warning and not an info, for both of the reasons the three below are:
        # turning this on starts copying every question, prompt and result row
        # into another datastore, which is the same kind of fact as "/v1 is
        # open" — and uvicorn configures handlers for its own loggers only, so
        # anything this module logs below WARNING reaches the root logger's
        # last-resort handler and is dropped. An informational line nobody sees
        # is not information.
        log.warning(
            "tracing to Langfuse at %s — questions, SQL and result rows are captured",
            settings().langfuse_host,
        )
    elif settings().langfuse_public_key or settings().langfuse_secret_key:
        # `Settings` has extra="ignore", so a mistyped key name is dropped in
        # silence and half-configured looks exactly like unconfigured: nothing
        # in the UI, nothing in the log, and no reason given.
        log.warning(
            "only one of LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY is set — "
            "tracing needs both and is off"
        )

    try:
        yield
    finally:
        # Before the pools: shutdown() flushes what is buffered, and a trace of
        # the last turn is worth more than a few milliseconds of teardown.
        tracing.shutdown()
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
