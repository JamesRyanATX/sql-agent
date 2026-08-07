"""A dedicated pair of databases for the test suite.

Tests and live runs were sharing one database, and it went wrong four separate
ways before this file existed: a suite run `TRUNCATE`d the cache mid-experiment;
an assertion that the cache was empty failed once anything had used it; a test
entry named `active customer` collided with one the model had genuinely learned
(the unique index sent the write down `DO UPDATE`, which leaves `created_turn`
alone, so scoped queries silently missed it); and finally the `plan` node
started branching on whatever happened to be cached.

Each of those got a local fix. They were all the same bug. The cache is global
state that the graph reads in full, so no amount of per-test scoping makes a
shared database safe — the suite needs its own.

**Two of them, now.** The agent's memory and the data it queries live on
separate servers, so the suite mirrors that: `agent_test` on agent-db,
`business_test` on demo-db. A test that could not tell them apart would not be
testing the thing this split exists for.

Rebuilt once per session, so they are also always consistent with the current
`migrations/` and `demo/demo.sql`.
"""

import asyncio
import os
import pathlib
from collections.abc import AsyncIterator

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from app.main import app
from app.settings import Settings, settings

AGENT_TEST = "agent_test"
DEMO_TEST = "business_test"

ROOT = pathlib.Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "migrations"
DEMO_SQL = ROOT / "demo" / "demo.sql"


def _swap_db(url: str, name: str) -> str:
    base, _, _ = url.rpartition("/")
    return f"{base}/{name}"


async def _setup_checkpoints(url: str) -> None:
    """Create LangGraph's checkpoint tables, as `app.main`'s lifespan does."""
    async with await psycopg.AsyncConnection.connect(
        url, autocommit=True, row_factory=dict_row
    ) as conn:
        await AsyncPostgresSaver(conn).setup()


def _recreate(admin_url: str, name: str) -> None:
    """Drop and recreate `name`, connecting through `admin_url`'s database.

    Fresh every session: the suite should never inherit yesterday's rows, and
    this keeps it honest about the migrations actually applying.
    """
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{name}"')


@pytest.fixture(scope="session", autouse=True)
def test_databases() -> str:
    dev = Settings()
    agent_url = _swap_db(dev.agent_database_url, AGENT_TEST)
    admin_url = _swap_db(dev.target_admin_url, DEMO_TEST)
    reader_url = _swap_db(dev.target_database_url, DEMO_TEST)

    for live, test in (
        (dev.agent_database_url, agent_url),
        (dev.target_admin_url, admin_url),
        (dev.target_database_url, reader_url),
    ):
        assert live != test, f"already pointed at a test database: {test}"

    _recreate(dev.agent_database_url, AGENT_TEST)
    _recreate(dev.target_admin_url, DEMO_TEST)

    # The agent's own schema, then LangGraph's checkpoint tables — which the
    # app creates in its lifespan, not in a migration. Without this the agent
    # test database is missing four of its six tables until some test happens
    # to run the lifespan first, which makes anything that counts them
    # order-dependent.
    with psycopg.connect(agent_url, autocommit=True) as conn:
        for path in sorted(MIGRATIONS.glob("*.sql")):
            conn.execute(path.read_text())
    asyncio.run(_setup_checkpoints(agent_url))

    # The demo database, whole — role, schema and data in one file. Applied as
    # the owner, which is what `demo.sql`'s ALTER DEFAULT PRIVILEGES needs.
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(DEMO_SQL.read_text())

    # Point everything at them *before* any app module resolves settings.
    os.environ["AGENT_DATABASE_URL"] = agent_url
    os.environ["TARGET_DATABASE_URL"] = reader_url
    os.environ["TARGET_ADMIN_URL"] = admin_url
    # Same reasoning as the databases: whatever is in the developer's .env must
    # not change what the suite tests. An API_TOKEN there would otherwise make
    # every /v1 call 401 on one machine and pass on another. The auth tests set
    # it explicitly.
    os.environ["API_TOKEN"] = ""
    settings.cache_clear()
    assert settings().agent_database_url == agent_url
    assert settings().target_database_url == reader_url
    assert settings().api_token == ""

    yield agent_url

    settings.cache_clear()


# ------------------------------------------------------------------ connections
#
# Autocommit, because the endpoints and the graph read on their own pooled
# connections and would not see an open transaction's writes. Tests clean up
# after themselves.


async def _connect(url: str) -> AsyncIterator[AsyncConnection]:
    async with await psycopg.AsyncConnection.connect(
        url, row_factory=dict_row, autocommit=True
    ) as conn:
        yield conn


@pytest.fixture
async def agent_conn() -> AsyncIterator[AsyncConnection]:
    """The agent's memory: cache_entry, turn, checkpoints."""
    async for conn in _connect(settings().agent_database_url):
        yield conn


@pytest.fixture
async def target_conn() -> AsyncIterator[AsyncConnection]:
    """The business data, **as its owner** — tests need DDL and writes.

    This is not the connection the app uses. For that, see `reader_conn`.
    """
    async for conn in _connect(settings().target_admin_url):
        yield conn


@pytest.fixture
async def reader_conn() -> AsyncIterator[AsyncConnection]:
    """The business data as the *agent* sees it: SELECT and nothing else."""
    async for conn in _connect(settings().target_database_url):
        yield conn


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """The app, over ASGI. No server and no container — the routes run in-process.

    Runs the lifespan, so `app.state.graph`, both pools and the checkpointer
    exist exactly as they do in production.
    """
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
