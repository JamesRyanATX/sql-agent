"""A dedicated database for the test suite.

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

Rebuilt once per session, so it is also always consistent with the current
migrations and seed.
"""

import asyncio
import os
import pathlib
from collections.abc import AsyncIterator

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.settings import Settings, settings

TEST_DB = "sql_agent_test"
MIGRATIONS = pathlib.Path(__file__).resolve().parent.parent / "migrations"


def _split(url: str) -> tuple[str, str]:
    base, _, name = url.rpartition("/")
    return base, name


@pytest.fixture(scope="session", autouse=True)
def test_database() -> str:
    dev_url = Settings().database_url
    base, dev_name = _split(dev_url)
    assert dev_name != TEST_DB, "already pointed at the test database"
    test_url = f"{base}/{TEST_DB}"

    with psycopg.connect(dev_url, autocommit=True) as conn:
        # Fresh every session: the suite should never inherit yesterday's rows,
        # and this keeps it honest about the migrations actually applying.
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{TEST_DB}"')

    with psycopg.connect(test_url, autocommit=True) as conn:
        for path in sorted(MIGRATIONS.glob("*.sql")):
            conn.execute(path.read_text())

    # Point everything at it *before* any app module resolves settings.
    os.environ["DATABASE_URL"] = test_url
    # Same reasoning as the database: whatever is in the developer's .env must
    # not change what the suite tests. An API_TOKEN there would otherwise make
    # every /v1 call 401 on one machine and pass on another. The auth tests set
    # it explicitly.
    os.environ["API_TOKEN"] = ""
    settings.cache_clear()
    assert settings().database_url == test_url
    assert settings().api_token == ""

    from scripts.seed import main as seed

    asyncio.run(seed())

    yield test_url

    settings.cache_clear()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """The app, over ASGI. No server and no container — the routes run in-process.

    Runs the lifespan, so `app.state.graph`, the pool and the checkpointer exist
    exactly as they do in production.
    """
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
