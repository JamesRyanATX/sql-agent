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
from dataclasses import dataclass
from typing import Any

import psycopg
import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from sqlalchemy.ext.asyncio import AsyncConnection as TargetConnection
from sqlalchemy.ext.asyncio import create_async_engine

from app import config as config_module
from app import db, store, tracing
from app.config import config, overrides
from app.main import app
from app.settings import Settings, settings

AGENT_TEST = "agent_test"
DEMO_TEST = "business_test"
# The portable fixture's Postgres home. Separate from the demo databases, which
# carry the trap-count contract.
PORTABLE_PG = "portable_test"


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
    admin_url = _swap_db(dev.test_admin_url, DEMO_TEST)
    reader_url = _swap_db(dev.target_database_url, DEMO_TEST)

    for live, test in (
        (dev.agent_database_url, agent_url),
        (dev.test_admin_url, admin_url),
        (dev.target_database_url, reader_url),
    ):
        assert live != test, f"already pointed at a test database: {test}"

    _recreate(dev.agent_database_url, AGENT_TEST)
    _recreate(dev.test_admin_url, DEMO_TEST)
    _recreate(dev.test_admin_url, PORTABLE_PG)

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
    os.environ["TEST_ADMIN_URL"] = admin_url
    # Same reasoning as the databases: whatever is in the developer's .env must
    # not change what the suite tests. An API_TOKEN there would otherwise make
    # every /v1 call 401 on one machine and pass on another. The auth tests set
    # it explicitly.
    os.environ["API_TOKEN"] = ""
    # And again: a developer's Langfuse keys in .env must not make the suite ship
    # their prompts to their real project — or hang teardown on an exporter
    # retrying against a stack that isn't running. test_tracing.py sets them
    # explicitly for the tests that are about tracing.
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
    settings.cache_clear()
    assert settings().agent_database_url == agent_url
    assert settings().target_database_url == reader_url
    assert settings().api_token == ""
    assert not tracing.enabled()

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
    async for conn in _connect(settings().test_admin_url):
        yield conn


async def _target_connect(url: str) -> AsyncIterator[TargetConnection]:
    """A SQLAlchemy connection to a target, as `app/` gets one.

    **No read-only hooks.** `app.dialects.install` is deliberately not applied:
    `test_the_reader_role_cannot_write` is about the *role*, and a connection
    that refused writes at the session level would fail with
    ReadOnlySqlTransaction instead of InsufficientPrivilege and quietly stop
    testing what its docstring says it tests.
    """
    engine = create_async_engine(url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            yield conn
    finally:
        await engine.dispose()


@pytest.fixture
async def reader_conn() -> AsyncIterator[TargetConnection]:
    """The business data as the *agent* sees it: SELECT and nothing else.

    A SQLAlchemy connection now, like the one `app/tools.py` is handed — the
    port's whole point is that this is not interchangeable with `agent_conn`.
    """
    # Through db.target_url, which is what normalises a bare `postgresql://`
    # onto the driver that is installed.
    async for conn in _target_connect(db.target_url()):
        yield conn


DIALECTS = ["postgresql", "sqlite", "mysql"]


def portable_url(dialect: str, tmp: pathlib.Path) -> str:
    """A driver-qualified URL for one dialect.

    Spelled out rather than normalised: `app/db.py` maps a bare scheme onto the
    driver that is installed, and a harness quietly relying on that would not
    notice the day the mapping changed.
    """
    if dialect == "sqlite":
        return f"sqlite+aiosqlite:///{tmp / 'portable.db'}"
    if dialect == "mysql":
        return os.environ.get(
            "MYSQL_TEST_URL", "mysql+asyncmy://root:root@localhost:3307/portable"
        )
    return _swap_db(Settings().test_admin_url, PORTABLE_PG).replace(
        "postgresql://", "postgresql+psycopg://", 1
    )


@dataclass
class Portable:
    """One dialect's copy of the portable fixture, on two engines.

    `engine` is plain: no read-only hooks, so the fixture can build itself and
    the reflection tests can read. `guarded` is what `app/db.py` builds for the
    target — AUTOCOMMIT plus `dialects.install` — and is what the capability
    tests attempt writes through. Two engines because the guard is a connect
    hook: applying it to the engine that just created the tables would only
    affect connections opened afterwards, which is a confusing way to test a
    claim about refusals.
    """

    dialect: str
    url: str
    engine: Any
    guarded: Any


@pytest.fixture
async def portable(dialect, tmp_path_factory) -> AsyncIterator[Portable]:
    """The same handful of tables, built on whichever engine `dialect` names.

    The engine carries `dialects.install`, exactly as `app/db.py` applies it to
    the one target — which is what lets the capability tests check the claims in
    `app/dialects.py` on every engine while the app is pointed at Postgres.

    Session-scoped would be faster, but a per-test build is what lets those
    tests attempt writes without leaving the next test a wrecked database.
    """
    from tests.fixtures import portable as fixture

    tmp = tmp_path_factory.mktemp(f"portable-{dialect}")
    url = portable_url(dialect, tmp)
    try:
        engine = create_async_engine(url)
        await fixture.build(engine)
    except Exception as e:  # noqa: BLE001
        # MySQL is opt-in: the driver is an optional extra and the container is
        # behind a compose profile, so "not available" is a skip rather than a
        # failure. Every other dialect not building is a real failure.
        if dialect == "mysql":
            pytest.skip(f"no MySQL available ({type(e).__name__}: {e})")
        raise
    from app import dialects
    from app.config import config

    guarded = create_async_engine(url, isolation_level="AUTOCOMMIT")
    dialects.install(guarded, dialect, config().statement_timeout_ms)
    try:
        yield Portable(dialect=dialect, url=url, engine=engine, guarded=guarded)
    finally:
        await guarded.dispose()
        await engine.dispose()


@pytest.fixture(params=DIALECTS)
def dialect(request) -> str:
    """Parametrises a test over every supported engine.

    A test using this runs three times. MySQL skips itself unless the container
    is up; SQLite always runs, which is the point of choosing it first.
    """
    name = request.param
    if name == "mysql":
        request.node.add_marker(pytest.mark.mysql)
    return name


@pytest.fixture(autouse=True)
def cli_env(monkeypatch, tmp_path):
    """The CLI's two environment variables, and a state file per test.

    Autouse because `sql_agent.config` reads the real environment and the
    real `~/.local/state`, and a suite that wrote to a developer's actual
    selected connection would be rude as well as flaky. Tests about a *missing*
    variable delete it themselves.
    """
    monkeypatch.setenv("SQL_AGENT_URL", "http://test/v1")
    monkeypatch.delenv("SQL_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("SQL_AGENT_CONNECTION", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


# What config.yaml says, minus what these tests are not about.
DEMO = {
    "model": {"provider": "anthropic", "model": "claude-opus-5"},
    "max_tool_calls": 24,
    "extract": {"effort": "low"},
    "explore": {"effort": "high"},
}


def clear_config() -> None:
    """Both memoised readers. Always both: an `overrides()` that outlives the
    `config()` it described would mark keys the running config never saw."""
    config.cache_clear()
    overrides.cache_clear()


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """A scratch CONFIG_DIR. `write` puts a config.yaml or an overlay in it.

    Request it *before* `client` in a test's signature: the lifespan reads
    `config()` at boot, and what it caches is whatever CONFIG_DIR said then.
    """

    def write(data: dict, *, local: dict | None = None) -> None:
        (tmp_path / config_module.FILE).write_text(yaml.safe_dump(data))
        if local is not None:
            (tmp_path / config_module.LOCAL).write_text(yaml.safe_dump(local))
        clear_config()

    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))
    settings.cache_clear()
    clear_config()
    write(DEMO)
    yield write
    monkeypatch.delenv("CONFIG_DIR", raising=False)
    settings.cache_clear()
    clear_config()


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
