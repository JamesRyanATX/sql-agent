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

**And a second business database**, `business_test_b`, because the agent now
holds a registry of targets rather than one. Cache scoping and cross-connection
fingerprinting are not testable against a single database: proving an entry
learned about one warehouse is invisible to another needs two warehouses. It is
a handful of tables built inline rather than a second `demo.sql` — that file is
slow and carries the trap-count contract, neither of which this needs.

Rebuilt once per session, so they are also always consistent with the current
`migrations/` and `demo/demo.sql`.
"""

import asyncio
import os
import pathlib
from dataclasses import dataclass
from typing import Any
from collections.abc import AsyncIterator

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from sqlalchemy.ext.asyncio import AsyncConnection as TargetConnection
from sqlalchemy.ext.asyncio import create_async_engine

from app import store, tracing
from app.main import app
from app.secrets import seal
from app.settings import Settings, settings

AGENT_TEST = "agent_test"
DEMO_TEST = "business_test"
DEMO_TEST_B = "business_test_b"
# The portable fixture's Postgres home. Separate from the demo databases, which
# carry the trap-count contract.
PORTABLE_PG = "portable_test"

# The registry ids the suite runs against. `default` is env-owned, exactly as it
# is in a real deployment, so tests exercise the same resolution path.
DEFAULT_CONNECTION = "default"
OTHER_CONNECTION = "other"

# `business_test_b`'s one table, deliberately the same *name* as the probe table
# tests/test_store.py creates on `business_test` and a different *shape*. That
# is what makes a crossed fingerprint detectable rather than plausible.
DEMO_B_SCHEMA = """
CREATE TABLE IF NOT EXISTS fp_probe (id bigint, created_at timestamptz);
CREATE TABLE IF NOT EXISTS widget (id bigint PRIMARY KEY, name text);
GRANT USAGE ON SCHEMA public TO reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO reader;
"""

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


def _register(conn, cid: str, url: str, *, origin: str) -> None:
    """Put one connection in the registry, upserting so a re-run repoints.

    Parsed with `store.connection_from_url`, which is what the lifespan uses —
    if the harness had its own parser the two could disagree about what a URL
    means, and the disagreement would show up as a test that passes against a
    database nobody intended.
    """
    row = store.connection_from_url(url, id=cid, origin=origin)
    conn.execute(
        """
        INSERT INTO connection (id, origin, host, port, database, username,
                                password, sslmode)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            origin = EXCLUDED.origin, host = EXCLUDED.host,
            port = EXCLUDED.port, database = EXCLUDED.database,
            username = EXCLUDED.username, password = EXCLUDED.password,
            sslmode = EXCLUDED.sslmode
        """,
        (row.id, row.origin, row.host, row.port, row.database, row.username,
         seal(row.password), row.sslmode),
    )


@pytest.fixture(scope="session", autouse=True)
def test_databases() -> str:
    dev = Settings()
    agent_url = _swap_db(dev.agent_database_url, AGENT_TEST)
    admin_url = _swap_db(dev.target_admin_url, DEMO_TEST)
    reader_url = _swap_db(dev.target_database_url, DEMO_TEST)
    admin_url_b = _swap_db(dev.target_admin_url, DEMO_TEST_B)
    reader_url_b = _swap_db(dev.target_database_url, DEMO_TEST_B)

    for live, test in (
        (dev.agent_database_url, agent_url),
        (dev.target_admin_url, admin_url),
        (dev.target_database_url, reader_url),
    ):
        assert live != test, f"already pointed at a test database: {test}"

    _recreate(dev.agent_database_url, AGENT_TEST)
    _recreate(dev.target_admin_url, DEMO_TEST)
    _recreate(dev.target_admin_url, DEMO_TEST_B)
    _recreate(dev.target_admin_url, PORTABLE_PG)

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

    # The second target. `reader` already exists by now — demo.sql created the
    # role, which is cluster-wide — so this only has to grant it access here.
    with psycopg.connect(admin_url_b, autocommit=True) as conn:
        conn.execute(f'GRANT CONNECT ON DATABASE "{DEMO_TEST_B}" TO reader')
        conn.execute(DEMO_B_SCHEMA)

    # Point everything at them *before* any app module resolves settings.
    os.environ["AGENT_DATABASE_URL"] = agent_url
    os.environ["TARGET_DATABASE_URL"] = reader_url
    os.environ["TARGET_ADMIN_URL"] = admin_url
    # Same reasoning as the databases: whatever is in the developer's .env must
    # not change what the suite tests. An API_TOKEN there would otherwise make
    # every /v1 call 401 on one machine and pass on another. The auth tests set
    # it explicitly.
    os.environ["API_TOKEN"] = ""
    # Same reasoning again: a developer's key must not change whether the suite
    # is testing the plaintext path or the encrypted one. test_connections.py
    # sets it explicitly for the tests that are about encryption.
    os.environ["CONNECTION_SECRET"] = ""
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

    # Seed the registry here rather than in the lifespan. test_store.py,
    # test_coldpath.py and test_isolation.py never run the lifespan — they open
    # pools directly or use raw psycopg — so a row created only there would not
    # exist for the tests that need it most.
    with psycopg.connect(agent_url, autocommit=True, row_factory=dict_row) as conn:
        _register(conn, DEFAULT_CONNECTION, reader_url, origin="env")
        _register(conn, OTHER_CONNECTION, reader_url_b, origin="api")

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
    url = store.connection_from_url(
        settings().target_database_url, id="_fixture"
    ).url()
    async for conn in _target_connect(url):
        yield conn


@pytest.fixture
async def target_conn_b() -> AsyncIterator[AsyncConnection]:
    """The second business database, as its owner. See `OTHER_CONNECTION`."""
    async for conn in _connect(_swap_db(settings().target_admin_url, DEMO_TEST_B)):
        yield conn


@pytest.fixture
async def reader_conn_b() -> AsyncIterator[TargetConnection]:
    """The second business database as the agent sees it."""
    url = store.connection_from_url(
        _swap_db(settings().target_database_url, DEMO_TEST_B), id="_fixture"
    ).url()
    async for conn in _target_connect(url):
        yield conn


# --------------------------------------------------------------- the dialects
#
# The dialect axis is **additive**: `default` and `other` stay exactly what they
# were — Postgres, same role, same shapes, same numbers — so every existing
# assertion keeps working as a regression test for the port itself, and a MySQL
# container being down cannot break the demo path.


# SQLite is free: a tempfile is a whole warehouse, no container and no role.
# MySQL needs one, so it is behind a marker the same way `live` is — see
# pyproject's addopts.
DIALECTS = ["postgresql", "sqlite", "mysql"]


def portable_url(dialect: str, tmp: pathlib.Path) -> str:
    """Always through `store.connection_from_url`, never a raw string.

    A bare `postgresql://` resolves to psycopg2 in SQLAlchemy, which is neither
    installed nor wanted — the normaliser is what maps it to
    `postgresql+psycopg`, and the harness using a different one from the app is
    exactly the disagreement `_register`'s docstring warns about.
    """
    if dialect == "sqlite":
        raw = f"sqlite:///{tmp / 'portable.db'}"
    elif dialect == "mysql":
        raw = os.environ.get(
            "MYSQL_TEST_URL", "mysql://root:root@localhost:3307/portable"
        )
    else:
        raw = _swap_db(Settings().target_admin_url, PORTABLE_PG)
    return (
        store.connection_from_url(raw, id="_portable")
        .url()
        .render_as_string(hide_password=False)
    )


@dataclass
class Portable:
    """One dialect's copy of the portable fixture, registered and connectable."""

    dialect: str
    cid: str
    url: str
    engine: Any  # a plain engine: no read-only hooks, so DDL still works


@pytest.fixture
async def portable(
    dialect, tmp_path_factory, agent_conn, client
) -> AsyncIterator[Portable]:
    """The portable fixture on `dialect`, plus a registry row pointing at it.

    Depends on `client` for its side effect: the lifespan is what opens the
    agent pool, and `db.target()` resolves the registry through it. One owner
    of the pool's lifetime beats each test opening and closing its own.

    Session-scoped would be faster, but a per-test build is what lets the
    capability tests attempt writes without leaving the next test a wrecked
    database.
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
    try:

        cid = f"portable-{dialect}"
        row = store.connection_from_url(url, id=cid, origin="api")
        await agent_conn.execute("DELETE FROM connection WHERE id = %s", (cid,))
        await store.create_connection(agent_conn, row)
        try:
            yield Portable(dialect=dialect, cid=cid, url=url, engine=engine)
        finally:
            from app import db as _db

            await _db.evict(cid)
            await agent_conn.execute("DELETE FROM connection WHERE id = %s", (cid,))
    finally:
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
