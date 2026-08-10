"""The agent's memory and the data it queries cannot reach each other.

This used to be two conventions in application code: a name filter in
`app.tools` that hid `cache_entry` from introspection, and the
`transaction_read_only` flag in `db.readonly()`. Both were one forgotten call
site from not being true, and nothing failed loudly when they weren't.

Now it is two servers and a role that holds SELECT. These tests are what say so
— in particular `test_the_agent_cannot_see_its_own_memory`, which replaces the
version that only proved a Python set was doing its job.

**One of those halves came back.** With a registered connection the credentials
are whatever the user handed us, so "the role holds SELECT" is a hope, not a
property. `test_a_user_supplied_dsn_still_cannot_write` is the one that says the
guarantee moved rather than evaporated: every target pool opens its connections
in a read-only session, which binds any role including a superuser.

Requires `make up && make migrate && make seed`.
"""

import psycopg
import pytest
from psycopg import AsyncConnection
from psycopg.errors import InsufficientPrivilege  # noqa: F401 — pg-only tests below
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from app import store
from app.settings import settings
from app.tools import ToolError, describe_table, list_tables

AGENT_TABLES = ("cache_entry", "turn", "checkpoints", "checkpoint_blobs")
DEMO_TABLES = ("customer", "orders", "order_item", "product")


async def table_names(conn: AsyncConnection) -> set[str]:
    cur = await conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    )
    return {r["table_name"] for r in await cur.fetchall()}


# ------------------------------------------------------- the databases differ


async def test_the_two_databases_share_no_tables(agent_conn, target_conn):
    """The whole feature in one assertion."""
    agent, demo = await table_names(agent_conn), await table_names(target_conn)

    assert agent & demo == set()
    assert set(AGENT_TABLES) <= agent
    assert set(DEMO_TABLES) <= demo


async def test_they_are_not_even_the_same_cluster(agent_conn, target_conn):
    """Two databases on one server would satisfy the test above and still share
    roles, a superuser, and a disk.

    `system_identifier` is stamped by initdb and unique per cluster, so this is
    the honest check. Ports are not: both containers listen on 5432 internally,
    and `inet_server_port()` reports that — it would pass while proving nothing.
    """
    agent = await (
        await agent_conn.execute("SELECT system_identifier AS id FROM pg_control_system()")
    ).fetchone()
    demo = await (
        await target_conn.execute("SELECT system_identifier AS id FROM pg_control_system()")
    ).fetchone()
    assert agent["id"] != demo["id"]


async def test_crossing_the_urls_fails_authentication(agent_conn):
    """Different credentials per server, so a copy-pasted URL fails at connect
    rather than connecting and failing later on a missing table — which reads
    like a bug in the agent instead of a bug in the config."""
    agent_creds = settings().agent_database_url.rsplit("@", 1)[0]
    demo_host = settings().target_admin_url.rsplit("@", 1)[1]

    with pytest.raises(psycopg.OperationalError):
        await psycopg.AsyncConnection.connect(f"{agent_creds}@{demo_host}")


# --------------------------------------------------- the agent's own view


async def test_the_agent_cannot_see_its_own_memory(reader_conn):
    """Otherwise it explores the cache, caches facts about the cache, and beat 2
    of the demo fills up with noise.

    Nothing filters these names any more. They are absent because they are on
    another server, which is a much harder thing to accidentally undo.
    """
    listed = await list_tables(reader_conn)
    names = {t.split(" ")[0] for t in listed["tables"]}

    assert names.isdisjoint(AGENT_TABLES)
    assert "customer" in names

    for hidden in AGENT_TABLES:
        with pytest.raises(ToolError, match="no such table"):
            await describe_table(reader_conn, hidden)


async def test_a_decoy_named_like_agent_state_is_still_visible(
    reader_conn, target_conn
):
    """The inverse of the above, and the reason the old filter had to go.

    A table called `cache_entry` on the *business* side is business data — a
    caching table someone's application owns. The name filter hid it, because it
    could only match on the name. Isolation by server does not have that
    problem: what matters is which database it is in.
    """
    await target_conn.execute("CREATE TABLE IF NOT EXISTS cache_entry (id int, note text)")
    try:
        described = await describe_table(reader_conn, "cache_entry")
        assert {c["name"] for c in described["columns"]} == {"id", "note"}
    finally:
        await target_conn.execute("DROP TABLE IF EXISTS cache_entry")


# ------------------------------------------------------ the role cannot write


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param("INSERT INTO customer (name, signed_up) VALUES ('x', now())", id="insert"),
        pytest.param("UPDATE customer SET name = 'x'", id="update"),
        pytest.param("DELETE FROM customer", id="delete"),
        pytest.param("CREATE TABLE evil (i int)", id="create"),
        pytest.param("TRUNCATE customer", id="truncate"),
    ],
)
async def test_the_reader_role_cannot_write(reader_conn, statement):
    """The read-only role's whole point, and nothing else checks it.

    These run *outside* `target_readonly()` deliberately — the transaction guard
    is not what is being tested. If someone deleted that guard tomorrow, these
    would still pass, and that is the property worth having.
    """
    # ProgrammingError is SQLAlchemy's wrapper — the driver's own class is on
    # `.orig`, and asserting on that is what keeps this about the *role* rather
    # than about whichever exception hierarchy is in the middle.
    with pytest.raises(ProgrammingError) as e:
        await reader_conn.execute(text(statement))
    assert type(e.value.orig).__name__ == "InsufficientPrivilege"


async def test_the_reader_role_can_still_read_everything_it_needs(reader_conn):
    """Read-only has to stay genuinely useful, or exploration degrades quietly.

    Constraint discovery is the one that nearly broke:
    `information_schema.table_constraints` only shows constraints to a caller
    with a *non-SELECT* privilege, so `describe_table` had to move to
    pg_catalog. Without it every table looks keyless and the agent guesses joins
    from column names.
    """
    described = await describe_table(reader_conn, "order_item")
    assert described["rows"] == 18_000
    assert set(described["primary_key"]) == {"order_id", "product_id"}
    assert described["foreign_keys"] == [
        "order_id -> orders.id",
        "product_id -> product.id",
    ]


# ------------------------------------------------------------ reset's reach


async def test_reset_cannot_reach_the_business_data(agent_conn, target_conn):
    """`reset_learned` used to be safe because it named six tables. Now it is
    safe because the connection it runs on cannot see any others."""
    before = await table_names(target_conn)
    counts = {}
    for table in ("customer", "orders"):
        cur = await target_conn.execute(f"SELECT count(*) AS n FROM {table}")
        counts[table] = (await cur.fetchone())["n"]

    await store.reset_everything(agent_conn)

    assert await table_names(target_conn) == before
    for table, n in counts.items():
        cur = await target_conn.execute(f"SELECT count(*) AS n FROM {table}")
        assert (await cur.fetchone())["n"] == n




# ------------------------------------------------ a target we did not choose


async def test_a_user_supplied_dsn_still_cannot_write(client, agent_conn):
    """Register credentials that genuinely *can* write, and watch them not.

    `demo/demo.sql` builds a SELECT-only role, so for the built-in connection
    the session guard was a second line of defence. A registered connection has
    no such promise behind it — the credentials are whatever the user handed us.
    This is the test that would fail if somebody removed the `dialects.install`
    call in app/db.py, which is otherwise five lines with nothing pointing at
    them.

    Postgres claims the `enforced` tier, so both halves are asserted here. What
    a weaker dialect claims, and whether it says so, is
    tests/test_capabilities.py.
    """
    from sqlalchemy.engine import make_url

    from app import db, dialects

    owner = make_url(
        store.connection_from_url(settings().target_admin_url, id="_owner")
        .url()
        .render_as_string(hide_password=False)
    )
    resp = await client.post(
        "/v1/connections",
        params={"probe": "false"},
        json={
            "id": "owner",
            "host": owner.host,
            "port": owner.port,
            "database": owner.database,
            "username": owner.username,
            "password": owner.password,
        },
    )
    assert resp.status_code == 201, resp.text

    cap = dialects.for_dialect("postgresql")
    assert cap.tier == "enforced"
    try:
        # These credentials own the database. Prove it, so the assertions below
        # are about the guard and not about a role that could never have written.
        probe = (await client.post("/v1/connections/owner/test")).json()
        assert probe["read_only"] is False

        # The plain connection — what `explore`, `infer_tables` and `extract`
        # use, and the one with no transaction of its own to guard.
        async with db.target("owner") as conn:
            statement, expected = cap.probe
            assert (await conn.exec_driver_sql(statement)).scalar_one() == expected
            with pytest.raises(Exception) as e:
                await conn.exec_driver_sql("DELETE FROM customer")
            assert dialects.is_read_only_error(e.value)

        # And where the generated SQL runs.
        async with db.target_readonly("owner") as conn:
            with pytest.raises(Exception) as e:
                await conn.exec_driver_sql("DELETE FROM customer")
            assert dialects.is_read_only_error(e.value)

        # Nothing was deleted by any of that. On a fresh connection, not on one
        # already handed back to the pool — the previous version of this test
        # read from a closed one and worked by luck.
        async with db.target("owner") as conn:
            n = (
                await conn.exec_driver_sql("SELECT count(*) FROM customer")
            ).scalar_one()
        assert n == 2000
    finally:
        await agent_conn.execute("DELETE FROM connection WHERE id = 'owner'")
        await db.evict("owner")
