"""Phase 1: the five traps demonstrably bite.

Each test asserts the *naive* query returns a different answer from the correct
one. That's the point — if a trap doesn't bite, T1 is a formality and there's no
recipe worth reading aloud in beat 2 of the demo.

These are also the regression net for `demo/demo.sql`: the schema and the seed
used to be a migration plus a Python script, and the counts they produce (1,840
active, 460 west, 750 cancelled) are the demo's contract.

Requires `make up && make migrate && make seed`.
"""

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncConnection


@pytest.fixture
def conn(reader_conn):
    """The demo database as the agent sees it — SELECT and nothing else."""
    return reader_conn


async def scalar(conn: AsyncConnection, sql: str):
    """`text()` is fine here — unlike the model's SQL, these strings are ours."""
    return (await conn.execute(text(sql))).scalar_one()


# ---------------------------------------------------------------- trap 1


async def test_soft_deletes(conn):
    """Fires on the most obvious question anyone can ask, which is what makes
    T1 a real investigation rather than a formality."""
    naive = await scalar(conn, "SELECT count(*) FROM customer")
    correct = await scalar(
        conn, "SELECT count(*) FROM customer WHERE deleted_at IS NULL"
    )

    assert naive == 2_000
    assert correct == 1_840, "the answer to beat 1 of the demo is fixed"
    assert naive != correct

    # And it's discoverable by sampling, which is how the agent is meant to
    # infer it rather than being told.
    non_null = await scalar(
        conn, "SELECT count(*) FROM customer WHERE deleted_at IS NOT NULL"
    )
    assert non_null == 160


# ---------------------------------------------------------------- trap 2


async def test_region_casing(conn):
    naive = await scalar(conn, "SELECT count(*) FROM customer WHERE region = 'west'")
    correct = await scalar(
        conn, "SELECT count(*) FROM customer WHERE lower(region) = 'west'"
    )

    assert naive < correct, "equality on region must under-count"
    assert (naive, correct) == (350, 500)

    # Grouping without normalising invents regions that don't exist.
    raw_groups = await scalar(conn, "SELECT count(DISTINCT region) FROM customer")
    norm_groups = await scalar(
        conn, "SELECT count(DISTINCT lower(region)) FROM customer"
    )
    assert raw_groups > norm_groups
    assert norm_groups == 4


async def test_region_and_soft_delete_compound(conn):
    """Beat 4 of the demo composes both recipes. Getting either wrong is visible."""
    both_wrong = await scalar(
        conn, "SELECT count(*) FROM customer WHERE region = 'west'"
    )
    both_right = await scalar(
        conn,
        "SELECT count(*) FROM customer "
        "WHERE lower(region) = 'west' AND deleted_at IS NULL",
    )
    assert both_wrong == 350
    assert both_right == 460
    assert both_wrong != both_right


# ---------------------------------------------------------------- trap 3


async def test_cancelled_orders_inflate_revenue(conn):
    including = await scalar(
        conn,
        "SELECT SUM(oi.qty * oi.price) FROM order_item oi JOIN orders o ON o.id = oi.order_id",
    )
    excluding = await scalar(
        conn,
        "SELECT SUM(oi.qty * oi.price) FROM order_item oi JOIN orders o ON o.id = oi.order_id "
        "WHERE o.status <> 'cancelled'",
    )

    assert including > excluding, "cancelled orders must overstate revenue"
    # Material, not a rounding difference — worth a recipe.
    assert (including - excluding) / including > 0.05

    # Discoverable by sampling the column.
    statuses = await scalar(conn, "SELECT count(DISTINCT status) FROM orders")
    assert statuses >= 4
    cancelled = await scalar(
        conn, "SELECT count(*) FROM orders WHERE status = 'cancelled'"
    )
    assert cancelled == 750


# ---------------------------------------------------------------- trap 4


async def test_historical_price_differs_from_current_unit_price(conn):
    """`order_item.price` is what it sold for. `product.unit_price` is what it
    costs today. Joining to the latter to compute revenue is silently wrong."""
    actual = await scalar(
        conn, "SELECT SUM(oi.qty * oi.price) FROM order_item oi"
    )
    naive = await scalar(
        conn,
        "SELECT SUM(oi.qty * p.unit_price) FROM order_item oi "
        "JOIN product p ON p.id = oi.product_id",
    )

    assert naive != actual
    assert naive > actual, "older sales went out below today's list price"
    assert abs(naive - actual) / actual > 0.05

    rows_that_differ = await scalar(
        conn,
        "SELECT count(*) FROM order_item oi JOIN product p ON p.id = oi.product_id "
        "WHERE oi.price <> p.unit_price",
    )
    assert rows_that_differ > 0


async def test_the_price_trap_fires_within_the_last_quarter(conn):
    """Regression: the divergence used to apply only to orders over a year old,
    so beat 5 — "revenue by region last quarter" — could join to
    product.unit_price and get the right answer by luck. The demo's flagship
    revenue question has to be able to get this wrong."""
    window = (
        "o.created >= date_trunc('quarter', CURRENT_DATE)::date - INTERVAL '3 months' "
        "AND o.created < date_trunc('quarter', CURRENT_DATE)::date"
    )
    actual = await scalar(
        conn,
        "SELECT SUM(oi.qty * oi.price) FROM order_item oi "
        f"JOIN orders o ON o.id = oi.order_id WHERE {window}",
    )
    naive = await scalar(
        conn,
        "SELECT SUM(oi.qty * p.unit_price) FROM order_item oi "
        "JOIN orders o ON o.id = oi.order_id JOIN product p ON p.id = oi.product_id "
        f"WHERE {window}",
    )
    assert actual is not None and naive != actual
    assert abs(naive - actual) / actual > 0.03, "must be material, not a rounding error"


# ---------------------------------------------------------------- trap 5


async def test_orders_has_created_not_created_at(conn):
    """Every model reaches for `created_at` first. It isn't there."""
    # Through reflection, which is what `app/tools.py` uses — so this asserts
    # what the agent would actually be told, not what a catalog query says.
    columns = {
        c["name"]
        for c in await conn.run_sync(
            lambda sync: inspect(sync).get_columns("orders")
        )
    }
    assert "created" in columns
    assert "created_at" not in columns

    # The failure is a hard error, so the fix node has something to react to.
    # ProgrammingError is SQLAlchemy's wrapper; the driver's own class is on
    # `.orig`, and `graph.driver_message` unwraps to it for exactly this reason.
    # No savepoint needed any more: the target engine runs AUTOCOMMIT, so each
    # statement is its own transaction and a failed one leaves `conn` usable.
    # That is the same property the psycopg pool had, kept deliberately —
    # `explore` must not hold a transaction open across a model round trip.
    with pytest.raises(ProgrammingError) as e:
        await conn.execute(text("SELECT max(created_at) FROM orders"))
    assert type(e.value.orig).__name__ == "UndefinedColumn"

    assert await scalar(conn, "SELECT 1"), "the connection survived the error"

    assert await scalar(conn, "SELECT count(*) FROM orders WHERE created IS NOT NULL")


# ---------------------------------------------------------------- the search


async def test_exploration_has_to_actually_search(conn):
    """Wide, not deep. If the relevant tables were obvious, T1 would be cheap
    and there'd be no delta to sell."""
    total = await scalar(
        conn,
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'",
    )
    assert total >= 40

    # Decoys are populated. An empty table is dismissed in one tool call; a
    # populated one has to be looked at.
    for table in ("app_user", "cart", "invoice", "payment", "price_history"):
        assert await scalar(conn, f"SELECT count(*) FROM {table}") > 0, table

    # The most tempting decoy: it has the soft-delete column, and it is not
    # the customer table.
    app_user_active = await scalar(
        conn, "SELECT count(*) FROM app_user WHERE deleted_at IS NULL"
    )
    assert app_user_active != 1_840
