"""A tiny business database that exists on every supported dialect.

Deliberately **not** in `demo/`. `demo/demo.sql` is 573 lines of
`generate_series`, `::interval`, `ARRAY[...][i]`, `DO $$` role creation and
`ALTER DEFAULT PRIVILEGES`, and it produces eight numbers — 2,000 customers,
1,840 active, 350-vs-500 west — that six test files assert on and that the demo
is gated on. A second dialect's copy would fork that contract into two things
that drift, and its own header says a translation that quietly changed them
would be found on stage. So the demo stays Postgres and this is a separate,
much smaller thing whose only job is to prove the abstraction is real.

Built from a SQLAlchemy `MetaData`, so it is portable by construction rather
than by three careful translations. Every shape here earns its place:

* `ledger_item` has a **composite primary key** and a **composite foreign key**.
  The Postgres implementation needed a `JOIN LATERAL unnest(conkey) WITH
  ORDINALITY` specifically to pair a composite FK's columns with the right
  referenced ones; the reflection port replaced that with a `zip`, so it needs a
  test that would notice if the pairing came back crossed.
* `order` is a **reserved word** in every dialect. Hand-written SQL never had to
  quote it because the demo has no such table; `quoted_name` does, and
  `list_tables` has to render it.
* `region` carries the demo's casing trap in miniature, so `count_distinct` can
  be checked for folding values together that should stay apart.
* `archived_at` is nullable and mostly null, so `sample_column`'s null count —
  the load-bearing part, the thing that turns a column into a convention — has
  something to count.
"""

from __future__ import annotations

import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKeyConstraint,
    Integer,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Table,
    insert,
)

metadata = MetaData()

# `order` is reserved everywhere. SQLAlchemy quotes it because the name is a
# quoted_name by the time it reaches a compiler; that is the property under test.
order = Table(
    "order",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=False),
    Column("region", String(32), nullable=False),
    Column("archived_at", DateTime, nullable=True),
)

ledger = Table(
    "ledger",
    metadata,
    Column("ledger_id", BigInteger, primary_key=True, autoincrement=False),
    Column("sku_id", BigInteger, primary_key=True, autoincrement=False),
    Column("label", String(64), nullable=False),
)

ledger_item = Table(
    "ledger_item",
    metadata,
    Column("ledger_id", BigInteger, nullable=False),
    Column("sku_id", BigInteger, nullable=False),
    Column("qty", Integer, nullable=False),
    Column("price", Numeric(10, 2), nullable=False),
    PrimaryKeyConstraint("ledger_id", "sku_id"),
    # The composite one. Two columns, one constraint, and the order matters.
    ForeignKeyConstraint(
        ["ledger_id", "sku_id"], ["ledger.ledger_id", "ledger.sku_id"]
    ),
)

# Deterministic, for the same reason demo/demo.sql is: an assertion on a count
# has to be a fact, not a probability.
ORDERS = [
    {
        "id": i,
        # The casing trap in miniature: `west` and `West` are different values
        # that a careless GROUP BY folds together.
        "region": ["north", "west", "West", "WEST", "east"][i % 5],
        # 4 of 20 archived, so null_rows is 16 and distinct_values is 4.
        "archived_at": None if i % 5 else datetime.datetime(2024, 1, 1),
    }
    for i in range(20)
]
LEDGERS = [{"ledger_id": i, "sku_id": i * 10, "label": f"item {i}"} for i in range(5)]
LEDGER_ITEMS = [
    {"ledger_id": i, "sku_id": i * 10, "qty": i + 1, "price": 10 + i}
    for i in range(5)
]

# What the reflection-parity tests assert against. Held here rather than in the
# test file so the fixture and the expectation cannot drift apart.
EXPECTED_TABLES = {"order", "ledger", "ledger_item"}
EXPECTED_ROWS = {"order": 20, "ledger": 5, "ledger_item": 5}
EXPECTED_PK = {
    "order": ["id"],
    "ledger": ["ledger_id", "sku_id"],
    "ledger_item": ["ledger_id", "sku_id"],
}
EXPECTED_FKS = {
    "order": [],
    "ledger": [],
    # Both halves of the composite key, paired with the right referenced column.
    "ledger_item": ["ledger_id -> ledger.ledger_id", "sku_id -> ledger.sku_id"],
}
# west/West/WEST are three values, not one. The count that fails on a MySQL
# instance with a case-insensitive default collation.
EXPECTED_REGIONS = {"north": 4, "west": 4, "West": 4, "WEST": 4, "east": 4}


async def build(engine) -> None:
    """Create the schema and fill it. Idempotent — drops first."""
    async with engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
        await conn.execute(insert(order), ORDERS)
        await conn.execute(insert(ledger), LEDGERS)
        await conn.execute(insert(ledger_item), LEDGER_ITEMS)
