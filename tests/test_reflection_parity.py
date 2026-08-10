"""The four tools return the same shapes on every engine.

`app/tools.py` moved from hand-written `information_schema` and `pg_catalog` SQL
to `sqlalchemy.inspect`. That is the part of the port that gets *simpler* — but
"simpler" is only true if the output the model reads is still the same shape,
because those strings are in its prompt on every T1.

Run against the SELECT-only principal where the dialect has one, because the
privilege problem is why the Postgres implementation could not use
`information_schema.table_constraints` in the first place: it shows constraints
only to a caller holding a *non-SELECT* privilege, so a read-only role saw an
empty set and every table came back keyless.
"""

from __future__ import annotations

import pytest

from app import tools
from tests.fixtures import portable as fixture


@pytest.fixture
async def conn(portable):
    async with portable.engine.connect() as c:
        yield c


# ------------------------------------------------------------------ the shapes


async def test_list_tables_finds_exactly_the_fixture(conn, portable):
    """**Exactly** equal, not a superset. That is what catches an engine's own
    internals leaking in through a wrong `schema=` — `sqlite_sequence` on
    SQLite, `performance_schema`/`sys` on MySQL — which is a new and real risk
    under a reflection port and would send the agent exploring the engine."""
    out = await tools.list_tables(conn)
    names = {t.split(" (")[0] for t in out["tables"]}
    assert names == fixture.EXPECTED_TABLES
    assert out["count"] == len(fixture.EXPECTED_TABLES)


async def test_the_listing_format_is_identical_across_dialects(conn):
    """`name (N cols)` is a prompt string. A format that varied by dialect would
    make every cached recipe dialect-specific for no reason at all."""
    out = await tools.list_tables(conn)
    assert "ledger_item (4 cols)" in out["tables"]
    assert "order (3 cols)" in out["tables"]


@pytest.mark.parametrize("table", sorted(fixture.EXPECTED_TABLES))
async def test_describe_table_reflects_the_keys(conn, table):
    """The composite foreign key is the point.

    Pairing its columns with the right referenced ones is what the Postgres
    implementation needed `JOIN LATERAL unnest(conkey) WITH ORDINALITY` for.
    The port replaced that with a `zip` over two parallel lists, and a crossed
    pairing would still look plausible — `ledger_id -> ledger.sku_id` reads fine
    right up until the agent writes a join on it.
    """
    out = await tools.describe_table(conn, table)
    assert out["table"] == table
    assert out["rows"] == fixture.EXPECTED_ROWS[table]
    # Ordered, not a set: get_pk_constraint returns constraint order on
    # Postgres and index order on MySQL, and a join needs them the right way up.
    assert out.get("primary_key", []) == fixture.EXPECTED_PK[table]
    assert out.get("foreign_keys", []) == fixture.EXPECTED_FKS[table]


async def test_a_reserved_word_table_is_quoted_not_concatenated(conn):
    """`order` is reserved on all three. The demo has no such table, so nothing
    used to exercise the quoting — `quoted_name` does it at compile time against
    whichever dialect runs the statement."""
    out = await tools.describe_table(conn, "order")
    assert out["rows"] == 20
    sample = await tools.sample_column(conn, "order", "region")
    assert sample["total_rows"] == 20


async def test_column_types_are_stable_within_a_dialect(conn):
    """Never compared *across* dialects: `character varying`, `VARCHAR(32)` and
    `TEXT` are all correct answers to the same question. Stability within one
    engine is the only property `schema_fingerprint` actually depends on."""
    first = await tools.describe_table(conn, "order")
    second = await tools.describe_table(conn, "order")
    types = {c["name"]: c["type"] for c in first["columns"]}
    assert types == {c["name"]: c["type"] for c in second["columns"]}
    assert all(t for t in types.values()), "a blank type tells the model nothing"


async def test_columns_keep_their_order_and_nullability(conn):
    out = await tools.describe_table(conn, "order")
    assert [c["name"] for c in out["columns"]] == ["id", "region", "archived_at"]
    nullable = {c["name"]: c["nullable"] for c in out["columns"]}
    assert nullable == {"id": False, "region": False, "archived_at": True}


# ---------------------------------------------------------------- the sampling


async def test_sample_column_counts_nulls(conn):
    """The load-bearing part: it is what turns a column that exists into a
    convention the agent can state. Computed as `count(*) - count(c)`, because
    `count(*) FILTER (WHERE ...)` compiles to FILTER on every dialect and MySQL
    has no FILTER clause."""
    out = await tools.sample_column(conn, "order", "archived_at")
    assert out["total_rows"] == 20
    assert out["null_rows"] == 16
    assert out["distinct_values"] == 1


async def test_sample_column_is_bounded(conn):
    out = await tools.sample_column(conn, "order", "region", limit=2)
    assert len(out["sample"]) == 2


async def test_count_distinct_keeps_casing_variants_apart(conn):
    """west, West and WEST are three values. The one that fails on a MySQL
    instance with a case-insensitive default collation — and the miniature of
    the trap the whole demo is built on."""
    out = await tools.count_distinct(conn, "order", "region")
    counts = {c["value"]: c["rows"] for c in out["counts"]}
    assert counts == fixture.EXPECTED_REGIONS
    assert sum(counts.values()) == 20


# ------------------------------------------------------------------ the gate


async def test_an_unknown_table_is_a_tool_error_not_an_exception(conn):
    with pytest.raises(tools.ToolError, match="no such table"):
        await tools.describe_table(conn, "order; DROP TABLE ledger")
    # And the table survived it.
    assert (await tools.describe_table(conn, "ledger"))["rows"] == 5


async def test_the_wrong_case_is_forgiven_when_it_is_unambiguous(conn):
    """The model getting the case wrong is the common failure and it is
    recoverable. Two candidates would not be — that raises, naming both."""
    assert (await tools.describe_table(conn, "LEDGER"))["table"] == "ledger"


async def test_run_tool_reports_errors_as_results(conn):
    import json

    payload, is_error = await tools.run_tool(conn, "describe_table", {"table": "nope"})
    assert is_error
    # And the message is the driver's, not SQLAlchemy's wrapper with the whole
    # statement and a docs link stapled to it.
    assert "sqlalche.me" not in payload
    assert "no such table" in json.loads(payload)["error"]
