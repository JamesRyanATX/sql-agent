"""Phase 3: the introspection tools are safe, and the traps are discoverable.

No API calls — these run against the seeded database only.
Requires `make up && make migrate && make seed`.
"""

import json
from collections.abc import AsyncIterator

import psycopg
import pytest
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from app.settings import settings
from app.tools import (
    AGENT_TABLES,
    ToolError,
    count_distinct,
    describe_table,
    list_tables,
    run_tool,
    sample_column,
)


@pytest.fixture
async def conn() -> AsyncIterator[AsyncConnection]:
    async with await psycopg.AsyncConnection.connect(
        settings().database_url, row_factory=dict_row
    ) as c:
        yield c
        await c.rollback()


# ------------------------------------------------------------------- safety


async def test_the_agent_cannot_see_its_own_memory(conn):
    """Otherwise it explores the cache, caches facts about the cache, and
    beat 2 of the demo fills up with noise."""
    listed = await list_tables(conn)
    names = {t.split(" ")[0] for t in listed["tables"]}
    assert names.isdisjoint(AGENT_TABLES)
    assert "customer" in names

    for hidden in ("cache_entry", "turn", "checkpoints"):
        with pytest.raises(ToolError, match="no such table"):
            await describe_table(conn, hidden)


async def test_identifiers_are_allowlisted_not_interpolated(conn):
    for hostile in (
        "customer; DROP TABLE customer",
        'customer" ; DROP TABLE customer --',
        "pg_shadow",
        "../etc/passwd",
    ):
        with pytest.raises(ToolError, match="no such table"):
            await describe_table(conn, hostile)

    with pytest.raises(ToolError, match="no such column"):
        await sample_column(conn, "customer", "name; DROP TABLE customer")

    # The table survived every one of those.
    cur = await conn.execute("SELECT count(*) AS n FROM customer")
    assert (await cur.fetchone())["n"] == 2_000


async def test_tool_errors_come_back_as_results_not_exceptions(conn):
    """A typo'd column name should let the model correct itself, not end the turn."""
    payload, is_error = await run_tool(conn, "describe_table", {"table": "nope"})
    assert is_error is True
    assert "no such table" in json.loads(payload)["error"]

    payload, is_error = await run_tool(conn, "not_a_tool", {})
    assert is_error is True

    payload, is_error = await run_tool(conn, "describe_table", {"wrong_arg": 1})
    assert is_error is True


# -------------------------------------------------------------- the search


async def test_list_tables_shows_the_haystack(conn):
    listed = await list_tables(conn)
    assert listed["count"] >= 40
    assert any(t.startswith("customer ") for t in listed["tables"])
    # Names and column counts only — an agent that could see which tables are
    # empty would skip the search, and the search is what T1 is.
    assert all("rows" not in t for t in listed["tables"])


async def test_describe_table_gives_enough_to_write_a_join(conn):
    described = await describe_table(conn, "order_item")
    names = {c["name"] for c in described["columns"]}
    assert names == {"order_id", "product_id", "qty", "price"}
    assert described["rows"] == 18_000
    assert set(described["primary_key"]) == {"order_id", "product_id"}
    assert "order_id -> orders.id" in described["foreign_keys"]
    assert "product_id -> product.id" in described["foreign_keys"]


# ------------------------------------------------- the traps are discoverable


async def test_sampling_deleted_at_reveals_the_soft_delete_convention(conn):
    """The agent is meant to infer soft deletes from the data, not be told."""
    sampled = await sample_column(conn, "customer", "deleted_at")
    assert sampled["total_rows"] == 2_000
    assert sampled["null_rows"] == 1_840   # <- the answer, visible in one call
    assert len(sampled["sample"]) > 0      # and it's populated, so it means something


async def test_counting_region_reveals_the_casing_variants(conn):
    counted = await count_distinct(conn, "customer", "region")
    values = {c["value"] for c in counted["counts"]}
    assert {"west", "West", "WEST"} <= values
    west = {c["value"]: c["rows"] for c in counted["counts"]}
    assert west["west"] + west["West"] + west["WEST"] == 500


async def test_counting_status_reveals_cancelled(conn):
    counted = await count_distinct(conn, "orders", "status")
    by_value = {c["value"]: c["rows"] for c in counted["counts"]}
    assert by_value["cancelled"] == 750


async def test_describing_orders_reveals_created_not_created_at(conn):
    described = await describe_table(conn, "orders")
    names = {c["name"] for c in described["columns"]}
    assert "created" in names
    assert "created_at" not in names


async def test_sampling_is_bounded(conn):
    """Cheap tool output keeps T1's cost a known quantity."""
    sampled = await sample_column(conn, "customer", "name", limit=500)
    assert len(sampled["sample"]) <= 50
    counted = await count_distinct(conn, "customer", "name")
    assert len(counted["counts"]) <= 25
