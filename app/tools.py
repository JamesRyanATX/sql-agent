"""Read-only introspection tools for the explore loop (PLAN.md §4).

Sampling is what discovers the traps: `sample_column('customer', 'deleted_at')`
comes back with non-nulls, and that's how the agent infers soft deletes rather
than being told.

Two safety properties, both structural rather than conventional:

1. **Identifiers are validated against `information_schema`, then quoted with
   `psycopg.sql.Identifier`.** Table and column names can't be bound as
   parameters, so an allowlist is the only correct answer. Everything else is a
   bound parameter.
2. **The agent's own tables are invisible.** `public` also holds `cache_entry`,
   `turn` and LangGraph's checkpoint tables; without this the agent explores its
   own memory and caches facts about the cache.
"""

from __future__ import annotations

import json
from typing import Any

from psycopg import AsyncConnection, sql

from app.store import AGENT_TABLES as _AGENT_TABLES

# The agent's memory is not part of the business it's answering questions about.
# Defined in `store`, which also wipes exactly this set — the list has to be one
# list, or a table added to the reset would stay visible to introspection.
AGENT_TABLES = frozenset(_AGENT_TABLES)

SAMPLE_LIMIT = 12


class ToolError(Exception):
    """Returned to the model as an error tool_result so it can correct itself."""


# ------------------------------------------------------------- identifier gate


async def _check_table(conn: AsyncConnection, table: str) -> str:
    if table in AGENT_TABLES:
        raise ToolError(f"no such table: {table!r}")
    cur = await conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s",
        (table,),
    )
    if await cur.fetchone() is None:
        raise ToolError(f"no such table: {table!r}")
    return table


async def _check_column(conn: AsyncConnection, table: str, column: str) -> str:
    await _check_table(conn, table)
    cur = await conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
        (table, column),
    )
    if await cur.fetchone() is None:
        raise ToolError(f"no such column: {table}.{column}")
    return column


# -------------------------------------------------------------------- the tools


async def list_tables(conn: AsyncConnection) -> dict[str, Any]:
    """Names only. Deliberately no row counts — an agent that can see which
    tables are empty skips the search, and the search is what T1 is."""
    cur = await conn.execute(
        """
        SELECT c.table_name, count(*) AS n_columns
        FROM information_schema.columns c
        JOIN information_schema.tables t
          ON t.table_schema = c.table_schema AND t.table_name = c.table_name
        WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE'
          AND NOT (c.table_name = ANY(%s))
        GROUP BY c.table_name ORDER BY c.table_name
        """,
        (list(AGENT_TABLES),),
    )
    rows = await cur.fetchall()
    return {
        "tables": [f"{r['table_name']} ({r['n_columns']} cols)" for r in rows],
        "count": len(rows),
    }


async def describe_table(conn: AsyncConnection, table: str) -> dict[str, Any]:
    """Columns, types, nullability, and the keys — enough to write a join."""
    await _check_table(conn, table)

    cur = await conn.execute(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table,),
    )
    columns = [
        {
            "name": r["column_name"],
            "type": r["data_type"],
            "nullable": r["is_nullable"] == "YES",
            **({"default": r["column_default"]} if r["column_default"] else {}),
        }
        for r in await cur.fetchall()
    ]

    cur = await conn.execute(
        """
        SELECT kcu.column_name, tc.constraint_type,
               ccu.table_name AS ref_table, ccu.column_name AS ref_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON kcu.constraint_name = tc.constraint_name
        LEFT JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND tc.constraint_type = 'FOREIGN KEY'
        WHERE tc.table_schema = 'public' AND tc.table_name = %s
          AND tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
        """,
        (table,),
    )
    primary_key, foreign_keys = [], []
    for r in await cur.fetchall():
        if r["constraint_type"] == "PRIMARY KEY":
            primary_key.append(r["column_name"])
        else:
            foreign_keys.append(
                f"{r['column_name']} -> {r['ref_table']}.{r['ref_column']}"
            )

    cur = await conn.execute(
        sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(table))
    )
    row = await cur.fetchone()

    out: dict[str, Any] = {"table": table, "rows": row["n"], "columns": columns}
    if primary_key:
        out["primary_key"] = primary_key
    if foreign_keys:
        out["foreign_keys"] = sorted(set(foreign_keys))
    return out


async def sample_column(
    conn: AsyncConnection, table: str, column: str, limit: int = SAMPLE_LIMIT
) -> dict[str, Any]:
    """Distinct values plus how many rows are null.

    The null count is the load-bearing part: it's what turns `deleted_at` from
    a column that exists into a soft-delete convention the agent can state.
    """
    await _check_column(conn, table, column)
    limit = max(1, min(int(limit), 50))

    ident = sql.Identifier(table)
    col = sql.Identifier(column)

    cur = await conn.execute(
        sql.SQL(
            "SELECT count(*) AS total, "
            "count(*) FILTER (WHERE {c} IS NULL) AS nulls, "
            "count(DISTINCT {c}) AS distinct_values FROM {t}"
        ).format(c=col, t=ident)
    )
    stats = await cur.fetchone()

    cur = await conn.execute(
        sql.SQL(
            "SELECT DISTINCT {c}::text AS v FROM {t} "
            "WHERE {c} IS NOT NULL ORDER BY 1 LIMIT %s"
        ).format(c=col, t=ident),
        (limit,),
    )
    values = [r["v"] for r in await cur.fetchall()]

    return {
        "table": table,
        "column": column,
        "total_rows": stats["total"],
        "null_rows": stats["nulls"],
        "distinct_values": stats["distinct_values"],
        "sample": values,
    }


async def count_distinct(
    conn: AsyncConnection, table: str, column: str
) -> dict[str, Any]:
    """Value frequencies. Shows an enum's real shape, and casing variants."""
    await _check_column(conn, table, column)

    cur = await conn.execute(
        sql.SQL(
            "SELECT {c}::text AS value, count(*) AS n FROM {t} "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 25"
        ).format(c=sql.Identifier(column), t=sql.Identifier(table))
    )
    return {
        "table": table,
        "column": column,
        "counts": [{"value": r["value"], "rows": r["n"]} for r in await cur.fetchall()],
    }


# ---------------------------------------------------------------- tool schemas

SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_tables",
        "description": (
            "List every table in the database with its column count. "
            "Start here when you don't yet know what exists."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "describe_table",
        "description": (
            "Show a table's columns, types, nullability, row count, primary key "
            "and foreign keys. Call this before writing SQL against a table — "
            "column names are frequently not what you'd guess."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"table": {"type": "string"}},
            "required": ["table"],
        },
    },
    {
        "name": "sample_column",
        "description": (
            "Show distinct values from a column, plus how many rows are null. "
            "Call this when a column's meaning affects the answer — how many "
            "rows are populated is often the difference between a column that "
            "exists and a convention the data actually follows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "column": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["table", "column"],
        },
    },
    {
        "name": "count_distinct",
        "description": (
            "Count rows per distinct value in a column, most frequent first. "
            "Use it to see the real shape of a categorical column."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "column": {"type": "string"},
            },
            "required": ["table", "column"],
        },
    },
]

_DISPATCH = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "sample_column": sample_column,
    "count_distinct": count_distinct,
}


async def run_tool(
    conn: AsyncConnection, name: str, args: dict[str, Any]
) -> tuple[str, bool]:
    """Execute one tool call. Returns (result_json, is_error).

    Tool errors come back as ordinary results with is_error set, so the model
    corrects itself rather than the turn dying on a typo'd column name.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"no such tool: {name}"}), True
    try:
        return json.dumps(await fn(conn, **args), default=str), False
    except ToolError as e:
        return json.dumps({"error": str(e)}), True
    except Exception as e:  # a bad argument shouldn't end the turn
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
