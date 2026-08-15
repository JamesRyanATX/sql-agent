"""Read-only introspection tools for the explore loop (PLAN.md §4).

Sampling is what discovers the traps: `sample_column('customer', 'deleted_at')`
comes back with non-nulls, and that is how the agent infers soft deletes rather
than being told.

Two safety properties, both structural:

1. **Identifiers are allowlisted against the dialect's own reflection, then
   carried as `quoted_name(..., quote=True)`.** Names cannot be bound as
   parameters, so only a name the database just told us exists may reach a
   statement, and the quoting happens at compile time against whichever dialect
   runs it. Everything else is a bound parameter.
2. **These run on a target connection.** The agent's own tables are not filtered
   out of the listing — they are on another server.

**Constraints come from SQLAlchemy's reflection.**
`information_schema.table_constraints` only shows constraints to a caller with a
non-SELECT privilege, so the read-only role would see none and every table would
come back keyless, degrading join discovery to guessing from column names. Each
dialect has its own answer to that, which is why this does not hand-write one.
`tests/test_isolation.py::test_the_reader_role_can_still_read_everything_it_needs`
is what fails if it stops being true.

**Every column read by name carries an explicit lower-case label.** `RowMapping`
keys come from `cursor.description`, and MySQL returns `SELECT count(*) AS n` in
whatever case it was written.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from sqlalchemy import distinct, func, inspect, select
from sqlalchemy import column as sa_column
from sqlalchemy import table as sa_table
from sqlalchemy.engine import Dialect
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import CompileError
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql.elements import quoted_name

SAMPLE_LIMIT = 12

T = TypeVar("T")


class ToolError(Exception):
    """Returned to the model as an error tool_result so it can correct itself."""


# ------------------------------------------------------------------ reflection


async def _reflect(
    conn: AsyncConnection, work: Callable[[Inspector, Dialect], T]
) -> T:
    """Run one closure's worth of reflection on the connection's greenlet.

    One closure per tool call, never one per question asked of the Inspector —
    its `info_cache` lives only as long as it does.
    """

    def run(sync_conn: Any) -> T:
        return work(inspect(sync_conn), sync_conn.dialect)

    return await conn.run_sync(run)


def _type_name(type_: Any, dialect: Dialect) -> str:
    """The type as this dialect spells it in DDL.

    `str(type_)` gives SQLAlchemy's generic `TIMESTAMP` for what Postgres calls
    `timestamp with time zone`, which the agent needs to write a date
    comparison. Falls back for extension types, which reflect as `NullType`.
    """
    try:
        return str(type_.compile(dialect=dialect))
    except (CompileError, Exception):  # noqa: B014 - dialects raise their own
        return str(type_)


# ------------------------------------------------------------- identifier gate


def _resolve(name: str, candidates: Sequence[str], what: str) -> str:
    """Match a model-supplied name against what the database actually has.

    `Customer` and `customer` can be two tables, so an exact match wins, a
    unique case-insensitive match is accepted, and two is an error naming both.
    """
    if name in candidates:
        return name
    folded = sorted(c for c in candidates if c.casefold() == name.casefold())
    if len(folded) == 1:
        return folded[0]
    if folded:
        raise ToolError(f"{name!r} is ambiguous — did you mean one of {folded}?")
    raise ToolError(f"no such {what}: {name!r}")


def _ident(name: str) -> quoted_name:
    """Force the dialect's own quoting, at compile time. The name came out of
    reflection, so it is byte-exact what the server stores.
    """
    return quoted_name(name, quote=True)


# -------------------------------------------------------------------- the tools


async def list_tables(conn: AsyncConnection) -> dict[str, Any]:
    """Names only. Deliberately no row counts — an agent that can see which
    tables are empty skips the search, and the search is what T1 is."""

    def work(i: Inspector, _d: Dialect) -> tuple[list[str], dict[str, int]]:
        # No schema= — the default honours search_path on Postgres, is the
        # connected database on MySQL and `main` on SQLite.
        names = i.get_table_names()
        # One round trip rather than one per table, and forty is the decoy schema.
        widths = i.get_multi_columns(filter_names=names)
        return names, {key[1]: len(cols) for key, cols in widths.items()}

    names, widths = await _reflect(conn, work)
    return {
        "tables": [f"{n} ({widths.get(n, 0)} cols)" for n in sorted(names)],
        "count": len(names),
    }


async def describe_table(conn: AsyncConnection, table: str) -> dict[str, Any]:
    """Columns, types, nullability, and the keys — enough to write a join."""

    def work(i: Inspector, d: Dialect) -> tuple:
        name = _resolve(table, i.get_table_names(), "table")
        return (
            name,
            d,
            i.get_columns(name),
            i.get_pk_constraint(name),
            i.get_foreign_keys(name),
        )

    name, dialect, cols, pk, fks = await _reflect(conn, work)

    columns = [
        {
            "name": c["name"],
            "type": _type_name(c["type"], dialect),
            "nullable": bool(c["nullable"]),
            **({"default": c["default"]} if c.get("default") else {}),
        }
        for c in cols
    ]
    primary_key = list(pk.get("constrained_columns") or [])
    # constrained_columns and referred_columns are parallel and ordered, so a
    # composite foreign key pairs correctly under a plain zip.
    foreign_keys = [
        f"{src} -> {fk['referred_table']}.{ref}"
        for fk in fks
        for src, ref in zip(fk["constrained_columns"], fk["referred_columns"])
    ]

    n = (
        await conn.execute(
            select(func.count().label("n")).select_from(sa_table(_ident(name)))
        )
    ).scalar_one()

    out: dict[str, Any] = {"table": name, "rows": n, "columns": columns}
    if primary_key:
        out["primary_key"] = primary_key
    if foreign_keys:
        out["foreign_keys"] = sorted(set(foreign_keys))
    return out


async def _column_of(conn: AsyncConnection, table: str, col: str) -> tuple[str, str]:
    """Resolve a table and one of its columns in a single reflection pass."""

    def work(i: Inspector, _d: Dialect) -> tuple[str, str]:
        name = _resolve(table, i.get_table_names(), "table")
        found = _resolve(col, [c["name"] for c in i.get_columns(name)], "column")
        return name, found

    return await _reflect(conn, work)


async def sample_column(
    conn: AsyncConnection, table: str, column: str, limit: int = SAMPLE_LIMIT
) -> dict[str, Any]:
    """Distinct values plus how many rows are null.

    The null count is what turns `deleted_at` from a column that exists into a
    soft-delete convention the agent can state.
    """
    tbl, col = await _column_of(conn, table, column)
    limit = max(1, min(int(limit), 50))

    c = sa_column(_ident(col))
    t = sa_table(_ident(tbl), c)

    stats = (
        await conn.execute(
            select(
                func.count().label("total"),
                # Not `count(*) FILTER (WHERE c IS NULL)`: MySQL has no FILTER
                # clause and SQLAlchemy compiles `.filter()` to it on every
                # dialect. count(c) counts non-nulls everywhere.
                (func.count() - func.count(c)).label("nulls"),
                func.count(distinct(c)).label("distinct_values"),
            ).select_from(t)
        )
    ).mappings().one()

    # DISTINCT and ORDER BY on the **raw column**, never on a cast. `CAST(x AS
    # CHAR)` on MySQL drops the column's collation for the connection's, which is
    # case-insensitive by default, folding `west`/`West`/`WEST` into one value.
    # Stringify in Python instead.
    values = [
        str(r[0])
        for r in (
            await conn.execute(
                select(c)
                .distinct()
                .select_from(t)
                .where(c.is_not(None))
                .order_by(c)
                .limit(limit)
            )
        ).all()
    ]

    return {
        "table": tbl,
        "column": col,
        "total_rows": stats["total"],
        "null_rows": stats["nulls"],
        "distinct_values": stats["distinct_values"],
        "sample": values,
    }


async def count_distinct(
    conn: AsyncConnection, table: str, column: str
) -> dict[str, Any]:
    """Value frequencies. Shows an enum's real shape, and casing variants."""
    tbl, col = await _column_of(conn, table, column)

    c = sa_column(_ident(col))
    t = sa_table(_ident(tbl), c)

    # GROUP BY the raw column, not a cast — see `sample_column`. On MySQL the
    # cast returns `west: 12` where the column returns `west: 4, West: 4,
    # WEST: 4`, and that is the shape this tool exists to show.
    rows = (
        await conn.execute(
            select(c.label("value"), func.count().label("n"))
            .select_from(t)
            .group_by(c)
            .order_by(func.count().desc())
            .limit(25)
        )
    ).mappings().all()

    return {
        "table": tbl,
        "column": col,
        "counts": [{"value": str(r["value"]), "rows": r["n"]} for r in rows],
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
        # `.orig`, not `e`: SQLAlchemy's wrapper stringifies to the driver's
        # message plus the whole statement plus a docs link — hundreds of tokens
        # per failed call, in a loop that runs up to 24 of them.
        orig = getattr(e, "orig", None) or e
        return json.dumps({"error": f"{type(orig).__name__}: {orig}"}), True
