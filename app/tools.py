"""Read-only introspection tools for the explore loop (PLAN.md §4).

Sampling is what discovers the traps: `sample_column('customer', 'deleted_at')`
comes back with non-nulls, and that's how the agent infers soft deletes rather
than being told.

Two safety properties, both structural rather than conventional:

1. **Identifiers are allowlisted against the dialect's own reflection, then
   carried as `quoted_name(..., quote=True)`.** Table and column names cannot be
   bound as parameters, so only a name the database has just told us exists may
   be put into a statement. `quoted_name` travels with the expression tree, so
   the quoting happens at compile time against whichever dialect runs it — a
   table called `order` or `Customer` works, and nothing is ever concatenated.
   Everything else is a bound parameter.
2. **These run on a target connection, which the agent reaches read-only.** The
   agent's own tables aren't hidden from this listing, they simply aren't in this
   database — `cache_entry`, `turn` and the checkpoints are on another server.
   There used to be a name filter here doing that job; it was one forgotten call
   site away from not working.

**Constraints come from SQLAlchemy's reflection, and the reason the old
hand-written `pg_catalog` query existed still holds**:
`information_schema.table_constraints` only shows constraints to a caller
holding a *non-SELECT* privilege, so the read-only role the agent connects as
sees an empty set and every table comes back keyless — join discovery would
silently degrade to guessing from column names. SQLAlchemy's PostgreSQL dialect
reads `pg_catalog` for that exact reason, and every other dialect has its own
answer to its own version of the problem. The invariant did not go away; it
stopped being ours to maintain. `tests/test_isolation.py::
test_the_reader_role_can_still_read_everything_it_needs` is what fails if that
stops being true.

**Every column read by name carries an explicit lower-case label.** `dict_row`
used to hand back Postgres's own folded names; `RowMapping` keys come straight
from `cursor.description`, and MySQL returns `SELECT count(*) AS n` as whatever
case it was written in. The labels here are ours, so this is free — but it is a
rule, not an accident.
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

    `Inspector` is sync — it issues queries as it walks — and `run_sync` hands it
    a sync-facing proxy over *this* connection rather than opening a second one.

    One closure per tool call, never one per question asked of the Inspector:
    each `run_sync` is a greenlet round trip, and the Inspector's `info_cache`
    lives only as long as the Inspector does, so splitting `describe_table` into
    three calls would reflect the table three times.
    """

    def run(sync_conn: Any) -> T:
        return work(inspect(sync_conn), sync_conn.dialect)

    return await conn.run_sync(run)


def _type_name(type_: Any, dialect: Dialect) -> str:
    """The type as this dialect spells it in DDL.

    `str(type_)` gives SQLAlchemy's generic name — `TIMESTAMP` for what Postgres
    calls `timestamp with time zone`, which is a distinction the agent needs
    when it writes a date comparison. Compiling against the dialect gives the
    database's own words back.

    An unrecognised extension type reflects as `NullType`, which has no DDL and
    raises rather than rendering. A name the model cannot use is still better
    than a tool call that fails.
    """
    try:
        return str(type_.compile(dialect=dialect))
    except (CompileError, Exception):  # noqa: B014 - dialects raise their own
        return str(type_)


# ------------------------------------------------------------- identifier gate


def _resolve(name: str, candidates: Sequence[str], what: str) -> str:
    """Match a model-supplied name against what the database actually has.

    The allowlist half of the safety property is unchanged: identifiers cannot
    be bound as parameters, so only a name the database just told us exists may
    be put into a statement. What changed is that "exists" is no longer
    case-blind — Postgres folds unquoted identifiers, MySQL on a case-sensitive
    filesystem does not, and `Customer` and `customer` can be two tables.

    So an exact match wins outright; a unique case-insensitive match is accepted,
    because the model getting the case wrong is the common failure and it is
    recoverable; two of them is an error naming both, because guessing there is
    how the agent silently reads the wrong table.
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
    """Force the dialect's own quoting, at compile time.

    The `psycopg.sql.Identifier` of this port. The name came out of reflection,
    so it is byte-exact what the server stores — which is the other half of the
    case-sensitivity fix.
    """
    return quoted_name(name, quote=True)


# -------------------------------------------------------------------- the tools


async def list_tables(conn: AsyncConnection) -> dict[str, Any]:
    """Names only. Deliberately no row counts — an agent that can see which
    tables are empty skips the search, and the search is what T1 is."""

    def work(i: Inspector, _d: Dialect) -> tuple[list[str], dict[str, int]]:
        # schema=None means "the default schema", which honours search_path on
        # Postgres rather than hardcoding 'public', is the connected database on
        # MySQL and `main` on SQLite. Passing a name explicitly would differ
        # from today's behaviour whenever search_path is unusual.
        names = i.get_table_names()
        # get_multi_columns, not a get_columns loop: one round trip instead of
        # forty, and forty is what the decoy schema is.
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
    # The column pairing that `JOIN LATERAL unnest(conkey) WITH ORDINALITY` was
    # doing by hand: constrained_columns and referred_columns are parallel and
    # ordered, so a composite foreign key pairs correctly under a plain zip.
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

    The null count is the load-bearing part: it's what turns `deleted_at` from
    a column that exists into a soft-delete convention the agent can state.
    """
    tbl, col = await _column_of(conn, table, column)
    limit = max(1, min(int(limit), 50))

    c = sa_column(_ident(col))
    t = sa_table(_ident(tbl), c)

    stats = (
        await conn.execute(
            select(
                func.count().label("total"),
                # `count(*) FILTER (WHERE c IS NULL)` was the direct spelling,
                # and MySQL has no FILTER clause — SQLAlchemy compiles
                # func.count().filter() to FILTER on *every* dialect, so it
                # would be a runtime syntax error rather than a portability
                # layer (verified). count(c) counts non-nulls in every dialect
                # there is, so the null count is a subtraction with no dialect
                # to get wrong.
                (func.count() - func.count(c)).label("nulls"),
                func.count(distinct(c)).label("distinct_values"),
            ).select_from(t)
        )
    ).mappings().one()

    # DISTINCT and ORDER BY on the **raw column**, never on the cast.
    # `CAST(x AS CHAR)` on MySQL drops the column's collation and falls back to
    # the connection's, which is case-insensitive by default — so casting first
    # folds `west`, `West` and `WEST` into one value and quietly erases exactly
    # the kind of convention this tool exists to reveal. Stringifying afterwards
    # in Python costs nothing and cannot change what the database considers
    # distinct.
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

    # GROUP BY the raw column, not the cast — see `sample_column`. On MySQL,
    # grouping by `CAST(region AS CHAR)` returns `west: 12` where grouping by
    # `region` returns `west: 4, West: 4, WEST: 4`, because the cast loses the
    # column's collation. This tool's whole job is showing a categorical
    # column's real shape, and that is the shape.
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
        # `.orig`, not `e`. SQLAlchemy's wrapper stringifies to the driver's
        # message *plus* the whole statement it just tried *plus* a
        # https://sqlalche.me/e/ link — a few hundred tokens of context per
        # failed call, in a loop that runs up to 24 of them, teaching the model
        # nothing it can act on.
        orig = getattr(e, "orig", None) or e
        return json.dumps({"error": f"{type(orig).__name__}: {orig}"}), True
