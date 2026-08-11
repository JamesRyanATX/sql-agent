"""The turn graph (PLAN.md §4).

Phases 3-4 wire the learning path:

    load_cache → explore ⇄ introspect → generate_sql → execute → extract → answer
                                                          │  ▲
                                                       error │
                                                          ▼  │
                                                        fix (≤3)

`plan` — the branch that skips exploration and makes T2 cheap — arrives in
phase 5. Until then every turn explores, but from phase 4 onward it also writes
down what it learned.
"""

from __future__ import annotations

import functools
import json
import re
import time
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import inspect
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app import db, llm, store, tools, tracing
from app.events import to_events
from app.settings import settings


def _add(a: int, b: int) -> int:
    return a + b


class TurnState(TypedDict, total=False):
    session_id: str
    question: str
    # Which registered database this turn is about. In the state and not in
    # `config["configurable"]` alongside thread_id, deliberately: a resumed
    # thread replays its checkpointed state, so a value living outside the
    # checkpoint could silently switch warehouses mid-conversation — replaying
    # a `cache` loaded from one while `execute` runs against another. It is
    # also domain data (it scopes load_cache and lands in turn.connection_id),
    # where thread_id is infrastructure.
    connection_id: str
    # Which SQL the model should write. Set by `load_cache` from the registry
    # row — no target connection is needed, because `driver` is a column. In
    # state for the same reason connection_id is: a value derived outside the
    # checkpoint lets a resumed thread plan in one dialect and execute in
    # another.
    dialect: str

    turn_id: int
    started_at: float
    cache: list[dict[str, Any]]
    # The Langfuse trace this turn is recorded as, or "" when tracing is off —
    # which is the common case. Minted in `stream_turn` and carried in, rather
    # than read back out of the ambient OpenTelemetry context inside `answer`,
    # so writing it to the turn row does not depend on that context reaching
    # LangGraph's tasks. Same argument as connection_id above: a turn's
    # identity belongs in the checkpointed state.
    trace_id: str

    sufficient: bool
    used_ids: list[int]
    missing: list[str]
    findings: str
    sql: str
    assumptions: list[str]
    rows: list[dict[str, Any]]
    error: str
    fix_attempts: int
    answer: str
    explored: bool
    tool_calls: int

    # Reduced across nodes, so the counter is a real sum of every API call the
    # turn made rather than a number reconstructed afterwards.
    tokens_in: Annotated[int, _add]
    tokens_out: Annotated[int, _add]


# --------------------------------------------------------------------- prompts

# A label, not the driver name: "postgresql+psycopg" is which library dials the
# socket, and the model needs to know which SQL to write.
DIALECT_LABEL = {"postgresql": "PostgreSQL", "mysql": "MySQL", "sqlite": "SQLite"}

# Appended to the system prompt of every node that reads or writes SQL. Only
# what the model gets wrong often enough to be worth the tokens — this is not a
# reference manual, and every line is paid for on every turn.
DIALECT_NOTES = {
    "postgresql": "",
    "mysql": (
        "MySQL: no FILTER clause and no `::` casts — use CAST(x AS CHAR). "
        "Identifiers quote with backticks. String comparison is "
        "case-insensitive under the default collation, so a GROUP BY folds "
        "casing variants together that PostgreSQL would keep apart."
    ),
    "sqlite": (
        "SQLite: no native date or boolean type — dates are text or integers, "
        "so compare them as such. No RIGHT or FULL OUTER JOIN before 3.39."
    ),
}


def dialect_note(dialect: str) -> str:
    """The dialect line for a system prompt, or nothing for the default."""
    note = DIALECT_NOTES.get(dialect, "")
    label = DIALECT_LABEL.get(dialect, dialect)
    return f"\n\nTarget dialect: {label}." + (f" {note}" if note else "")


EXPLORE_SYSTEM = """\
You are answering a business question against a SQL database you have \
never seen. Use the introspection tools to find out what you need.

The schema is wide and mostly irrelevant — expect to discard most tables. \
Column names are frequently not what you would guess, and a column existing \
tells you less than how its values are actually distributed.

Work until you could write correct SQL, then stop calling tools and reply with \
a short plain-English summary of what you found: the tables that matter, the \
join keys, and any convention the data follows that a newcomer would miss. \
Write it for someone who has not seen the tool output.

Deliver what was asked, at the scope intended. Do not explore tables that \
cannot affect this question."""

PLAN_SYSTEM = """\
You know some things about this database already. Decide whether they are \
enough to answer the question, and if they are, write the SQL now.

Enough means you can name every table and column the query needs and every \
convention that changes the result. A recipe marked as verified has been run \
successfully before; an unverified note is a lead, not a fact.

If it is enough, set sufficient to true and return the SQL, listing the entries \
you relied on by name. Otherwise set it to false and say what you still need to \
find out — be specific, since that list is what the next step goes looking for. \
Guessing a column name is not sufficiency.

Answer exactly the question asked. Do not narrow it to something you have a \
recipe for, and do not broaden it — a cached recipe for a related question is a \
reason to say what is missing, not to answer a different question."""

SQL_SYSTEM = """\
Write one SELECT that answers the question, using the findings given.

Return the SQL and the assumptions you made — an assumption is anything a \
reader would need to know to agree the answer is correct, such as which rows \
you excluded and why. Read-only: no DDL, no writes, no CTEs that modify."""

FIX_SYSTEM = """\
The SQL failed. Given the error and the schema findings, return corrected SQL.

Fix the specific cause. Do not restructure the query beyond what the error \
requires."""

EXTRACT_SYSTEM = """\
A query just ran successfully. Write down what a newcomer to this database \
would need to know to get it right first time.

You are given the SQL that actually executed. **Everything you record must be \
supported by that SQL or by the schema findings — not by what you meant to do.** \
If the SQL excludes two statuses, the recipe covers two statuses, whatever the \
intent was.

Record two kinds of thing:

- **schema_fact** — something durably true about the shape of the data: a \
table's purpose, a column that isn't named what you'd guess, a join key, an \
enum's real values.
- **recipe** — how to express a business concept in SQL. Give it the name a \
person would use ("revenue", "active customer") and a `sql_fragment` copied \
from the query that ran.

Neither kind is a census. A row count, a percentage, or a parenthetical like \
"(currently 1,840)" is this query's answer, not a fact about the schema — it \
goes stale the instant a row changes, and nothing ever revisits it to check. \
"deleted_at is a nullable soft-delete flag" is a schema_fact; "160 of 2,000 \
rows are soft-deleted" is not — write the rule, not the count it produced today.

Every entry needs a short, stable `name` — it is the key this is filed under, \
and reusing a name **overwrites** what is already there.

Reuse a name only when this query taught you more about *that same concept*, so \
the note is refined rather than duplicated. If what you learned is narrower, \
broader, or merely related — "revenue" against "revenue by region", "active \
customer" against "active customer in a region" — give it its own name. \
Overwriting a general rule with a special case destroys the general rule, and \
every later question that relied on it inherits the narrower one.

Write claims as plain English a colleague could read aloud. State the \
convention, not the query you happened to write: "an active customer is one \
whose deleted_at is null", not "I filtered on deleted_at".

Record only what this query actually establishes. Nothing speculative, nothing \
you did not verify, and nothing that merely restates the question."""

# This used to say "for someone who did not see the query", and that reader
# existed: the whole result was `=> [(1840)]` and the answer text was not printed
# at all. Now the rows are a table directly above this, and a model told to "lead
# with the number" reads a nine-row result as nine numbers and types the table
# out again underneath itself.
#
# Measured on the quarters question: 353 output tokens before, 278 after, so the
# saving is real but modest — most of that 353 was thinking, not the recitation.
# The reason to change it is the duplication, not the tokens. What a table cannot
# say is what was counted and what the shape means, and that is the whole job
# left for prose.
ANSWER_SYSTEM = """\
State the answer in one or two sentences. The reader is looking at the result \
rows already, so do not list them back.

Say what the numbers mean and what was counted: the load-bearing assumptions \
inline in the prose, and anything about the shape of the result a reader would \
otherwise misread. Where the result is a single value, lead with that value. \
Do not restate the SQL, and do not add caveats that change nothing."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "sql": {
            "type": "string",
            "description": "required when sufficient is true; a single SELECT",
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "names of the cache entries relied on",
        },
        "missing": {
            "type": "array",
            "items": {"type": "string"},
            "description": "what still needs discovering; drives exploration",
        },
    },
    "required": ["sufficient", "used", "missing"],
    "additionalProperties": False,
}

SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string", "description": "a single SELECT statement"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sql", "assumptions"],
    "additionalProperties": False,
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["schema_fact", "recipe"]},
                    "name": {
                        "type": "string",
                        "description": (
                            "short stable key for this fact or concept, e.g. "
                            "'revenue', 'active customer', 'orders.created'. "
                            "Reuse the same name if you learn more about it."
                        ),
                    },
                    "claim": {"type": "string", "description": "plain English"},
                    "sql_fragment": {
                        "type": "string",
                        "description": "recipes only; copied from the SQL that ran",
                    },
                    "tables": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["kind", "name", "claim", "tables"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entries"],
    "additionalProperties": False,
}

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string"},
        "what_was_wrong": {"type": "string"},
    },
    "required": ["sql", "what_was_wrong"],
    "additionalProperties": False,
}


def render_cache(entries: list[dict[str, Any]]) -> str:
    """The prose the model reads. Empty until phase 4 starts writing entries."""
    if not entries:
        return ""
    lines = ["What you already know about this database:"]
    for e in entries:
        prefix = "NOT TRUE: " if e.get("tombstone") else ""
        label = f"{e['name']}: " if e.get("name") else ""
        lines.append(f"- {prefix}{label}{e['claim']}")
        if e.get("sql_fragment"):
            lines.append(f"  SQL: {e['sql_fragment']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------- nodes


async def load_cache(state: TurnState) -> TurnState:
    """Open the turn and load everything not disabled, ordered by hits."""
    async with db.agent() as conn:
        turn_id = await store.start_turn(
            conn,
            connection_id=state["connection_id"],
            session_id=state["session_id"],
            question=state["question"],
        )
        entries = await store.load_cache(conn, connection_id=state["connection_id"])
        # From the registry row, not from a target connection — `driver` is a
        # column, so the dialect is known before anything is dialled.
        registered = await store.get_connection(conn, state["connection_id"])

    cache = [
        {
            "id": e.id,
            "name": e.name,
            "claim": e.claim,
            "sql_fragment": e.sql_fragment,
            "tombstone": e.tombstone,
        }
        for e in entries
    ]
    return {
        "turn_id": turn_id,
        "started_at": time.monotonic(),
        "cache": cache,
        "dialect": registered.dialect if registered else "postgresql",
        "fix_attempts": 0,
        "tool_calls": 0,
    }


async def plan(state: TurnState) -> TurnState:
    """Can this question be answered from what's cached? **This is the product.**

    On the cached path it also writes the SQL, in the same call. Splitting the
    decision from the generation costs a second round trip that re-sends the
    system prompt and the whole cache — and if the cache is sufficient to
    *answer*, it is sufficient to write the query (§4 amendment).
    """
    emit = get_stream_writer()
    cache = state.get("cache", [])

    # A cold cache can only produce one answer, so don't pay a model call to
    # hear it. This is every T1, and it keeps the cold path exactly as cheap as
    # it was before this node existed.
    if not cache:
        emit({"type": "plan", "cache_entries": 0, "sufficient": False, "missing": []})
        return {"sufficient": False}

    result = await llm.complete(
        # The dialect goes in the *cached* system block, between the durable
        # instruction and the cache text. That costs nothing: prompt caching
        # keys on an exact prefix and the cache text is already per-connection,
        # so the number of live entries is unchanged. The rule for anything
        # added here — it must be a function of `connection_id` alone. The cache
        # text is; the dialect is; the question is not, which is why the
        # question stays in the user message.
        system=(
            f"{PLAN_SYSTEM}{dialect_note(state['dialect'])}\n\n"
            f"{render_cache(cache)}"
        ),
        messages=[{"role": "user", "content": f"Question: {state['question']}"}],
        effort=settings().effort_plan,
        schema=PLAN_SCHEMA,
        cache_system=True,
        node="plan",
    )
    parsed = result.parsed()

    # Sufficiency is a claim about having SQL. Without SQL it isn't one.
    sql = (parsed.get("sql") or "").strip()
    sufficient = bool(parsed.get("sufficient")) and bool(sql)

    used = set(parsed.get("used") or [])
    used_ids = [e["id"] for e in cache if e.get("name") in used and e.get("id")]
    missing = parsed.get("missing") or []

    emit(
        {
            "type": "plan",
            "cache_entries": len(cache),
            "sufficient": sufficient,
            "used": sorted(used),
            "missing": missing,
        }
    )
    out: TurnState = {
        "sufficient": sufficient,
        "used_ids": used_ids,
        "missing": missing,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
    }
    if sufficient:
        emit({"type": "sql", "sql": sql, "assumptions": parsed.get("assumptions") or []})
        out["sql"] = sql
        out["assumptions"] = parsed.get("assumptions") or []
    return out


async def explore(state: TurnState) -> TurnState:
    """A ReAct loop over the introspection tools.

    Bounded by max_tool_calls — an unbounded loop is a demo that runs forever
    on stage, and the cap is also what makes T1's cost a known quantity.
    """
    emit = get_stream_writer()
    known = render_cache(state.get("cache", []))
    system = (
        EXPLORE_SYSTEM
        + dialect_note(state["dialect"])
        + (f"\n\n{known}" if known else "")
    )

    # Whatever `plan` could not resolve is exactly what this loop is for, so
    # say so — an incremental turn should not re-derive what is already cached.
    gaps = state.get("missing") or []
    ask = f"Question: {state['question']}"
    if gaps:
        ask += "\n\nStill unknown:\n" + "\n".join(f"- {g}" for g in gaps)
    messages: list[dict[str, Any]] = [{"role": "user", "content": ask}]
    tokens_in = tokens_out = calls = 0
    result: llm.Result | None = None

    async with db.target(state["connection_id"]) as conn:
        while calls < settings().max_tool_calls:
            result = await llm.complete(
                system=system,
                messages=messages,
                effort=settings().effort_explore,
                tools=tools.SCHEMAS,
                node="explore",
            )
            tokens_in += result.tokens_in
            tokens_out += result.tokens_out

            if not result.tool_uses:
                break

            messages.append(llm.assistant_turn(result))
            outcomes: list[tuple[str, str, bool]] = []
            for call in result.tool_uses:
                calls += 1
                # A span each, because the alternative is what the turn table
                # already shows: 24 introspection calls as the integer 24.
                with tracing.span(
                    name=f"tool.{call.name}", input=call.input, as_type="tool"
                ) as sp:
                    payload, is_error = await tools.run_tool(conn, call.name, call.input)
                    sp.update(output=payload, level="WARNING" if is_error else None)
                emit(
                    {
                        "type": "explore",
                        "tool": call.name,
                        "input": call.input,
                        "error": is_error,
                        "calls": calls,
                    }
                )
                outcomes.append((call.id, payload, is_error))
            messages.extend(llm.tool_results(outcomes))

        # Hitting the cap means the model was mid-investigation and never got
        # to write its summary. Ask for it explicitly rather than handing
        # generate_sql "(no findings)" and letting it write SQL blind.
        if result is not None and result.tool_uses:
            messages.append(llm.assistant_turn(result))
            messages.extend(
                llm.tool_results(
                    [
                        (t.id, json.dumps({"error": "exploration budget spent"}), True)
                        for t in result.tool_uses
                    ]
                )
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That's the whole exploration budget. Summarise what you "
                        "found and say plainly what you could not confirm."
                    ),
                }
            )
            result = await llm.complete(
                system=system,
                messages=messages,
                effort=settings().effort_explore,
                node="explore.summary",
            )
            tokens_in += result.tokens_in
            tokens_out += result.tokens_out

    findings = (result.text if result else "") or "(no findings)"
    emit({"type": "findings", "text": findings, "tool_calls": calls})
    return {
        "findings": findings,
        "explored": True,
        "tool_calls": calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


async def generate_sql(state: TurnState) -> TurnState:
    emit = get_stream_writer()
    result = await llm.complete(
        system=SQL_SYSTEM + dialect_note(state["dialect"]),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"Findings:\n{state.get('findings', '')}"
                ),
            }
        ],
        effort=settings().effort_sql,
        schema=SQL_SCHEMA,
        node="generate_sql",
    )
    parsed = result.parsed()
    emit({"type": "sql", "sql": parsed["sql"], "assumptions": parsed["assumptions"]})
    return {
        "sql": parsed["sql"],
        "assumptions": parsed["assumptions"],
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
    }


def driver_message(e: BaseException) -> str:
    """The error text the model is shown in `fix`, and the text a user sees in
    `answer` when the turn gives up.

    SQLAlchemy wraps the driver's exception: `str()` prepends
    `(psycopg.errors.UndefinedColumn)`, appends `[SQL: <the whole query>]` and a
    docs link. `fix` already re-sends the SQL itself, so the echo is the same
    context twice, and "Background on this error at: https://sqlalche.me/..." is
    the last thing a user should read when their question failed. `.orig` is the
    one sentence a database user would recognise, and it is exactly what this
    node produced before the port.
    """
    orig = getattr(e, "orig", None)
    return str(orig if orig is not None else e).strip()


async def execute(state: TurnState) -> TurnState:
    """Run the generated SQL under a read-only transaction with a timeout."""
    emit = get_stream_writer()
    try:
        with tracing.span(name="sql.execute", input=state["sql"]) as sp:
            async with db.target_readonly(state["connection_id"]) as conn:
                # exec_driver_sql, never text(). `text()` reads `:name` as a bind
                # parameter, and this string is whatever the model wrote — a
                # literal like `WHERE status = ':pending'` would become a
                # missing-parameter error instead of the SQL error the fix node
                # knows how to react to. Postgres `::` casts happen to survive
                # text()'s regex, which makes the failure rare enough to ship and
                # confusing enough to lose a day to. exec_driver_sql hands the
                # string to the driver untouched, exactly as psycopg did.
                result = await conn.exec_driver_sql(state["sql"])
                # A statement that returned no result set — the model
                # occasionally writes an EXPLAIN — has already closed its cursor,
                # and .mappings() on a closed one raises ResourceClosedError,
                # which reads to `fix` as a driver fault rather than as "that
                # isn't a SELECT".
                #
                # Drained inside the `async with`: the connection returns to the
                # pool on exit and the result closes with it. dict(m) because
                # RowMapping is dict-*like* and json.dumps will not serialise it.
                fetched = (
                    [dict(m) for m in result.mappings().fetchmany(settings().max_rows)]
                    if result.returns_rows
                    else []
                )
            rows = json.loads(json.dumps(fetched, default=str))
            # The span keeps a preview; a trace does not need fifty rows to be
            # legible, and the client is the thing that has to show the result.
            sp.update(output={"count": len(rows), "rows": rows[:5]})
        emit(
            {
                "type": "rows",
                "count": len(rows),
                # Every row the model saw, not a hardcoded five. Five was
                # silently a different number from `max_rows`, so a 20-row answer
                # rendered as 5 with nothing on the wire saying so.
                "rows": rows,
                # `fetchmany(max_rows)` cannot tell a full page from a result
                # that happened to be exactly that long, so the honest thing a
                # client can say is "more may exist" — never a total we never had.
                "capped": len(rows) == settings().max_rows,
            }
        )
        return {"rows": rows, "error": ""}
    except Exception as e:
        # Including programming errors: a bad column name is exactly what the
        # fix node exists to react to.
        message = driver_message(e)
        emit({"type": "error", "message": message})
        return {"error": message, "rows": []}


async def fix(state: TurnState) -> TurnState:
    """Error text plus SQL back to the model, ≤3 attempts."""
    emit = get_stream_writer()
    attempt = state.get("fix_attempts", 0) + 1
    result = await llm.complete(
        system=FIX_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"Findings:\n{state.get('findings', '')}\n\n"
                    f"SQL that failed:\n{state['sql']}\n\n"
                    f"{DIALECT_LABEL[state['dialect']]} error:\n{state['error']}"
                ),
            }
        ],
        effort=settings().effort_sql,
        schema=FIX_SCHEMA,
        node="fix",
    )
    parsed = result.parsed()
    emit(
        {
            "type": "fix",
            "attempt": attempt,
            "was_wrong": parsed["what_was_wrong"],
            "sql": parsed["sql"],
        }
    )
    return {
        "sql": parsed["sql"],
        "fix_attempts": attempt,
        "error": "",
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
    }


_TOKEN = re.compile(r"'(?:[^']|'')*'|[A-Za-z_][A-Za-z0-9_.]*|\d+|[^\s\w]")


def _tokens(sql: str) -> list[str]:
    """Tokens for the subsequence gate, with string literals kept verbatim.

    Everything used to be lower-cased. That is right for identifiers and
    keywords, which SQL folds, and it is wrong for literals — which is trap 2:
    `customer.region` holds `west`, `West` and `WEST`, so a recipe claiming
    `region = 'west'` would be marked *verified* against SQL that filtered on
    `'WEST'`. The gate exists to catch a recipe saying something the query did
    not; a fold that erases the difference the demo is built on is the gate
    lying.

    `casefold()` rather than `lower()` for the rest: `İ` folds to `i̇`, and a
    Turkish column name is not this function's problem to get wrong.
    """
    return [
        t if t.startswith("'") else t.casefold()
        for t in _TOKEN.findall(sql or "")
    ]


def grounded_in(fragment: str | None, sql: str) -> bool:
    """Does this recipe's fragment actually appear in the SQL that ran?

    This is the whole verification gate, and it exists because of a specific
    observed failure: the model reported revenue excluding pending orders while
    its SQL excluded only cancelled and refunded — a 16% gap between the answer
    and the query. Prose is a claim; executed SQL is evidence. Only evidence
    marks an entry `verified`.

    Matching is an **order-preserving token subsequence**, not a substring.
    Substring matching was the first attempt and it was too brittle to be
    useful: a recipe of `COUNT(*) FROM customer WHERE deleted_at IS NULL` failed
    against `SELECT COUNT(*) AS active_customer_count FROM customer WHERE
    deleted_at IS NULL`, because the alias sits between two adjacent fragment
    tokens. A gate that rejects correct recipes is no more use than one that
    accepts wrong ones.

    Subsequence matching tolerates aliases, whitespace and formatting while
    still rejecting a fragment that names anything the query never mentioned —
    an invented `'pending'` has no token to match, wherever you look.

    Unverified entries are still written — they're useful context, and §5's
    compaction is what eventually drops the ones nothing uses. They just don't
    carry the authority a verified recipe does.
    """
    if not fragment:
        return False
    wanted = _tokens(fragment)
    if not wanted:
        return False
    ran = iter(_tokens(sql))
    return all(token in ran for token in wanted)


async def infer_tables(sql: str, connection_id: str) -> list[str]:
    """Which real tables does this SQL touch?

    The model is asked for `tables` and sometimes returns an empty list. Left
    alone that silently disables drift detection — `schema_fp` becomes a hash
    over nothing, so the entry can never go stale (§5) no matter what happens
    to the schema it depends on.
    """
    async with db.target(connection_id) as conn:
        known = await conn.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
    # Folded on both sides, and the *stored* spelling is what comes back.
    # Postgres folds unquoted identifiers to lower case, so lower-casing both
    # sides was right there and wrong everywhere else: MySQL on a
    # case-sensitive filesystem stores `Orders` as `Orders`, and the old
    # intersection returned [] for it. That does not fail — it quietly switches
    # off drift detection for every entry the turn writes, which is exactly what
    # this function exists to prevent.
    by_fold = {n.casefold(): n for n in known}
    seen = {t.casefold() for t in _tokens(sql)}
    return sorted(by_fold[k] for k in by_fold.keys() & seen)


async def extract(state: TurnState) -> TurnState:
    """Write down what this turn learned, so the next one doesn't re-learn it."""
    emit = get_stream_writer()
    if state.get("error") or not state.get("sql"):
        return {}

    # Nothing was learned. `plan` answering from cache without exploring means
    # the cache already held everything the question needed — re-deriving it
    # writes near-duplicates of entries that are right there, and bills a call
    # per turn to do it. This is a third of a cached turn's cost, and it recurs
    # forever. A turn that needed a fix *did* learn something, so it still runs.
    if state.get("sufficient") and not state.get("fix_attempts"):
        emit({"type": "learned", "count": 0, "skipped": 0, "entries": [], "cached": True})
        return {}

    # Show it what is already filed, so a refinement updates the existing entry
    # instead of landing beside it. Without this the model paraphrases its own
    # names — a live run produced "active customer count" on one turn and
    # "active customers count" on the next, which the upsert cannot merge.
    known = "\n".join(
        f"- {e['name']}: {e['claim']}"
        for e in state.get("cache", [])
        if e.get("name")
    )
    filed = (
        f"\n\nAlready filed. Reuse a name only to refine that same concept — "
        f"reusing it replaces what is shown here:\n{known}"
        if known
        else ""
    )

    # Extraction is a bonus, not the deliverable. The question is already
    # answered by the time we get here, and throwing that away because the
    # model failed to *learn* from it would be a bad trade — the turn would
    # report failure to a user who has a correct answer sitting right there.
    try:
        result = await llm.complete(
            system=EXTRACT_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n\n"
                        f"SQL that ran:\n{state['sql']}\n\n"
                        f"Schema findings:\n{state.get('findings', '(none)')}"
                        f"{filed}"
                    ),
                }
            ],
            effort=settings().effort_extract,
            schema=EXTRACT_SCHEMA,
            node="extract",
        )
        parsed = result.parsed().get("entries", [])
    except Exception as e:
        emit(
            {
                "type": "learned",
                "count": 0,
                "skipped": 0,
                "entries": [],
                "failed": f"{type(e).__name__}: {e}",
            }
        )
        return {}

    sql = state["sql"]
    fallback_tables = await infer_tables(sql, state["connection_id"])
    entries = [
        store.CacheEntry(
            kind=e["kind"],
            name=(e.get("name") or None),
            claim=e["claim"],
            sql_fragment=e.get("sql_fragment") or None,
            tables=e.get("tables") or fallback_tables,
            verified=grounded_in(e.get("sql_fragment"), sql),
        )
        for e in parsed
    ]

    # The one operation that spans both databases, and the order is forced: the
    # fingerprint is a fact about the business schema, the entry it lands on is
    # the agent's own memory, and no single connection reaches both.
    async with db.target(state["connection_id"]) as conn:
        await store.fingerprint_entries(conn, entries)
    async with db.agent() as conn:
        written = await store.write_entries(
            conn,
            entries,
            connection_id=state["connection_id"],
            turn_id=state["turn_id"],
        )

    emit(
        {
            "type": "learned",
            "count": len(written),
            "skipped": len(entries) - len(written),
            "entries": [
                {
                    "name": e.name,
                    "kind": e.kind,
                    "claim": e.claim,
                    "verified": e.verified,
                }
                for e in entries
                if e.id is not None
            ],
        }
    )
    return {"tokens_in": result.tokens_in, "tokens_out": result.tokens_out}


async def answer(state: TurnState) -> TurnState:
    emit = get_stream_writer()
    tokens_in = tokens_out = 0

    if state.get("error"):
        text = (
            f"I couldn't answer that — the query kept failing. "
            f"Last error: {state['error']}"
        )
    else:
        result = await llm.complete(
            system=ANSWER_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n\n"
                        f"SQL:\n{state.get('sql', '')}\n\n"
                        f"Result: {json.dumps(state.get('rows', []))}\n\n"
                        f"Assumptions: {json.dumps(state.get('assumptions', []))}"
                    ),
                }
            ],
            effort=settings().effort_extract,
            node="answer",
        )
        text, tokens_in, tokens_out = result.text, result.tokens_in, result.tokens_out

    total_in = state.get("tokens_in", 0) + tokens_in
    total_out = state.get("tokens_out", 0) + tokens_out
    latency = int((time.monotonic() - state.get("started_at", time.monotonic())) * 1000)

    async with db.agent() as conn:
        # Credit the entries this turn leaned on, but only now — an entry that
        # fed a query which never ran has not earned a hit, and `hits` both
        # orders the cache and shows blast radius in /admin/cache.
        if not state.get("error"):
            await store.bump_hits(
                conn,
                state.get("used_ids") or [],
                connection_id=state["connection_id"],
                turn_id=state["turn_id"],
            )
        await store.finish_turn(
            conn,
            state["turn_id"],
            sql=state.get("sql"),
            answer=text,
            tool_calls=state.get("tool_calls", 0),
            explored=state.get("explored", False),
            tokens_in=total_in,
            tokens_out=total_out,
            latency_ms=latency,
            cache_entries=len(state.get("cache", [])),
            trace_id=state.get("trace_id") or None,
        )

    emit(
        {
            "type": "answer",
            "text": text,
            "sql": state.get("sql"),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "total_tokens": total_in + total_out,
            "latency_ms": latency,
            "explored": state.get("explored", False),
        }
    )
    return {"answer": text, "tokens_in": tokens_in, "tokens_out": tokens_out}


# ----------------------------------------------------------------------- edges


def route_after_plan(state: TurnState) -> str:
    """The branch the whole demo rests on."""
    return "execute" if state.get("sufficient") else "explore"


def route_after_execute(state: TurnState) -> str:
    if state.get("error"):
        if state.get("fix_attempts", 0) < settings().max_fix_attempts:
            return "fix"
        return "answer"  # gave up; nothing worth learning from a query that never ran
    return "extract"


async def stream_turn(compiled, session_id: str, question: str, connection_id: str):
    """Drive one turn, yielding UI events. Never raises.

    A model timeout or a dropped connection would otherwise surface as a
    traceback and leave the turn row open. On stage that reads as the demo
    crashing; here it reads as one turn that failed, and the next question
    still works.

    This is also the turn boundary for tracing, and the `with` sits *outside* the
    `try` deliberately: app/api.py breaks out of this generator when the client
    disconnects, so the span has to close on `aclose()` too and not only on the
    paths that reach the end.
    """
    with tracing.turn(
        session_id=session_id, question=question, connection_id=connection_id
    ) as trace:
        try:
            async for mode, chunk in compiled.astream(
                {
                    "session_id": session_id,
                    "question": question,
                    "connection_id": connection_id,
                    "trace_id": trace.trace_id or "",
                },
                stream_mode=["updates", "custom"],
                config={"configurable": {"thread_id": session_id}},
            ):
                if mode == "custom" and isinstance(chunk, dict):
                    # The turn's own output, taken off the event stream rather
                    # than out of the final state — `answer` computes the totals
                    # and this is where they arrive already assembled.
                    if chunk.get("type") == "answer":
                        trace.update(output=chunk)
                    yield chunk
                else:
                    # Per-node token deltas, so the counter climbs during
                    # exploration rather than jumping once at the end.
                    for ev in to_events(mode, chunk):
                        yield ev
        except Exception as e:
            # Transport errors often stringify to nothing (httpx.ReadTimeout), so
            # the class name has to carry the meaning.
            detail = str(e).strip()
            message = f"{type(e).__name__}{': ' + detail if detail else ''}"
            trace.update(level="ERROR", status_message=message)
            try:
                async with db.agent() as conn:
                    await store.fail_open_turn(
                        conn,
                        session_id,
                        f"failed — {message}",
                        connection_id=connection_id,
                    )
            except Exception:  # the turn is already lost; don't lose the event too
                pass
            yield {"type": "error", "message": message, "fatal": True}
    yield {"type": "done"}


def traced(node):
    """A node, wrapped in a span named after it.

    The nesting the trace shows is just Python's call stack, so nothing here
    needs to know about the generations and tool spans that open underneath it.
    Written by hand rather than taken from `langfuse.langchain.CallbackHandler`,
    which needs the whole `langchain` package: this project has langgraph and
    langchain-core only, and a meta-package pulled in to draw a box on a diagram
    would be the largest dependency in the lock file.

    The span's output is the node's returned *delta* — the same thing LangGraph
    streams as an `updates` chunk, and the smallest honest answer to "what did
    this node do".
    """

    @functools.wraps(node)
    async def wrapper(state: TurnState) -> TurnState:
        with tracing.span(name=node.__name__) as sp:
            out = await node(state)
            sp.update(output=out)
            return out

    return wrapper


def build_graph(checkpointer: AsyncPostgresSaver | None = None):
    g = StateGraph(TurnState)
    for node in (
        load_cache, plan, explore, generate_sql, execute, fix, extract, answer
    ):
        # functools.wraps keeps __name__, which is what names the node — the
        # graph's shape does not change because it is being watched.
        g.add_node(node.__name__, traced(node))

    g.add_edge(START, "load_cache")
    g.add_edge("load_cache", "plan")
    g.add_conditional_edges(
        "plan", route_after_plan, {"execute": "execute", "explore": "explore"}
    )
    g.add_edge("explore", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_conditional_edges(
        "execute",
        route_after_execute,
        {"fix": "fix", "extract": "extract", "answer": "answer"},
    )
    g.add_edge("fix", "execute")
    g.add_edge("extract", "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer)
