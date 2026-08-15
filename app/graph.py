"""The turn graph (PLAN.md §4).

    load_cache → plan ─(sufficient)→ execute → extract → answer
                     └(insufficient)→ explore → generate_sql → execute
                                                 execute ⇄ fix (≤3 attempts)

`plan` is the branch the demo rests on: it answers from cache without exploring,
which is what makes T2 cheap.
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

from app import db, llm, prompts, store, tools, tracing
from app.config import config
from app.events import to_events


def _add(a: int, b: int) -> int:
    return a + b


class TurnState(TypedDict, total=False):
    session_id: str
    question: str
    # Which registered database this turn is about. In the state rather than in
    # `config["configurable"]` beside thread_id: state is checkpointed, and a
    # value outside it lets a resumed thread switch warehouses mid-conversation,
    # replaying a cache loaded from one while `execute` runs against another.
    connection_id: str
    # Which SQL the model should write. Set by `load_cache` from the registry
    # row's `driver` column, so nothing is dialled. In state for the same reason
    # as connection_id: a resumed thread must not plan in one dialect and
    # execute in another.
    dialect: str

    turn_id: int
    started_at: float
    cache: list[dict[str, Any]]
    # The Langfuse trace this turn is recorded as, or "" when tracing is off.
    # Minted in `stream_turn` and carried in rather than read out of the ambient
    # OpenTelemetry context, so writing it to the turn row does not depend on
    # that context reaching LangGraph's tasks.
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

    # Reduced across nodes, so the counter sums every API call the turn made.
    tokens_in: Annotated[int, _add]
    tokens_out: Annotated[int, _add]


# --------------------------------------------------------------------- prompts

# A label, not the driver name: the model needs to know which SQL to write, not
# which library dials the socket.
DIALECT_LABEL = {"postgresql": "PostgreSQL", "mysql": "MySQL", "sqlite": "SQLite"}

# Appended to the system prompt of every node that reads or writes SQL. Only
# what the model gets wrong often enough to be worth the tokens — every line is
# paid for on every turn.
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


# The prose is config/prompts/, which an optimiser can write through. Schemas
# stay here: they are the node's wire contract, not prose. So do `dialect_note`
# and `render_cache`, which compose per turn.

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
    """The prose the model reads.

    `verified` is marked on the SQL line, and only when *false* — a copied
    fragment is the common case, so the exception carries the information. A
    missing key reads as unverified, which understates authority.
    """
    if not entries:
        return ""
    lines = ["What you already know about this database:"]
    for e in entries:
        prefix = "NOT TRUE: " if e.get("tombstone") else ""
        label = f"{e['name']}: " if e.get("name") else ""
        lines.append(f"- {prefix}{label}{e['claim']}")
        if e.get("sql_fragment"):
            mark = "" if e.get("verified") else " (unverified)"
            lines.append(f"  SQL{mark}: {e['sql_fragment']}")
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
        # From the registry row: `driver` is a column, so the dialect is known
        # before anything is dialled.
        registered = await store.get_connection(conn, state["connection_id"])

    cache = [
        {
            "id": e.id,
            "name": e.name,
            "claim": e.claim,
            "sql_fragment": e.sql_fragment,
            "tombstone": e.tombstone,
            # `render_cache` marks the unverified ones, and the plan prompt tells
            # the model what that marking means.
            "verified": e.verified,
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

    On the cached path it also writes the SQL, in the same call — splitting the
    two costs a second round trip that re-sends the whole cache (§4 amendment).
    """
    emit = get_stream_writer()
    cache = state.get("cache", [])

    # A cold cache can only produce one answer, so don't pay a model call to
    # hear it. This is every T1, and it keeps the cold path cheap.
    if not cache:
        emit({"type": "plan", "cache_entries": 0, "sufficient": False, "missing": []})
        return {"sufficient": False}

    result = await llm.complete(
        # Anything added to this system block must be a function of
        # `connection_id` alone — prompt caching keys on an exact prefix. The
        # dialect and the cache text are; the question is not, so it stays in
        # the user message.
        system=(
            f"{prompts.get('plan')}{dialect_note(state['dialect'])}\n\n"
            f"{render_cache(cache)}"
        ),
        messages=[{"role": "user", "content": f"Question: {state['question']}"}],
        effort=config().effort_for("plan"),
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

    Bounded by max_tool_calls, which is what makes T1's cost a known quantity.
    """
    emit = get_stream_writer()
    known = render_cache(state.get("cache", []))
    system = (
        prompts.get("explore")
        + dialect_note(state["dialect"])
        + (f"\n\n{known}" if known else "")
    )

    # Whatever `plan` could not resolve is what this loop is for, so say so —
    # an incremental turn should not re-derive what is already cached.
    gaps = state.get("missing") or []
    ask = f"Question: {state['question']}"
    if gaps:
        ask += "\n\nStill unknown:\n" + "\n".join(f"- {g}" for g in gaps)
    messages: list[dict[str, Any]] = [{"role": "user", "content": ask}]
    tokens_in = tokens_out = calls = 0
    result: llm.Result | None = None

    async with db.target(state["connection_id"]) as conn:
        while calls < config().max_tool_calls:
            result = await llm.complete(
                system=system,
                messages=messages,
                effort=config().effort_for("explore"),
                tools=tools.SCHEMAS,
                node="explore",
            )
            tokens_in += result.tokens_in
            tokens_out += result.tokens_out

            if not result.tool_uses:
                break

            messages.append(llm.assistant_turn(result, node="explore"))
            outcomes: list[tuple[str, str, bool]] = []
            for call in result.tool_uses:
                calls += 1
                # A span each, so 24 introspection calls are not just the
                # integer 24 the turn table shows.
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
            messages.extend(llm.tool_results(outcomes, node="explore"))

        # Hitting the cap means the model never got to write its summary. Ask
        # for one rather than handing generate_sql "(no findings)".
        if result is not None and result.tool_uses:
            messages.append(llm.assistant_turn(result, node="explore"))
            messages.extend(
                llm.tool_results(
                    [
                        (t.id, json.dumps({"error": "exploration budget spent"}), True)
                        for t in result.tool_uses
                    ],
                    node="explore",
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
                effort=config().effort_for("explore"),
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
        system=prompts.get("generate_sql") + dialect_note(state["dialect"]),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"Findings:\n{state.get('findings', '')}"
                ),
            }
        ],
        effort=config().effort_for("generate_sql"),
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
    """The error text `fix` is shown, and what a user sees when the turn gives up.

    SQLAlchemy's `str()` prepends its own type name, appends the whole query and
    a docs link. `.orig` is the one sentence a database user would recognise.
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
                # parameter, and this string is whatever the model wrote, so
                # `WHERE status = ':pending'` becomes a missing-parameter error
                # instead of the SQL error `fix` knows how to react to. Postgres
                # `::` casts survive text()'s regex, which makes it rare.
                result = await conn.exec_driver_sql(state["sql"])
                # A statement returning no result set — the model occasionally
                # writes an EXPLAIN — has closed its cursor, and .mappings() on a
                # closed one raises ResourceClosedError, which reads to `fix` as
                # a driver fault rather than "that isn't a SELECT".
                #
                # Drained inside the `async with`, because the result closes with
                # the connection. dict(m) because json.dumps cannot take a
                # RowMapping.
                fetched = (
                    [dict(m) for m in result.mappings().fetchmany(config().max_rows)]
                    if result.returns_rows
                    else []
                )
            rows = json.loads(json.dumps(fetched, default=str))
            # A preview: a trace does not need fifty rows to be legible.
            sp.update(output={"count": len(rows), "rows": rows[:5]})
        emit(
            {
                "type": "rows",
                "count": len(rows),
                # Every row the model saw, so a client never renders fewer than
                # the answer was based on.
                "rows": rows,
                # `fetchmany(max_rows)` cannot tell a full page from a result
                # that happened to be exactly that long, so a client can only say
                # "more may exist" — never a total nothing ever counted.
                "capped": len(rows) == config().max_rows,
            }
        )
        return {"rows": rows, "error": ""}
    except Exception as e:
        # Including programming errors: a bad column name is what `fix` is for.
        message = driver_message(e)
        emit({"type": "error", "message": message})
        return {"error": message, "rows": []}


async def fix(state: TurnState) -> TurnState:
    """Error text plus SQL back to the model, ≤3 attempts."""
    emit = get_stream_writer()
    attempt = state.get("fix_attempts", 0) + 1
    result = await llm.complete(
        system=prompts.get("fix"),
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
        effort=config().effort_for("fix"),
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

    Identifiers and keywords fold because SQL folds them; literals must not.
    `customer.region` holds `west`, `West` and `WEST`, so a folded recipe
    claiming `region = 'west'` would verify against SQL filtering on `'WEST'`.
    """
    return [
        t if t.startswith("'") else t.casefold()
        for t in _TOKEN.findall(sql or "")
    ]


def grounded_in(fragment: str | None, sql: str) -> bool:
    """Does this recipe's fragment actually appear in the SQL that ran?

    The verification gate: prose is a claim, executed SQL is evidence, and only
    evidence marks an entry `verified`. The failure it catches is a model
    reporting revenue excluding pending orders while its SQL excluded only
    cancelled and refunded — a 16% gap between the answer and the query.

    An **order-preserving token subsequence**, not a substring, so an alias
    between two fragment tokens cannot reject a correct recipe. Unverified
    entries are still written; they just carry less authority.
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

    The model sometimes returns an empty `tables`, which would make `schema_fp`
    a hash over nothing — an entry that can never go stale (§5).
    """
    async with db.target(connection_id) as conn:
        known = await conn.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
    # Folded on both sides, and the *stored* spelling comes back. MySQL on a
    # case-sensitive filesystem stores `Orders` as `Orders`, and a lower-cased
    # intersection would return [] for it — switching off drift detection for
    # every entry the turn writes, which is what this function prevents.
    by_fold = {n.casefold(): n for n in known}
    seen = {t.casefold() for t in _tokens(sql)}
    return sorted(by_fold[k] for k in by_fold.keys() & seen)


# The two anchors `extract_message` writes and `optim/` splits back out. Named
# rather than inline because a harvested case is replayed verbatim and only the
# SQL is parsed back out of it — a literal duplicated across two packages drifts
# silently, and the metric would score recipes against the wrong query.
EXTRACT_SQL_ANCHOR = "SQL that ran:\n"
EXTRACT_FILED_ANCHOR = "\n\nAlready filed."


def extract_message(
    *, question: str, sql: str, findings: str, cache: list[dict[str, Any]]
) -> str:
    """The user turn `extract` sends. Pure, so a probe case can build a real one.

    Showing the model what is already filed stops it paraphrasing its own keys
    into "active customer count" and "active customers count", which the upsert
    cannot merge.
    """
    known = "\n".join(
        f"- {e['name']}: {e['claim']}" for e in cache if e.get("name")
    )
    filed = (
        f"{EXTRACT_FILED_ANCHOR} Reuse a name only to refine that same concept — "
        f"reusing it replaces what is shown here:\n{known}"
        if known
        else ""
    )
    return (
        f"Question: {question}\n\n"
        f"{EXTRACT_SQL_ANCHOR}{sql}\n\n"
        f"Schema findings:\n{findings}"
        f"{filed}"
    )


def entries_from(
    parsed: list[dict[str, Any]], sql: str, fallback_tables: list[str]
) -> list[store.CacheEntry]:
    """Model output → cache entries, with the verification gate applied.

    Pure, so the optimiser scores what production would actually write rather
    than a re-implementation that agrees until one of the two is edited.
    """
    return [
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


async def extract(state: TurnState) -> TurnState:
    """Write down what this turn learned, so the next one doesn't re-learn it."""
    emit = get_stream_writer()
    if state.get("error") or not state.get("sql"):
        return {}

    # Nothing was learned: `plan` answering from cache without exploring means
    # the cache already held what the question needed, and re-deriving it writes
    # near-duplicates at a model call per turn, forever. A turn that needed a fix
    # did learn something, so it still runs.
    if state.get("sufficient") and not state.get("fix_attempts"):
        emit({"type": "learned", "count": 0, "skipped": 0, "entries": [], "cached": True})
        return {}

    # Extraction is a bonus, not the deliverable: the question is already
    # answered by now, and failing the turn because the model could not learn
    # from it would report failure to a user who has a correct answer.
    try:
        result = await llm.complete(
            system=prompts.get("extract"),
            messages=[
                {
                    "role": "user",
                    "content": extract_message(
                        question=state["question"],
                        sql=state["sql"],
                        findings=state.get("findings", "(none)"),
                        cache=state.get("cache", []),
                    ),
                }
            ],
            effort=config().effort_for("extract"),
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
    entries = entries_from(parsed, sql, fallback_tables)

    # The one operation spanning both databases, and the order is forced: no
    # single connection reaches the target and the agent's own memory.
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
            system=prompts.get("answer"),
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
            effort=config().effort_for("answer"),
            node="answer",
        )
        text, tokens_in, tokens_out = result.text, result.tokens_in, result.tokens_out

    total_in = state.get("tokens_in", 0) + tokens_in
    total_out = state.get("tokens_out", 0) + tokens_out
    latency = int((time.monotonic() - state.get("started_at", time.monotonic())) * 1000)

    async with db.agent() as conn:
        # Credit the entries this turn leaned on, and only now: an entry that
        # fed a query which never ran has not earned a hit.
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
        if state.get("fix_attempts", 0) < config().max_fix_attempts:
            return "fix"
        return "answer"  # gave up; nothing worth learning from a query that never ran
    return "extract"


async def stream_turn(compiled, session_id: str, question: str, connection_id: str):
    """Drive one turn, yielding UI events. Never raises — a model timeout is one
    failed turn, not a traceback and a turn row left open.

    The tracing `with` sits *outside* the `try`: app/api.py breaks out of this
    generator on client disconnect, so the span has to close on `aclose()` too.
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
                    # Off the event stream rather than the final state, because
                    # `answer` computes the totals and they arrive assembled.
                    if chunk.get("type") == "answer":
                        trace.update(output=chunk)
                    yield chunk
                else:
                    # Per-node token deltas, so the counter climbs during
                    # exploration rather than jumping once at the end.
                    for ev in to_events(mode, chunk):
                        yield ev
        except Exception as e:
            # Transport errors often stringify to nothing (httpx.ReadTimeout),
            # so the class name has to carry the meaning.
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

    Hand-rolled rather than `langfuse.langchain.CallbackHandler`, which needs
    the whole `langchain` meta-package; this project has langchain-core only.
    The span's output is the node's returned delta.
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
        # functools.wraps keeps __name__, which is what names the node, so the
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
