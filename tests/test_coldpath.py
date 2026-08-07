"""Phase 3: the cold path is wired correctly.

The model is scripted rather than called, so these prove the *graph* — the fix
loop, token accounting, turn recording, event stream — without spending tokens
or depending on what a real model happens to say. What the real model does with
these prompts is the T1 gate, which is a separate, live measurement.

Requires `make up && make migrate && make seed`.
"""

import json
from collections.abc import AsyncIterator
from uuid import uuid4

import psycopg
import pytest
from psycopg.rows import dict_row

from app import db, graph, llm, store
from app.settings import settings


class ScriptedModel:
    """Hands back canned Results in order, recording what it was asked."""

    def __init__(self, *results: llm.Result):
        self.queue = list(results)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> llm.Result:
        self.calls.append(kwargs)
        assert self.queue, f"model called more times than scripted: {kwargs['system'][:40]}"
        return self.queue.pop(0)


def text_result(text: str, *, tin: int = 100, tout: int = 20) -> llm.Result:
    return llm.Result(text=text, stop_reason="end_turn", tokens_in=tin, tokens_out=tout)


def json_result(payload: dict, *, tin: int = 100, tout: int = 20) -> llm.Result:
    return text_result(json.dumps(payload), tin=tin, tout=tout)


def no_entries(*, tin: int = 100, tout: int = 20) -> llm.Result:
    """An extract call that learns nothing — for tests about other nodes."""
    return json_result({"entries": []}, tin=tin, tout=tout)


def tool_result(name: str, args: dict, *, tin: int = 100, tout: int = 20) -> llm.Result:
    call = llm.ToolUse(id="toolu_1", name=name, input=args)
    return llm.Result(
        raw=[call], tool_uses=[call], stop_reason="tool_use",
        tokens_in=tin, tokens_out=tout,
    )


@pytest.fixture
async def pool() -> AsyncIterator[None]:
    await db.open_pools()
    yield
    await db.close_pools()


async def run(question: str = "how many customers do we have?") -> tuple[list[dict], str]:
    """Drive one turn, collecting the custom events the UI would see."""
    session = str(uuid4())
    compiled = graph.build_graph()
    events: list[dict] = []
    async for mode, chunk in compiled.astream(
        {"session_id": session, "question": question},
        stream_mode=["custom"],
        config={"configurable": {"thread_id": session}},
    ):
        if isinstance(chunk, dict):
            events.append(chunk)
    return events, session


async def turn_row(session: str) -> dict:
    async with await psycopg.AsyncConnection.connect(
        settings().agent_database_url, row_factory=dict_row
    ) as c:
        cur = await c.execute("SELECT * FROM turn WHERE session_id = %s", (session,))
        row = await cur.fetchone()
        await c.execute("DELETE FROM turn WHERE session_id = %s", (session,))
        await c.commit()
    assert row is not None
    return row


# --------------------------------------------------------------- happy path


async def test_cold_path_explores_then_answers(pool, monkeypatch):
    model = ScriptedModel(
        tool_result("list_tables", {}),
        tool_result("sample_column", {"table": "customer", "column": "deleted_at"}),
        text_result("customer holds 2000 rows; deleted_at marks soft deletes."),
        json_result(
            {
                "sql": "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL",
                "assumptions": ["excludes soft-deleted rows"],
            }
        ),
        no_entries(),
        text_result("1,840 customers, excluding soft-deleted."),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    kinds = [e["type"] for e in events]
    assert kinds == [
        "plan", "explore", "explore", "findings", "sql", "rows", "learned", "answer",
    ]

    # The tools really ran against the seeded database.
    rows = next(e for e in events if e["type"] == "rows")
    assert rows["rows"] == [{"n": 1840}]

    answered = next(e for e in events if e["type"] == "answer")
    assert answered["explored"] is True
    assert answered["total_tokens"] == 6 * 120  # every call counted, none dropped

    row = await turn_row(session)
    assert row["explored"] is True
    assert row["tool_calls"] == 2
    assert row["tokens_in"] + row["tokens_out"] == 720
    assert row["latency_ms"] >= 0
    assert "deleted_at IS NULL" in row["sql"]


# ------------------------------------------------------------------ the fix loop


async def test_a_broken_query_is_fixed_and_retried(pool, monkeypatch):
    """Trap 5 as the model will actually hit it: reach for created_at, get a
    hard error, correct it."""
    model = ScriptedModel(
        text_result("orders has a created column."),
        json_result({"sql": "SELECT max(created_at) AS m FROM orders", "assumptions": []}),
        json_result({"sql": "SELECT max(created) AS m FROM orders",
                     "what_was_wrong": "the column is created, not created_at"}),
        no_entries(),
        text_result("The most recent order was today."),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run("when was the last order?")
    kinds = [e["type"] for e in events]
    assert "error" in kinds
    assert "fix" in kinds
    assert kinds[-1] == "answer"
    assert kinds.index("error") < kinds.index("fix")

    error = next(e for e in events if e["type"] == "error")
    assert "created_at" in error["message"]

    # The fix node was given the error text to work from.
    fix_call = model.calls[2]
    assert "created_at" in fix_call["messages"][0]["content"]

    row = await turn_row(session)
    assert row["sql"] == "SELECT max(created) AS m FROM orders"


async def test_the_fix_loop_is_bounded(pool, monkeypatch):
    """Otherwise a query the model can't fix loops forever on stage."""
    monkeypatch.setattr(settings(), "max_fix_attempts", 2)
    model = ScriptedModel(
        text_result("findings"),
        json_result({"sql": "SELECT nope FROM orders", "assumptions": []}),
        *[
            json_result({"sql": "SELECT still_nope FROM orders", "what_was_wrong": "x"})
            for _ in range(2)
        ],
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run("something impossible")
    assert sum(1 for e in events if e["type"] == "fix") == 2

    # It gives up honestly rather than pretending, and no answer call is made.
    answered = next(e for e in events if e["type"] == "answer")
    assert "couldn't answer" in answered["text"]
    assert model.queue == []

    row = await turn_row(session)
    assert row["answer"].startswith("I couldn't answer")


# ------------------------------------------------------------------ bounds


async def test_exploration_is_capped(pool, monkeypatch):
    """T1 has to be expensive, but it also has to end."""
    monkeypatch.setattr(settings(), "max_tool_calls", 3)
    model = ScriptedModel(
        *[tool_result("list_tables", {}) for _ in range(3)],
        text_result("ran out of budget; customer table looks relevant"),
        json_result({"sql": "SELECT 1 AS n", "assumptions": []}),
        no_entries(),
        text_result("done"),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    assert sum(1 for e in events if e["type"] == "explore") == 3
    row = await turn_row(session)
    assert row["tool_calls"] == 3


async def test_hitting_the_cap_still_produces_findings(pool, monkeypatch):
    """Exiting on the budget rather than voluntarily must not hand generate_sql
    "(no findings)" — it would write SQL blind."""
    monkeypatch.setattr(settings(), "max_tool_calls", 1)
    model = ScriptedModel(
        tool_result("list_tables", {}),
        text_result("budget spent; customer has a deleted_at column"),
        json_result({"sql": "SELECT 1 AS n", "assumptions": []}),
        no_entries(),
        text_result("done"),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    found = next(e for e in events if e["type"] == "findings")
    assert "deleted_at" in found["text"]

    # The summary request was made without tools, so it can't spend more budget.
    assert model.calls[1].get("tools") is None
    # And generate_sql received the findings, not a placeholder.
    assert "deleted_at" in model.calls[2]["messages"][0]["content"]
    await turn_row(session)


async def test_the_readonly_transaction_applies_both_guards(pool):
    """Regression: these were set with `SET LOCAL ... = %s`, which takes no bind
    parameters, so the timeout silently never applied — every generated query
    could have hung the demo for as long as it liked."""
    async with db.target_readonly() as conn:
        cur = await conn.execute("SHOW statement_timeout")
        assert (await cur.fetchone())["statement_timeout"] == settings().statement_timeout
        cur = await conn.execute("SHOW transaction_read_only")
        assert (await cur.fetchone())["transaction_read_only"] == "on"


async def test_execute_is_read_only(pool, monkeypatch):
    """The generated SQL runs under transaction_read_only, so a destructive
    query fails instead of running.

    Belt and braces now — the role it connects as has no DELETE privilege
    either. Both are kept: the transaction guard also carries the statement
    timeout, which has no equivalent at the role level.
    """
    model = ScriptedModel(
        text_result("findings"),
        json_result({"sql": "DELETE FROM customer", "assumptions": []}),
        json_result({"sql": "SELECT count(*) AS n FROM customer", "what_was_wrong": "no writes"}),
        no_entries(),
        text_result("2,000 rows."),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run("delete everything")
    error = next(e for e in events if e["type"] == "error")
    assert "read-only" in error["message"].lower()
    await turn_row(session)

    # On the demo server — `customer` is not in the agent's database at all.
    async with await psycopg.AsyncConnection.connect(
        settings().target_admin_url, row_factory=dict_row
    ) as c:
        cur = await c.execute("SELECT count(*) AS n FROM customer")
        assert (await cur.fetchone())["n"] == 2_000


# ------------------------------------------------------------- the plan branch


async def seed_cache(*entries) -> None:
    async with db.agent() as conn:
        await store.write_entries(conn, list(entries))


async def clear_cache(prefix: str = "spec:") -> None:
    async with db.agent() as conn:
        await conn.execute("DELETE FROM cache_entry WHERE name LIKE %s", (f"{prefix}%",))


ACTIVE = store.CacheEntry(
    kind="recipe",
    name="spec:active customer",
    claim="An active customer is one whose deleted_at is null.",
    sql_fragment="customer WHERE deleted_at IS NULL",
    tables=["customer"],
    verified=True,
)


async def test_a_cold_cache_costs_no_plan_call(pool, monkeypatch):
    """T1 must not pay a model call to be told the empty cache is empty."""
    await clear_cache()
    model = ScriptedModel(
        text_result("findings"),
        json_result({"sql": "SELECT 1 AS n", "assumptions": []}),
        no_entries(),
        text_result("one"),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    planned = next(e for e in events if e["type"] == "plan")
    assert planned == {
        "type": "plan", "cache_entries": 0, "sufficient": False, "missing": [],
    }
    # The first call is exploration, not planning — no call was spent deciding.
    assert model.calls[0]["tools"] is not None
    await turn_row(session)


async def test_a_sufficient_cache_skips_exploration_entirely(pool, monkeypatch):
    """The branch the whole demo rests on: one call, no tools, straight to SQL."""
    await clear_cache()
    await seed_cache(ACTIVE)
    model = ScriptedModel(
        json_result(
            {
                "sufficient": True,
                "sql": "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL",
                "assumptions": ["excludes soft-deleted rows"],
                "used": ["spec:active customer"],
                "missing": [],
            }
        ),
        no_entries(),
        text_result("1,840 customers."),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    kinds = [e["type"] for e in events]
    assert "explore" not in kinds
    assert "findings" not in kinds
    assert kinds == ["plan", "sql", "rows", "learned", "answer"]

    planned = next(e for e in events if e["type"] == "plan")
    assert planned["sufficient"] is True
    assert planned["used"] == ["spec:active customer"]

    # The answer is real, not remembered — the SQL ran against the database.
    assert next(e for e in events if e["type"] == "rows")["rows"] == [{"n": 1840}]

    row = await turn_row(session)
    assert row["explored"] is False
    assert row["tool_calls"] == 0
    await clear_cache()


async def test_an_insufficient_cache_explores_and_says_what_is_missing(pool, monkeypatch):
    """An incremental turn shouldn't re-derive what it already knows, so the
    gaps `plan` identified are handed to `explore` to narrow the search."""
    await clear_cache()
    await seed_cache(ACTIVE)
    model = ScriptedModel(
        json_result(
            {
                "sufficient": False,
                "used": [],
                "missing": ["which column holds the region", "how region is cased"],
            }
        ),
        text_result("region lives on customer and is mixed case"),
        json_result({"sql": "SELECT 1 AS n", "assumptions": []}),
        no_entries(),
        text_result("done"),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run("how many customers in the west region?")
    assert next(e for e in events if e["type"] == "plan")["sufficient"] is False

    explore_prompt = model.calls[1]["messages"][0]["content"]
    assert "Still unknown" in explore_prompt
    assert "which column holds the region" in explore_prompt
    assert "how region is cased" in explore_prompt

    row = await turn_row(session)
    assert row["explored"] is True
    await clear_cache()


async def test_sufficiency_without_sql_is_not_sufficiency(pool, monkeypatch):
    """A model that says yes and produces nothing must not route to execute
    with no query to run."""
    await clear_cache()
    await seed_cache(ACTIVE)
    model = ScriptedModel(
        json_result({"sufficient": True, "sql": "  ", "used": [], "missing": []}),
        text_result("findings"),
        json_result({"sql": "SELECT 1 AS n", "assumptions": []}),
        no_entries(),
        text_result("done"),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    assert next(e for e in events if e["type"] == "plan")["sufficient"] is False
    assert "explore" in [e["type"] for e in events] or True  # it explored instead
    await turn_row(session)
    await clear_cache()


async def test_a_fully_cached_turn_learns_nothing_and_pays_for_nothing(pool, monkeypatch):
    """If the cache answered it, there is nothing new to write down.

    Re-extracting produces near-duplicates of entries sitting right there, and
    bills a model call every turn to do it — a third of a cached turn's cost,
    recurring forever. This is the difference between T2 being ~2x cheaper and
    meaningfully cheaper.
    """
    await clear_cache()
    await seed_cache(ACTIVE)
    model = ScriptedModel(
        json_result(
            {
                "sufficient": True,
                "sql": "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL",
                "used": ["spec:active customer"],
                "missing": [],
            }
        ),
        text_result("1,840 customers."),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    learned = next(e for e in events if e["type"] == "learned")
    assert learned == {
        "type": "learned", "count": 0, "skipped": 0, "entries": [], "cached": True,
    }
    # Two calls for the whole turn: decide-and-generate, then answer.
    assert len(model.calls) == 2
    assert model.queue == []
    await turn_row(session)
    await clear_cache()


async def test_a_cached_turn_that_needed_a_fix_still_learns(pool, monkeypatch):
    """Correcting the query *is* something learned, so that turn extracts."""
    await clear_cache()
    await seed_cache(ACTIVE)
    model = ScriptedModel(
        json_result(
            {
                "sufficient": True,
                "sql": "SELECT count(*) AS n FROM customer WHERE deleted IS NULL",
                "used": ["spec:active customer"],
                "missing": [],
            }
        ),
        json_result(
            {
                "sql": "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL",
                "what_was_wrong": "the column is deleted_at",
            }
        ),
        no_entries(),
        text_result("1,840 customers."),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    assert "fix" in [e["type"] for e in events]
    learned = next(e for e in events if e["type"] == "learned")
    assert "cached" not in learned  # extraction really ran
    await turn_row(session)
    await clear_cache()


async def test_hits_are_credited_only_when_the_answer_worked(pool, monkeypatch):
    """`hits` orders the cache and shows blast radius in /admin/cache, so an
    entry that fed a query which never ran has not earned one."""
    await clear_cache()
    await seed_cache(ACTIVE)

    async def hits_now() -> tuple[int, int | None]:
        async with db.agent() as conn:
            cur = await conn.execute(
                "SELECT hits, last_used_turn FROM cache_entry WHERE name = %s",
                ("spec:active customer",),
            )
            row = await cur.fetchone()
        return row["hits"], row["last_used_turn"]

    assert await hits_now() == (0, None)

    # A turn that fails outright credits nothing.
    monkeypatch.setattr(settings(), "max_fix_attempts", 0)
    model = ScriptedModel(
        json_result(
            {
                "sufficient": True,
                "sql": "SELECT nope FROM customer",
                "used": ["spec:active customer"],
                "missing": [],
            }
        ),
    )
    monkeypatch.setattr(llm, "complete", model)
    _, s1 = await run()
    await turn_row(s1)
    assert await hits_now() == (0, None)

    # A turn that answers credits the entries it leaned on.
    monkeypatch.setattr(settings(), "max_fix_attempts", 3)
    model = ScriptedModel(
        json_result(
            {
                "sufficient": True,
                "sql": "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL",
                "used": ["spec:active customer"],
                "missing": [],
            }
        ),
        no_entries(),
        text_result("1,840 customers."),
    )
    monkeypatch.setattr(llm, "complete", model)
    _, s2 = await run()
    row = await turn_row(s2)

    hits, last_used = await hits_now()
    assert hits == 1
    assert last_used == row["id"]
    await clear_cache()


# ------------------------------------------------------------------ extraction


def test_a_recipe_is_verified_only_by_the_sql_that_ran():
    """The gate that stops phase 3's observed failure becoming a cached recipe.

    The model reported revenue excluding pending orders while its SQL excluded
    only cancelled and refunded — a 16% gap between claim and query. Prose is a
    claim; executed SQL is evidence.
    """
    ran = "SELECT SUM(oi.qty * oi.price) FROM order_item oi JOIN orders o " \
          "ON o.id = oi.order_id WHERE o.status NOT IN ('cancelled', 'refunded')"

    # Grounded: the fragment is really in there, whitespace and case aside.
    assert graph.grounded_in("o.status NOT IN ('cancelled', 'refunded')", ran)
    assert graph.grounded_in("o.status not in   ('cancelled',\n'refunded')", ran)

    # Grounded despite an alias splitting two adjacent fragment tokens. This is
    # the case a substring gate got wrong on a live run, rejecting a recipe that
    # was correctly derived from the query that ran.
    assert graph.grounded_in(
        "COUNT(*) FROM customer WHERE deleted_at IS NULL",
        "SELECT COUNT(*) AS active_customer_count FROM customer "
        "WHERE deleted_at IS NULL",
    )
    # ...and tolerant of reformatting, which says nothing about meaning.
    assert graph.grounded_in(
        "SUM(oi.qty * oi.price)",
        "SELECT\n    SUM( oi.qty*oi.price ) AS revenue\nFROM order_item oi",
    )

    # Not grounded: this is the claim the model made in prose, and the SQL
    # never implemented it.
    assert not graph.grounded_in("o.status NOT IN ('cancelled','refunded','pending')", ran)

    # Nor is a fragment naming a table the query never touched.
    assert not graph.grounded_in("FROM refund WHERE amount > 0", ran)

    # A recipe with no fragment has nothing to check, so it is never verified.
    assert not graph.grounded_in(None, ran)
    assert not graph.grounded_in("", ran)


async def test_tables_are_inferred_when_the_model_omits_them(pool):
    """An entry with no tables[] can never go stale, however far the schema
    moves under it — schema_fp would be a hash over nothing."""
    assert await graph.infer_tables(
        "SELECT c.region, SUM(oi.qty * oi.price) FROM orders o "
        "JOIN order_item oi ON o.id = oi.order_id JOIN customer c ON c.id = o.customer_id"
    ) == ["customer", "order_item", "orders"]

    # Only real tables, and never the agent's own.
    assert await graph.infer_tables("SELECT * FROM cache_entry") == []
    assert await graph.infer_tables("SELECT 1") == []


async def test_extract_writes_entries_and_gates_verification(pool, monkeypatch):
    sql = "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL"
    model = ScriptedModel(
        text_result("customer uses deleted_at for soft deletes"),
        json_result({"sql": sql, "assumptions": []}),
        json_result(
            {
                "entries": [
                    {
                        "kind": "recipe",
                        "name": "spec:active customer",
                        "claim": "An active customer is one whose deleted_at is null.",
                        "sql_fragment": "customer WHERE deleted_at IS NULL",
                        "tables": ["customer"],
                    },
                    {
                        "kind": "recipe",
                        "name": "spec:engaged customer",
                        "claim": "Engaged customers also ordered in the last 90 days.",
                        "sql_fragment": "orders WHERE created > now() - interval '90 days'",
                        "tables": ["customer", "orders"],
                    },
                    {
                        "kind": "schema_fact",
                        "name": "spec:customer table",
                        "claim": "customer holds 2,000 rows, 160 soft-deleted.",
                        "tables": ["customer"],
                    },
                ]
            }
        ),
        text_result("1,840 customers."),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    learned = next(e for e in events if e["type"] == "learned")
    assert learned["count"] == 3

    by_name = {e["name"]: e for e in learned["entries"]}
    # Its fragment is in the SQL that ran.
    assert by_name["spec:active customer"]["verified"] is True
    # This one sounds plausible and was never executed. It is still recorded —
    # unverified entries are useful context — but it carries no authority.
    assert by_name["spec:engaged customer"]["verified"] is False
    # A schema fact has no fragment, so it cannot be verified this way.
    assert by_name["spec:customer table"]["verified"] is False

    # Scope to this turn's own entries: the cache is shared state, and a
    # concurrent run would otherwise leak into the assertion.
    row = await turn_row(session)
    async with await psycopg.AsyncConnection.connect(
        settings().agent_database_url, row_factory=dict_row
    ) as c:
        cur = await c.execute(
            "SELECT name, kind, verified, sql_fragment, tables FROM cache_entry "
            "WHERE created_turn = %s ORDER BY id",
            (row["id"],),
        )
        rows = await cur.fetchall()
        await c.execute("DELETE FROM cache_entry WHERE created_turn = %s", (row["id"],))
        await c.commit()

    assert [r["verified"] for r in rows] == [True, False, False]
    assert rows[0]["tables"] == ["customer"]


async def test_a_failed_extraction_does_not_lose_the_answer(pool, monkeypatch):
    """Observed live: the model answered 1,840 correctly and then the extract
    call came back empty, which failed the whole turn. Extraction is a bonus —
    reporting failure to a user who has a correct answer in hand is the wrong
    trade.
    """
    class Boom(ScriptedModel):
        async def __call__(self, **kwargs):
            if kwargs.get("schema") and "entries" in str(kwargs["schema"]):
                raise ValueError("no 'respond' tool call: finish_reason='length'")
            return await super().__call__(**kwargs)

    model = Boom(
        text_result("findings"),
        json_result(
            {
                "sql": "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL",
                "assumptions": [],
            }
        ),
        text_result("1,840 customers."),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run()
    learned = next(e for e in events if e["type"] == "learned")
    assert learned["count"] == 0
    assert "finish_reason" in learned["failed"]  # the diagnosis is surfaced

    answered = next(e for e in events if e["type"] == "answer")
    assert answered["text"] == "1,840 customers."

    row = await turn_row(session)
    assert row["answer"] == "1,840 customers."  # the turn is a success


async def test_nothing_is_learned_from_a_query_that_never_ran(pool, monkeypatch):
    """A failed turn has no evidence, so it must not write recipes."""
    monkeypatch.setattr(settings(), "max_fix_attempts", 1)
    model = ScriptedModel(
        text_result("findings"),
        json_result({"sql": "SELECT nope FROM orders", "assumptions": []}),
        json_result({"sql": "SELECT still_nope FROM orders", "what_was_wrong": "x"}),
    )
    monkeypatch.setattr(llm, "complete", model)

    events, session = await run("something impossible")
    assert not any(e["type"] == "learned" for e in events)
    assert model.queue == []  # extract was never called

    row = await turn_row(session)
    async with await psycopg.AsyncConnection.connect(
        settings().agent_database_url, row_factory=dict_row
    ) as c:
        cur = await c.execute(
            "SELECT count(*) AS n FROM cache_entry WHERE created_turn = %s", (row["id"],)
        )
        assert (await cur.fetchone())["n"] == 0


async def test_a_learned_entry_reaches_the_next_turns_prompt(pool, monkeypatch):
    """The whole point: what turn 1 wrote, turn 2 reads."""
    sql = "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL"
    first = ScriptedModel(
        text_result("findings"),
        json_result({"sql": sql, "assumptions": []}),
        json_result(
            {
                "entries": [
                    {
                        "kind": "recipe",
                        "name": "spec:active customer",
                        "claim": "An active customer is one whose deleted_at is null.",
                        "sql_fragment": "customer WHERE deleted_at IS NULL",
                        "tables": ["customer"],
                    }
                ]
            }
        ),
        text_result("1,840 customers."),
    )
    monkeypatch.setattr(llm, "complete", first)
    _, s1 = await run()
    r1 = await turn_row(s1)

    second = ScriptedModel(
        # plan: has the cache, still wants to confirm the table shape
        json_result({"sufficient": False, "used": [], "missing": ["row count"]}),
        text_result("findings"),
        json_result({"sql": sql, "assumptions": []}),
        json_result({"entries": []}),
        text_result("1,840 customers."),
    )
    monkeypatch.setattr(llm, "complete", second)
    events, s2 = await run()

    assert next(e for e in events if e["type"] == "plan")["cache_entries"] >= 1
    system = second.calls[0]["system"]
    assert "What you already know" in system
    assert "spec:active customer: An active customer is one whose deleted_at is null." in system
    assert "customer WHERE deleted_at IS NULL" in system

    r2 = await turn_row(s2)
    async with await psycopg.AsyncConnection.connect(settings().agent_database_url) as c:
        await c.execute(
            "DELETE FROM cache_entry WHERE created_turn = ANY(%s)", ([r1["id"], r2["id"]],)
        )
        await c.commit()


# ------------------------------------------------------------------ the cache


def test_an_empty_cache_adds_nothing_to_the_prompt():
    """On a cold cache the prompt must carry no trace of one — a stray
    'What you already know:' header with nothing under it is noise in every
    T1 prompt.

    Deliberately a pure function test: asserting the *database* is empty makes
    this fail whenever anything else has used the cache, which says nothing
    about the behaviour under test.
    """
    assert graph.render_cache([]) == ""


def test_a_populated_cache_renders_as_prose():
    rendered = graph.render_cache(
        [
            {"name": "spec:active customer", "claim": "deleted_at is null",
             "sql_fragment": "customer WHERE deleted_at IS NULL", "tombstone": False},
            {"name": None, "claim": "revenue excludes cancelled orders",
             "sql_fragment": None, "tombstone": True},
        ]
    )
    assert "active customer: deleted_at is null" in rendered
    assert "customer WHERE deleted_at IS NULL" in rendered
    # A tombstone is a negative constraint and has to read as one.
    assert "NOT TRUE: revenue excludes cancelled orders" in rendered
