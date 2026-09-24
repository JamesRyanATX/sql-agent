"""One whole turn under a candidate's text, recorded.

What the record has to get right is everything a metric and its feedback read:
which tools were called and in what order, how many times the SQL had to be
fixed, what it finally answered, what it cost. A record that quietly loses the
tool sequence would still score candidates — it would just score them all the
same, and the run would look like GEPA finding nothing.

Requires Postgres up (`make up && make migrate && make seed`). No model calls.
"""

from __future__ import annotations

import pytest

from app import db, llm, overrides, store
from tests.test_coldpath import (  # noqa: F401 — `pool` is a fixture
    ScriptedModel,
    json_result,
    no_entries,
    pool,
    text_result,
    tool_result,
)
from tools.gepa.replay_turn import replay_turn

QUESTION = "how many customers do we have?"
SQL = "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL"


def cold_turn() -> ScriptedModel:
    """Two exploring calls, the SQL, extract, the answer."""
    return ScriptedModel(
        tool_result("list_tables", {}),
        text_result("customer has a deleted_at column"),
        json_result({"sql": SQL, "assumptions": ["soft deletes excluded"]}),
        no_entries(),
        text_result("1,840 customers."),
    )


async def replay(**kw) -> object:
    return await replay_turn(
        QUESTION,
        candidate=kw.pop("candidate", overrides.NONE),
        **kw,
    )


async def test_it_records_what_the_turn_did(monkeypatch, pool):
    monkeypatch.setattr(llm, "complete", cold_turn())

    out = await replay()

    assert out.error is None
    assert out.answer == "1,840 customers."
    assert out.sql == SQL
    assert out.rows == [{"n": 1840}]
    assert out.explored is True
    assert out.fix_attempts == 0
    # Five scripted calls at the harness's 100 in / 20 out.
    assert out.tokens == 5 * 120


async def test_the_tool_sequence_is_kept_in_order(monkeypatch, pool):
    """Order and repetition are the signal. "It called `sample_column` three
    times on one column" is a sentence about a tool description; "3 tool calls"
    is not."""
    monkeypatch.setattr(
        llm,
        "complete",
        ScriptedModel(
            tool_result("list_tables", {}),
            tool_result("describe_table", {"table": "customer"}),
            tool_result("sample_column", {"table": "customer", "column": "deleted_at"}),
            text_result("customer has a deleted_at column"),
            json_result({"sql": SQL, "assumptions": []}),
            no_entries(),
            text_result("1,840 customers."),
        ),
    )

    out = await replay()

    assert out.tool_names == ["list_tables", "describe_table", "sample_column"]
    assert out.tools[2].args == {"table": "customer", "column": "deleted_at"}
    assert [t.error for t in out.tools] == [False, False, False]


async def test_a_tool_error_is_recorded_rather_than_raised(monkeypatch, pool):
    """The model asked for a column that does not exist. That is the loop
    working, and it is also the kind of thing a better tool description
    prevents — so the record has to carry it."""
    monkeypatch.setattr(
        llm,
        "complete",
        ScriptedModel(
            tool_result("sample_column", {"table": "customer", "column": "nope"}),
            text_result("no such column"),
            json_result({"sql": SQL, "assumptions": []}),
            no_entries(),
            text_result("1,840 customers."),
        ),
    )

    out = await replay()

    assert out.error is None
    assert [t.error for t in out.tools] == [True]


async def test_fix_attempts_are_counted(monkeypatch, pool):
    """Two fixes is a different candidate from none, at the same final answer.
    The count is the difference."""
    monkeypatch.setattr(
        llm,
        "complete",
        ScriptedModel(
            tool_result("list_tables", {}),
            text_result("customer has a deleted_at column"),
            json_result({"sql": "SELECT count(*) FROM custmer", "assumptions": []}),
            json_result({"sql": SQL, "what_was_wrong": "table was misspelled"}),
            no_entries(),
            text_result("1,840 customers."),
        ),
    )

    out = await replay()

    assert out.fix_attempts == 1
    assert out.sql == SQL
    assert out.rows == [{"n": 1840}]
    assert out.sql_error == ""


async def test_a_candidate_that_breaks_the_turn_scores_rather_than_stops(
    monkeypatch, pool
):
    """A run that has already spent an hour must not end because one rollout
    fell over. The error is on the record; the metric turns it into zero."""

    async def explode(**kwargs):
        raise RuntimeError("the model refused")

    monkeypatch.setattr(llm, "complete", explode)

    out = await replay()

    assert out.error == "RuntimeError: the model refused"
    assert out.answer == ""


async def test_the_candidates_text_is_what_ran(monkeypatch, pool):
    """The whole point of the function. Asserted on what the model was sent,
    because a candidate that silently did not apply would still produce a
    plausible record."""
    model = cold_turn()
    monkeypatch.setattr(llm, "complete", model)

    await replay(
        candidate=overrides.Overrides(
            prompts={"explore": "Look around."},
            tools={"list_tables": "The tables, briefly."},
            efforts={"explore": "low"},
        )
    )

    first = next(c for c in model.calls if c.get("node") == "explore")
    assert first["system"].startswith("Look around.")
    assert first["effort"] == "low"
    assert {t["name"]: t["description"] for t in first["tools"]}["list_tables"] == (
        "The tables, briefly."
    )


async def test_it_neither_reads_the_memory_nor_adds_to_it(monkeypatch, pool, agent_conn):
    """The whole bargain. Tool descriptions only matter on the cold path — a
    warm turn never calls a tool — and a run that wiped the memory to get that
    would be a run nobody dares start."""
    await store.write_entries(
        agent_conn,
        [
            store.CacheEntry(
                kind="schema_fact",
                name="soft deletes",
                claim="customer.deleted_at marks removed rows",
                tables=["customer"],
            )
        ],
    )
    async with db.agent() as conn:
        assert await store.load_cache(conn)

    monkeypatch.setattr(llm, "complete", cold_turn())
    out = await replay()

    # It explored, so it did not read the entry that was sitting there.
    assert out.explored is True
    async with db.agent() as conn:
        after = await store.load_cache(conn)
    # And the entry is still there, untouched: the run leaves the memory as it
    # found it, which is what makes it safe to run against your own agent.
    assert [e.name for e in after] == ["soft deletes"]



async def test_it_opens_the_pools_it_needs_rather_than_assuming_them(monkeypatch):
    """No `pool` fixture, deliberately, and that is the entire point.

    Every other test in this file takes it, which is how `replay_turn` shipped
    needing an agent pool nobody opened for it. `make gepa-tools` scored
    nineteen rollouts zero in 1.6 seconds and the suite was green throughout: a
    turn writes to the turn log whatever its memory setting, and the only
    callers of `open_pools` were the server's startup and that fixture.

    So this one runs the way the optimiser runs, from a cold process.
    """
    await db.close_pools()
    monkeypatch.setattr(llm, "complete", cold_turn())

    out = await replay()

    assert out.error is None, "a cold process must be able to drive a turn"
    assert out.answer == "1,840 customers."
    await db.close_pools()


# ------------------------------------------------------------ where it went
#
# A config search needs to know which node spent what, and the answer event
# carries only the turn's totals. The per-node ledger is a second record of the
# same tokens, read off a different stream, so the two have to agree.


async def test_the_tokens_are_kept_per_node_and_sum_to_the_turn(monkeypatch, pool):
    """Five scripted calls: two in explore, one each in generate_sql, extract
    and answer. `plan` runs on a cold turn too, but a cold turn has an empty
    cache and the scripted model's first result is explore's — so the ledger
    names the nodes that returned tokens and nothing else."""
    monkeypatch.setattr(llm, "complete", cold_turn())

    out = await replay()

    assert sum(tin + tout for tin, tout in out.per_node.values()) == out.tokens
    assert out.per_node["explore"] == (200, 40), "two calls, 100 in / 20 out each"
    assert out.per_node["generate_sql"] == (100, 20)
    assert "execute" not in out.per_node, "no model call, no line"
    assert "load_cache" not in out.per_node


async def test_fix_accumulates_across_attempts(monkeypatch, pool):
    monkeypatch.setattr(
        llm,
        "complete",
        ScriptedModel(
            tool_result("list_tables", {}),
            text_result("customer has a deleted_at column"),
            json_result({"sql": "SELECT count(*) FROM custmer", "assumptions": []}),
            json_result({"sql": "SELECT count(*) FROM custome", "what_was_wrong": "typo"}),
            json_result({"sql": SQL, "what_was_wrong": "typo again"}),
            no_entries(),
            text_result("1,840 customers."),
        ),
    )

    out = await replay()

    assert out.fix_attempts == 2
    assert out.per_node["fix"] == (200, 40)


async def test_the_effort_each_node_ran_at_is_on_the_record(monkeypatch, pool):
    """Recorded with the candidate in force, because by the time the feedback
    is read the candidate is gone. The override lands on the node it names and
    the disk value stays on the rest."""
    from app.config import config

    monkeypatch.setattr(llm, "complete", cold_turn())

    out = await replay(candidate=overrides.Overrides(efforts={"explore": "low"}))

    assert out.efforts["explore"] == "low"
    assert out.efforts["generate_sql"] == config().effort_for("generate_sql")
    assert set(out.efforts) == {"plan", "explore", "generate_sql", "fix", "extract", "answer"}


async def test_a_rollout_opens_no_turn_span(monkeypatch, pool):
    """The harvests key on the `turn` span: `turn_scope` for extract, and
    `turn_cases` outright. A rollout that opened one would be the next
    round's training data. This was a comment in `replay_turn`; now it is
    checked."""
    from app import tracing

    def never(**kwargs):
        raise AssertionError(f"a rollout opened a turn span: {kwargs}")

    monkeypatch.setattr(tracing, "turn", never)
    monkeypatch.setattr(llm, "complete", cold_turn())

    out = await replay()

    assert out.error is None
