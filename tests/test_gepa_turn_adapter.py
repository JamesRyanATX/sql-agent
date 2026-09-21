"""The whole-turn adapter, driven by a scripted model.

Same bargain as `tests/test_gepa_adapter.py`: the model is canned, so these
prove the *harness* rather than what a real model says. What is different here
is that a rollout is a whole turn, so this one needs the database — the
candidate's SQL runs, and so does the reference query it is scored against.

The two properties worth the most are both about not measuring the wrong thing.
A candidate must reach the graph as tool descriptions and nothing else, and a
rollout that falls over must score zero rather than ending a run that has
already spent an hour.

Requires `make up && make migrate && make seed`. No live model calls.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gepa", reason="uv run --group gepa")

from app import llm  # noqa: E402
from tools.gepa import golden, targets  # noqa: E402
from tools.gepa.adapter import Loop, TurnAdapter  # noqa: E402
from tests.test_coldpath import (  # noqa: E402,F401 — `pool` is a fixture
    ScriptedModel,
    json_result,
    no_entries,
    pool,
    text_result,
    tool_result,
)

SQL = "SELECT count(*) AS customers FROM customer WHERE deleted_at IS NULL"
WRONG = "SELECT count(*) AS customers FROM customer"

CASES = [c for c in golden.load() if c.name == "customers_active"]


@pytest.fixture
def loop():
    with Loop() as l:
        yield l


@pytest.fixture
def adapter(loop):
    return TurnAdapter(
        loop,
        to_overrides=targets._tools_overrides,
        focus=targets._tools_focus,
        concurrency=1,
    )


def cold(sql: str = SQL, answer: str = "1,840 customers.") -> ScriptedModel:
    """Two exploring calls, the SQL, extract, the answer."""
    return ScriptedModel(
        tool_result("list_tables", {}),
        text_result("customer has a deleted_at column"),
        json_result({"sql": sql, "assumptions": []}),
        no_entries(),
        text_result(answer),
    )


def test_a_right_answer_scores_and_carries_every_term(adapter, monkeypatch, pool):
    """The happy path, and the invariant the Pareto front depends on: the term
    keys are the same on every case, so an objective front has no holes."""
    monkeypatch.setattr(llm, "complete", cold())

    batch = adapter.evaluate(CASES, targets.TOOLS.seed())

    assert batch.scores == [1.0]
    assert batch.objective_scores[0].keys() == {"correct", "cost", "tool_calls"}
    assert batch.outputs[0]["sql"] == SQL
    assert batch.outputs[0]["tools"] == ["list_tables"]


def test_a_wrong_answer_scores_zero_without_ending_the_batch(
    adapter, monkeypatch, pool
):
    """2,000 rather than 1,840 — TRAP 1, missed. It is a bad candidate, not a
    broken run, and the term keys are still all present."""
    monkeypatch.setattr(llm, "complete", cold(sql=WRONG, answer="2,000 customers."))

    batch = adapter.evaluate(CASES, targets.TOOLS.seed())

    assert batch.scores == [0.0]
    assert batch.objective_scores[0] == {"correct": 0.0, "cost": 0.0, "tool_calls": 0.0}


def test_a_rollout_that_raises_scores_zero_rather_than_ending_the_run(
    adapter, monkeypatch, pool
):
    """GEPA's adapter contract. A run that has already spent an hour must not
    end because one rollout fell over."""

    async def explode(**kwargs):
        raise RuntimeError("the model refused")

    monkeypatch.setattr(llm, "complete", explode)

    batch = adapter.evaluate(CASES, targets.TOOLS.seed())

    assert batch.scores == [0.0]
    assert set(batch.objective_scores[0]) == {"correct", "cost", "tool_calls"}


def test_the_candidates_descriptions_are_what_the_model_was_sent(
    adapter, monkeypatch, pool
):
    """The whole point of the adapter. A candidate that silently did not apply
    would still produce a plausible score, and a whole search would be measuring
    the seed against itself."""
    model = cold()
    monkeypatch.setattr(llm, "complete", model)

    adapter.evaluate(CASES, {**targets.TOOLS.seed(), "list_tables": "The tables."})

    explore = next(c for c in model.calls if c.get("node") == "explore")
    sent = {t["name"]: t["description"] for t in explore["tools"]}
    assert sent["list_tables"] == "The tables."
    # The three nobody mutated are still the live text.
    assert sent["describe_table"] == targets.TOOLS.seed()["describe_table"]


def test_nothing_but_tool_descriptions_reaches_the_graph(adapter, monkeypatch, pool):
    """A search over descriptions that also moved a prompt would be measuring
    something nobody asked about, and would still look like it worked."""
    model = cold()
    monkeypatch.setattr(llm, "complete", model)
    from app import prompts

    adapter.evaluate(CASES, targets.TOOLS.seed())

    explore = next(c for c in model.calls if c.get("node") == "explore")
    assert explore["system"].startswith(prompts.get("explore"))


def test_the_first_turn_to_answer_a_case_sets_its_baseline(
    adapter, monkeypatch, pool
):
    """Cost and tool calls are one-sided against the seed's own spend. First
    seen rather than best seen, so what a candidate is measured against does not
    move underneath the run."""
    monkeypatch.setattr(llm, "complete", cold())
    adapter.evaluate(CASES, targets.TOOLS.seed())

    baseline = adapter._baseline["customers_active"]
    assert baseline.tokens > 0
    assert baseline.tool_calls == 1

    # A second, dearer pass is scored against the first and does not replace it.
    monkeypatch.setattr(
        llm,
        "complete",
        ScriptedModel(
            tool_result("list_tables", {}),
            tool_result("describe_table", {"table": "customer"}),
            text_result("customer has a deleted_at column"),
            json_result({"sql": SQL, "assumptions": []}),
            no_entries(),
            text_result("1,840 customers."),
        ),
    )
    batch = adapter.evaluate(CASES, targets.TOOLS.seed())

    assert adapter._baseline["customers_active"] == baseline
    assert batch.objective_scores[0]["tool_calls"] < 1.0


# ------------------------------------------------------ the reflective dataset


def test_the_reflective_dataset_foregrounds_the_component_being_mutated(
    adapter, monkeypatch, pool
):
    """"It called `sample_column` three times on one column" is a sentence about
    `sample_column`'s description. The undifferentiated turn summary is not, and
    a reflection that cannot see the difference proposes generic prose."""
    monkeypatch.setattr(
        llm,
        "complete",
        ScriptedModel(
            tool_result("sample_column", {"table": "customer", "column": "region"}),
            tool_result("sample_column", {"table": "customer", "column": "region"}),
            text_result("customer has a deleted_at column"),
            json_result({"sql": SQL, "assumptions": []}),
            no_entries(),
            text_result("1,840 customers."),
        ),
    )

    batch = adapter.evaluate(CASES, targets.TOOLS.seed(), capture_traces=True)
    records = adapter.make_reflective_dataset(
        targets.TOOLS.seed(), batch, ["sample_column"]
    )

    assert set(records) == {"sample_column"}
    record = records["sample_column"][0]
    assert "called 2 time(s)" in record["How `sample_column` was used"]
    assert record["Inputs"]["the question"] == CASES[0].question


def test_a_tool_that_was_never_called_is_said_to_have_not_been(
    adapter, monkeypatch, pool
):
    """As informative as a redundant call, and dropping the case would hide half
    the signal a description search runs on."""
    monkeypatch.setattr(llm, "complete", cold())

    batch = adapter.evaluate(CASES, targets.TOOLS.seed(), capture_traces=True)
    records = adapter.make_reflective_dataset(
        targets.TOOLS.seed(), batch, ["count_distinct"]
    )

    assert "never called" in records["count_distinct"][0]["How `count_distinct` was used"]


def test_only_the_components_gepa_asked_about_come_back(adapter, monkeypatch, pool):
    """Round-robin mutates one per iteration. Returning all four would spend the
    reflection model's context on three components nobody is changing."""
    monkeypatch.setattr(llm, "complete", cold())

    batch = adapter.evaluate(CASES, targets.TOOLS.seed(), capture_traces=True)

    assert set(
        adapter.make_reflective_dataset(targets.TOOLS.seed(), batch, ["list_tables"])
    ) == {"list_tables"}


# --------------------------------------------------------- the cheap pre-check


def test_the_seed_check_reports_every_case_and_what_it_cost(
    monkeypatch, pool, capsys
):
    """`make gepa-tools GEPA_ARGS=--probe-only`, which is two things at once.

    It is the before half of the talk's before-and-after table, and it is the
    cheap way to find out whether the corpus is answerable at all — if the seed
    gets most of it wrong, two of three terms are gated off on every case and a
    search has nothing to optimise but a term it cannot move.

    One case here rather than nineteen: what is being checked is the reporting,
    and a scripted model is a queue that nineteen concurrent turns would drain
    in an order nobody can predict.
    """
    from tools.gepa import cli

    monkeypatch.setattr(golden, "load", lambda: CASES)
    monkeypatch.setattr(cli, "ask_before_spending", lambda *a, **k: None)
    monkeypatch.setattr(llm, "complete", cold())

    with Loop() as loop:
        status = targets._tools_check(loop)

    out = capsys.readouterr().err
    assert status == 0
    assert "customers_active" in out
    assert "the seed answers 1 of 1 correctly" in out


def test_a_corpus_the_seed_mostly_fails_is_reported_as_the_problem_it_is(
    monkeypatch, pool, capsys
):
    """Not a search that found nothing. The fix is easier cases, and saying so
    here costs one run instead of one run plus a search."""
    from tools.gepa import cli

    monkeypatch.setattr(golden, "load", lambda: CASES)
    monkeypatch.setattr(cli, "ask_before_spending", lambda *a, **k: None)
    monkeypatch.setattr(llm, "complete", cold(sql=WRONG, answer="2,000 customers."))

    with Loop() as loop:
        status = targets._tools_check(loop)

    assert status == 1
    assert "Under half" in capsys.readouterr().err
