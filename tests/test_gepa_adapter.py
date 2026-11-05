"""The GEPA wiring, driven by a scripted model.

Same bargain as `tests/test_coldpath.py`: the model is canned, so these prove
the *harness* — the sync/async bridge, the adapter contract, the reflective
dataset, the probe gate, and that `gepa.optimize` actually turns over — without
spending tokens or depending on what a real model happens to say. What a real
model does with a candidate is `make gepa-extract`, which is a separate and
deliberately manual measurement.

No database either: `extract` is a function of three strings.
"""

from __future__ import annotations

import json

import pytest

# `gepa` is a dependency group, so `make test` (which syncs dev only, like the
# image) does not have it. Skipping is right rather than adding it to dev: the
# point of the group is that the optimiser is not part of what ships or of what
# every contributor installs.
pytest.importorskip("gepa", reason="uv run --group gepa")

from app import llm  # noqa: E402
from tools.gepa.adapter import COMPONENT, ExtractAdapter, Loop  # noqa: E402
from tools.gepa.cases import ExtractCase  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SQL = (
    "SELECT sum(oi.qty * oi.price) AS revenue FROM order_item oi "
    "JOIN orders o ON o.id = oi.order_id WHERE o.status <> 'cancelled'"
)

GOOD_OUTPUT = {
    "entries": [
        {
            "kind": "recipe",
            "name": "revenue",
            "claim": "revenue is quantity times price on order lines, counting "
                     "only orders whose status is not cancelled",
            "sql_fragment": "sum(oi.qty * oi.price) FROM order_item oi JOIN orders o "
                            "ON o.id = oi.order_id WHERE o.status <> 'cancelled'",
            "tables": ["order_item", "orders"],
        },
        {
            "kind": "schema_fact",
            "name": "orders.status",
            "claim": "orders.status takes the values pending, paid, cancelled and refunded",
            "tables": ["orders"],
        },
    ]
}

CENSUS_OUTPUT = {
    "entries": [
        {
            "kind": "schema_fact",
            "name": "soft deletes",
            "claim": "160 of 2,000 customer rows are soft-deleted",
            "tables": ["customer"],
        }
    ]
}


def cases(n: int = 4) -> list[ExtractCase]:
    return [
        ExtractCase.authored(
            name=f"case-{i}",
            question=f"question {i}?",
            sql=SQL,
            findings="orders joins order_item on order_id",
        )
        for i in range(n)
    ]


@pytest.fixture
def scripted(monkeypatch):
    """Replace the one seam that talks to a model."""
    state = {"payload": GOOD_OUTPUT, "calls": [], "reflections": 0}

    async def fake_complete(**kwargs):
        state["calls"].append(kwargs)
        if kwargs.get("node") == "gepa.reflect":
            state["reflections"] += 1
            return llm.Result(
                text=f"an improved instruction, revision {state['reflections']}",
                stop_reason="end_turn", tokens_in=500, tokens_out=200,
            )
        return llm.Result(
            text=json.dumps(state["payload"]),
            stop_reason="end_turn", tokens_in=400, tokens_out=120,
        )

    monkeypatch.setattr(llm, "complete", fake_complete)
    # `replay` and `adapter` import the module, not the function, so patching
    # the attribute reaches both.
    return state


@pytest.fixture
def loop():
    with Loop() as l:
        yield l


# ------------------------------------------------------------------ the bridge


def test_the_bridge_runs_async_work_from_synchronous_code(loop, scripted):
    """GEPA's engine is synchronous and `llm.complete` is not.

    Deliberately not `asyncio.run()` per call: the Anthropic and httpx clients
    are module singletons that bind to the loop that first used them, and
    tearing a loop down around them fails intermittently — the worst thing to
    hit two hundred rollouts into a run.
    """
    adapter = ExtractAdapter(loop)
    first = adapter.evaluate(cases(2), {COMPONENT: "a prompt"})
    second = adapter.evaluate(cases(2), {COMPONENT: "another prompt"})

    assert len(first.scores) == 2
    assert len(second.scores) == 2, "the same loop has to survive a second batch"


# ----------------------------------------------------------- the adapter contract


def test_evaluate_returns_one_score_per_case(loop, scripted):
    batch = cases(3)
    result = ExtractAdapter(loop).evaluate(batch, {COMPONENT: "a prompt"})

    assert len(result.outputs) == len(result.scores) == len(batch)
    assert all(0.0 <= s <= 1.0 for s in result.scores)
    assert result.trajectories is None, "not requested"


def test_capture_traces_returns_a_trajectory_per_case(loop, scripted):
    batch = cases(3)
    result = ExtractAdapter(loop).evaluate(
        batch, {COMPONENT: "a prompt"}, capture_traces=True
    )

    assert len(result.trajectories) == len(batch)
    assert all(t.score.value > 0.5 for t in result.trajectories)


def test_the_candidate_is_what_reaches_the_model(loop, scripted):
    ExtractAdapter(loop).evaluate(cases(1), {COMPONENT: "THE CANDIDATE TEXT"})
    assert scripted["calls"][0]["system"] == "THE CANDIDATE TEXT"


def test_the_harness_never_writes_under_the_node_it_harvests(loop, scripted):
    """Otherwise round two of an optimisation trains on round one's output."""
    ExtractAdapter(loop).evaluate(cases(1), {COMPONENT: "a prompt"})
    assert scripted["calls"][0]["node"] == "extract.replay"


def test_only_the_one_score_per_case_reaches_gepa(loop, scripted):
    """The extract score is one number per case. GEPA selects parents from
    candidates against validation examples, and per-term scores would only
    give it a second table that nothing here reads."""
    result = ExtractAdapter(loop).evaluate(cases(2), {COMPONENT: "a prompt"})
    assert result.objective_scores is None


# ------------------------------------------------------------ the size limit


def test_a_prompt_over_the_limit_scores_zero_on_every_case_and_calls_nothing(
    loop, scripted
):
    adapter = ExtractAdapter(loop, max_chars=10)
    result = adapter.evaluate(
        cases(3), {COMPONENT: "a prompt that is longer than ten"}, capture_traces=True
    )

    assert result.scores == [0.0, 0.0, 0.0]
    assert scripted["calls"] == [], "no model call for a prompt over the limit"
    assert adapter.calls == 0, "nothing was called, so nothing is counted"
    for t in result.trajectories:
        assert "the limit is 10" in t.score.text()


def test_a_prompt_over_the_limit_reports_zero_metric_calls(loop, scripted):
    """GEPA charges `len(batch)` against the budget unless the batch says how
    many calls it made. None were made, so the batch says 0."""
    adapter = ExtractAdapter(loop, max_chars=10)
    result = adapter.evaluate(cases(3), {COMPONENT: "a prompt that is longer than ten"})

    assert result.num_metric_calls == 0


def test_a_run_batch_leaves_the_metric_call_count_to_gepa(loop, scripted):
    result = ExtractAdapter(loop, max_chars=1_000).evaluate(cases(2), {COMPONENT: "p"})
    assert result.num_metric_calls is None


def test_every_record_from_a_parents_batch_states_the_limit(loop, scripted):
    """GEPA reflects on the parent's batch and never on a rejected child's,
    so the reason a child was rejected is never read. The limit is stated on
    the parent's records instead, before the next prompt is written."""
    adapter = ExtractAdapter(loop, max_chars=3_114)
    parent = "p" * 2_076
    scripted["payload"] = GOOD_OUTPUT
    good = adapter.evaluate(cases(1), {COMPONENT: parent}, capture_traces=True)
    scripted["payload"] = CENSUS_OUTPUT
    bad = adapter.evaluate(cases(1), {COMPONENT: parent}, capture_traces=True)
    merged = type(good)(
        outputs=good.outputs + bad.outputs,
        scores=good.scores + bad.scores,
        trajectories=good.trajectories + bad.trajectories,
    )

    records = adapter.make_reflective_dataset({COMPONENT: parent}, merged, [COMPONENT])[
        COMPONENT
    ]

    sentence = (
        "The prompt is 2,076 characters. A prompt longer than 3,114 characters "
        "is not run and scores 0 on every case."
    )
    assert len(records) == 2
    for record in records:
        assert record["Feedback"].endswith(sentence)


def test_no_limit_sentence_when_there_is_no_limit(loop, scripted):
    adapter = ExtractAdapter(loop)
    batch = adapter.evaluate(cases(1), {COMPONENT: "p"}, capture_traces=True)
    record = adapter.make_reflective_dataset({COMPONENT: "p"}, batch, [COMPONENT])[
        COMPONENT
    ][0]

    assert record["Feedback"] == "No problems found."
    assert "characters" not in record["Feedback"]


def test_a_prompt_at_the_limit_is_run(loop, scripted):
    adapter = ExtractAdapter(loop, max_chars=len("a prompt"))
    result = adapter.evaluate(cases(2), {COMPONENT: "a prompt"})

    assert len(scripted["calls"]) == 2
    assert adapter.calls == 2
    assert all(s > 0 for s in result.scores)


def test_a_model_failure_scores_zero_rather_than_ending_the_run(loop, monkeypatch):
    async def explode(**kwargs):
        raise RuntimeError("the provider is having a day")

    monkeypatch.setattr(llm, "complete", explode)
    result = ExtractAdapter(loop).evaluate(cases(2), {COMPONENT: "a prompt"})

    assert result.scores == [0.0, 0.0]


# --------------------------------------------------------- the reflective dataset


def test_the_reflective_dataset_leads_with_the_worst_case(loop, scripted):
    adapter = ExtractAdapter(loop)
    batch = cases(2)
    scripted["payload"] = GOOD_OUTPUT
    good = adapter.evaluate(batch[:1], {COMPONENT: "p"}, capture_traces=True)
    scripted["payload"] = CENSUS_OUTPUT
    bad = adapter.evaluate(batch[1:], {COMPONENT: "p"}, capture_traces=True)

    merged = type(good)(
        outputs=good.outputs + bad.outputs,
        scores=good.scores + bad.scores,
        trajectories=good.trajectories + bad.trajectories,
    )
    records = adapter.make_reflective_dataset({COMPONENT: "p"}, merged, [COMPONENT])[
        COMPONENT
    ]

    assert records[0]["score"] < records[1]["score"], "worst first"


def test_the_feedback_is_prose_a_mutation_can_aim_at(loop, scripted):
    """GEPA's whole contribution over a scalar reward is that the reflection
    step reads diagnostics. '0.62' gives it nothing to work with."""
    scripted["payload"] = CENSUS_OUTPUT
    adapter = ExtractAdapter(loop)
    batch = adapter.evaluate(cases(1), {COMPONENT: "p"}, capture_traces=True)
    record = adapter.make_reflective_dataset({COMPONENT: "p"}, batch, [COMPONENT])[
        COMPONENT
    ][0]

    assert "soft deletes" in record["Feedback"]
    assert "160 of 2,000" in record["Feedback"]
    assert "write the rule, not the number" in record["Feedback"]
    assert record["Inputs"]["the SQL that had already run"] == SQL
    # JSON-serialisable: it is passed verbatim into the proposal prompt.
    json.dumps(record)


def test_a_component_nobody_asked_about_produces_nothing(loop, scripted):
    adapter = ExtractAdapter(loop)
    batch = adapter.evaluate(cases(1), {COMPONENT: "p"}, capture_traces=True)
    assert adapter.make_reflective_dataset({COMPONENT: "p"}, batch, ["plan"]) == {}


# ------------------------------------------------------------------ end to end


def test_gepa_turns_over_against_the_adapter(loop, scripted):
    """The one test that proves the whole loop composes.

    A tiny budget — the point is that GEPA accepts our adapter, calls evaluate,
    builds a reflective dataset, asks the reflection model for a new candidate
    and returns a pool. Whether the score improves is not assertable against a
    scripted model that returns the same thing regardless of the prompt.

    The scripted output has to be *imperfect* for this to prove anything.
    `skip_perfect_score` is on by default, so a seed already scoring 1.0 makes
    GEPA correctly decline to mutate at all — which is a real outcome on a
    well-tuned prompt, and the reason `optimize` reports it as a result rather
    than letting an empty pool read as a broken run.
    """
    import gepa

    from tools.gepa.adapter import reflection_lm

    scripted["payload"] = CENSUS_OUTPUT
    train, val = cases(4), cases(2)
    result = gepa.optimize(
        seed_candidate={COMPONENT: "the seed instruction"},
        trainset=train,
        valset=val,
        adapter=ExtractAdapter(loop),
        reflection_lm=reflection_lm(loop),
        max_metric_calls=20,
        display_progress_bar=False,
    )

    assert result.candidates, "no pool came back"
    assert all(COMPONENT in c for c in result.candidates)
    assert result.total_metric_calls > 0
    assert scripted["reflections"] > 0, "the reflection model was never asked"


def test_the_reflection_record_carries_one_score_and_no_breakdown(loop, scripted):
    """The extract score has no parts, so a breakdown would be an empty dict
    in front of the reflection model."""
    adapter = ExtractAdapter(loop)
    batch = adapter.evaluate(cases(1), {COMPONENT: "p"}, capture_traces=True)
    record = adapter.make_reflective_dataset({COMPONENT: "p"}, batch, [COMPONENT])[
        COMPONENT
    ][0]

    assert record["score"] == 1.0
    assert "score_breakdown" not in record
