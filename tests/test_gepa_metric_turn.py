"""Scoring one whole turn against an answer we know.

The scalar decides which candidates survive a search. The prose decides what the
reflection model tries next, and it is the larger half of the file — GEPA points
a direction because a metric explained its number, and a metric that returns
0.62 with no sentence attached reduces the search to random mutation.

So these check both, and one of them checks something a comment cannot enforce:
that the reference query never reaches the feedback. A tool description carrying
the corpus's own queries would score beautifully here and nowhere else.

Pure. No database, no model.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tools.gepa import metric_turn
from tools.gepa.golden import GoldenCase
from tools.gepa.metric_turn import WITHHELD, Baseline, score
from tools.gepa.reference import Reference
from tools.gepa.replay_turn import ToolCall, TurnReplayed
from tools.gepa.resultset import rows_of

SQL = "SELECT count(*) AS customers FROM customer WHERE deleted_at IS NULL"
SECRET = "SELECT count(*) AS unmistakable_reference_query FROM customer"


def case(**kw) -> GoldenCase:
    return GoldenCase(
        **{
            "name": "customers_active",
            "question": "how many customers do we have?",
            "reference_sql": SECRET,
            "tables": ["customer"],
            "expect": [[1840]],
            "reading": "Customers that still exist.",
            **kw,
        }
    )


def reference(*rows) -> Reference:
    return Reference(
        case=case(), rows=rows_of(rows), at=datetime.now(timezone.utc)
    )


def turn(**kw) -> TurnReplayed:
    return TurnReplayed(
        **{
            "question": "how many customers do we have?",
            "answer": "1,840 customers.",
            "sql": SQL,
            "rows": [{"customers": 1840}],
            "tokens_in": 9000,
            "tokens_out": 500,
            **kw,
        }
    )


def tools(*calls) -> list[ToolCall]:
    return [ToolCall(name=n, args=a, error=e) for n, a, e in calls]


# ------------------------------------------------------------------ the happy path


def test_a_right_answer_with_nothing_to_compare_against_scores_one():
    """No baseline is the first pass, before the seed has been measured. Cost
    and tool calls have nothing to be one-sided against, so they cannot
    penalise — the same way `metric_extract._cost` handles a missing one."""
    result = score(turn(), case(), reference((1840,)))

    assert result.value == pytest.approx(1.0)
    assert result.terms == {"correct": 1.0, "cost": 1.0, "tool_calls": 1.0}
    assert result.feedback == []


def test_a_cheaper_turn_than_the_seed_earns_full_marks_and_no_more():
    """One-sided, deliberately. An unbounded reward for cheapness buys a
    candidate that skips the explore loop, and the explore loop is what finds
    the soft-delete column in the first place."""
    result = score(
        turn(tokens_in=1000, tokens_out=100),
        case(),
        reference((1840,)),
        baseline=Baseline(tokens=9500, tool_calls=4),
    )

    assert result.terms["cost"] == 1.0


def test_a_dearer_turn_is_told_what_it_spent_it_on():
    """The number alone says a candidate was worse. The sequence says where the
    tokens went, which is the thing a tool description can act on."""
    result = score(
        turn(
            tokens_in=18000,
            tokens_out=1000,
            tools=tools(
                ("list_tables", {}, False),
                ("describe_table", {"table": "customer"}, False),
            ),
        ),
        case(),
        reference((1840,)),
        baseline=Baseline(tokens=9500, tool_calls=4),
    )

    assert result.terms["cost"] == pytest.approx(0.0)
    assert "against the seed's 9500" in result.text()
    assert "describe_table(table='customer')" in result.text()


# ----------------------------------------------------------------- the gates


def test_a_rollout_that_fell_over_scores_zero_and_says_so():
    """GEPA's adapter contract: never raise per example, score the failure. The
    text distinguishes a broken harness from a bad candidate, because those
    want different responses from whoever reads the run."""
    result = score(
        turn(error="RuntimeError: the model refused"), case(), reference((1840,))
    )

    assert result.value == 0.0
    assert "RuntimeError: the model refused" in result.text()
    assert "fell over" in result.text()


def test_answering_without_querying_scores_zero():
    """The cheapest possible turn, and the one every cost term rewards. It is
    gated rather than weighted for exactly that reason."""
    result = score(
        turn(sql="", rows=[], answer="Roughly two thousand.", tools=tools(("list_tables", {}, False))),
        case(),
        reference((1840,)),
    )

    assert result.value == 0.0
    assert "without querying" in result.text()
    assert "'Roughly two thousand.'" in result.text()


def test_sql_that_never_ran_scores_zero_and_quotes_the_database():
    """TRAP 5's failure, as the metric sees it. The driver's own message is what
    tells a reflection model that `created_at` does not exist, and the fix count
    is what says how much that cost."""
    result = score(
        turn(
            sql="SELECT count(*) FROM orders WHERE created_at > now()",
            rows=[],
            sql_error='column "created_at" does not exist',
            fix_attempts=2,
            tools=tools(("describe_table", {"table": "orders"}, False)),
        ),
        case(),
        reference((1840,)),
    )

    assert result.value == 0.0
    assert 'column "created_at" does not exist' in result.text()
    assert "2 fix attempts" in result.text()
    assert "`describe_table` was called on `orders`" in result.text()


def test_an_empty_result_against_a_real_answer_scores_zero():
    """The filters removed everything. Distinct from SQL that errored: this one
    ran, so the feedback points at the column the filter names rather than at
    the syntax."""
    result = score(
        turn(
            sql="SELECT count(*) FROM customer WHERE region = 'West'",
            rows=[],
            tools=tools(("list_tables", {}, False)),
        ),
        case(),
        reference(("west", 500), ("east", 500)),
    )

    assert result.value == 0.0
    assert "no rows" in result.text()
    assert "`region`" in result.text(), "name the column nobody sampled"


# ------------------------------------------------------------- wrong answers


def test_a_wrong_answer_withholds_cost_and_tool_calls():
    """CHALLENGE item 5: a wrong answer scores zero regardless of cost. Cost and
    tool-call counts describe how a *right* answer was reached, so on a wrong
    one they have nothing to describe — and a cheap wrong turn must not be able
    to out-score an expensive right one."""
    result = score(
        turn(rows=[{"customers": 2000}], tokens_in=100, tokens_out=10),
        case(),
        reference((1840,)),
        baseline=Baseline(tokens=9500, tool_calls=4),
    )

    assert result.terms["correct"] == 0.0
    assert result.terms["cost"] == 0.0
    assert result.terms["tool_calls"] == 0.0
    assert result.value == 0.0
    assert WITHHELD in result.text()


def test_the_rows_that_differed_are_both_reported():
    """TRAP 2, as a reflection model reads it. Four rows where the answer has
    two, and three spellings of west with the count split between them — that
    is a sentence about grouping on an unnormalised column. A similarity of
    0.25 is not."""
    result = score(
        turn(
            rows=[
                {"region": "west", "n": 350},
                {"region": "West", "n": 100},
                {"region": "WEST", "n": 50},
                {"region": "east", "n": 500},
            ],
            sql="SELECT region, count(*) FROM customer GROUP BY region",
        ),
        case(expect=None),
        reference(("west", 500), ("east", 500)),
    )

    assert 0 < result.terms["correct"] < 1
    text = result.text()
    assert "did not come back" in text and "do not belong" in text
    assert "('west', 500)" in text
    assert "how this question is read here" in text


def test_right_rows_in_the_wrong_order_are_told_that_specifically():
    """A different mutation from wrong rows, so a different sentence. "Every
    value is right and the order is not" points at an ORDER BY; a list of
    differing rows would point at the WHERE clause."""
    result = score(
        turn(rows=[{"r": "east", "v": 2}, {"r": "west", "v": 9}]),
        case(expect=None, ordered=True),
        reference(("west", 9), ("east", 2)),
    )

    assert result.terms["correct"] == pytest.approx(0.5)
    assert "the order is not" in result.text()


def test_a_scalar_that_is_merely_wrong_says_so_plainly():
    result = score(turn(rows=[{"n": 2000}]), case(), reference((1840,)))

    assert "The number is wrong." in result.text()
    assert "2000" in result.text() and "1840" in result.text()


# ------------------------------------------------------ tool-call observations


def test_a_repeated_call_is_named_even_when_the_count_beat_the_seed():
    """A repeat is a fact about a description whether or not the total came in
    under budget: the second call asked a question the first had answered."""
    result = score(
        turn(
            tools=tools(
                ("list_tables", {}, False),
                ("sample_column", {"table": "customer", "column": "region"}, False),
                ("sample_column", {"table": "customer", "column": "region"}, False),
            )
        ),
        case(),
        reference((1840,)),
        baseline=Baseline(tokens=99999, tool_calls=99),
    )

    assert result.terms["tool_calls"] == 1.0, "under the seed's count"
    assert "more than once" in result.text()
    assert "x2" in result.text()


def test_an_errored_call_is_named_as_an_argument_nobody_explained():
    result = score(
        turn(tools=tools(("sample_column", {"table": "customer", "column": "nope"}, True))),
        case(),
        reference((1840,)),
    )

    assert "errored" in result.text()
    assert "did not explain" in result.text()


# ------------------------------------------------------------- the two invariants


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param({"error": "boom"}, id="harness broke"),
        pytest.param({"sql": "", "rows": []}, id="no sql"),
        pytest.param({"sql_error": "no such column", "rows": []}, id="never ran"),
        pytest.param({"rows": []}, id="empty result"),
        pytest.param({"rows": [{"n": 2000}]}, id="wrong answer"),
        pytest.param({}, id="correct"),
    ],
)
def test_every_term_is_present_on_every_path(broken):
    """`terms` becomes `EvaluationBatch.objective_scores`, which is what the
    Pareto front is drawn from. `metric_extract` returns an empty dict on its
    gates; doing that here would put holes in the front that nothing reports.
    """
    result = score(turn(**broken), case(), reference((1840,)))

    assert set(result.terms) == set(metric_turn.TERMS)
    assert all(isinstance(v, float) for v in result.terms.values())


@pytest.mark.parametrize(
    "broken",
    [
        {"error": "boom"},
        {"sql": "", "rows": []},
        {"sql_error": "no such column", "rows": []},
        {"rows": []},
        {"rows": [{"n": 2000}]},
        {},
    ],
)
def test_the_reference_query_never_reaches_the_feedback(broken):
    """The leak that would turn a tool-description search into memorisation.

    The reflection model reads this text and rewrites a component from it. A
    description carrying the corpus's own queries scores well on the corpus and
    nowhere else, and it would be found on stage. The rows a candidate got
    wrong are a diagnosis; the query that gets them right is the answer key.
    A comment cannot enforce that, so this does.
    """
    result = score(turn(**broken), case(), reference((1840,)))

    assert SECRET not in result.text()
    assert "unmistakable_reference_query" not in result.text()


def test_the_weights_can_be_varied_without_editing_the_file():
    """CHALLENGE item 5 asks for a sensitivity check: rerun selection with cost
    at 0.15 and at 0.35 and report whether the winner changes. If that needs an
    edit, nobody runs it twice."""
    heavy = {"correct": 0.60, "cost": 0.15, "tool_calls": 0.25}
    dear = turn(tokens_in=18000, tokens_out=1000)
    base = Baseline(tokens=9500, tool_calls=4)

    default = score(dear, case(), reference((1840,)), baseline=base)
    varied = score(dear, case(), reference((1840,)), baseline=base, weights=heavy)

    assert default.terms == varied.terms, "the measurement does not move"
    assert default.value != varied.value, "only what it is worth does"


# ------------------------------------------------------------ where it went
#
# The config target searches six efforts, and a reflection choosing between
# them needs to read which node spent what. A turn total cannot say that.


def spent(**per_node) -> dict[str, tuple[int, int]]:
    return {node: (tin, tout) for node, (tin, tout) in per_node.items()}


def test_a_dearer_turn_says_which_node_spent_it_and_at_what_effort():
    result = score(
        turn(
            tokens_in=18000,
            tokens_out=1000,
            per_node=spent(explore=(14000, 600), generate_sql=(3000, 300), answer=(1000, 100)),
            efforts={"explore": "high", "generate_sql": "high", "answer": "low"},
        ),
        case(),
        reference((1840,)),
        baseline=Baseline(tokens=9500, tool_calls=4),
    )

    text = result.text()
    assert "Where they went:" in text
    assert "explore (high)" in text and "14,600 tokens" in text
    # Dearest first, so the line to act on is the first one read.
    assert text.index("explore (high)") < text.index("generate_sql (high)")
    assert text.index("generate_sql (high)") < text.index("answer (low)")


def test_a_wrong_answer_names_the_effort_the_sql_was_written_at():
    result = score(
        turn(
            rows=[{"customers": 2000}],
            efforts={"generate_sql": "medium", "fix": "high"},
        ),
        case(),
        reference((1840,)),
    )

    assert "written by `generate_sql` at effort medium" in result.text()
    assert "corrected by `fix`" not in result.text(), "no fix ran"


def test_a_corrected_wrong_answer_names_both():
    result = score(
        turn(
            rows=[{"customers": 2000}],
            fix_attempts=1,
            efforts={"generate_sql": "low", "fix": "medium"},
        ),
        case(),
        reference((1840,)),
    )

    assert "at effort low and corrected by `fix` at effort medium" in result.text()


def test_a_record_without_efforts_says_nothing_about_them():
    """Records made before efforts were kept, and any harness that does not
    set them. A sentence with a blank in it is worse than no sentence."""
    result = score(turn(rows=[{"customers": 2000}]), case(), reference((1840,)))

    assert "at effort" not in result.text()
    assert metric_turn.by_node(turn()) == "(no per-node figures on this record)"


# ----------------------------------------------------------------- rejected


def test_a_rejected_candidate_scores_zero_on_every_term_with_the_reason():
    """Raised by a target's parser, caught by the adapter, scored here. Every
    term present and zero, so the front is not reshaped by a missing key, and
    the reason in the prose so the reflection stops proposing it."""
    result = metric_turn.rejected("plan: effort none — Claude has no such level")

    assert result.value == 0.0
    assert set(result.terms) == set(metric_turn.TERMS)
    assert all(v == 0.0 for v in result.terms.values())
    assert "rejected before any turn ran" in result.text()
    assert "Claude has no such level" in result.text()
    assert "fell over" not in result.text(), "not the harness's fault"
