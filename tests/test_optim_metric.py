"""The metric, and the ways an optimiser would try to cheat it.

Every test here is a degenerate prompt someone reasoned their way to before the
weights existed. If one of them starts passing, the metric has a hole and the
next GEPA run will find it — that is the whole reason they are written as tests
rather than as a comment about gaming.

Free: no model, no database. The metric is a pure function of replayed output.
"""

from __future__ import annotations

import pytest

from app import store
from app.graph import grounded_in
from optim import metric_extract as metric
from optim.cases import ExtractCase
from optim.replay import Replayed

SQL = (
    "SELECT sum(oi.qty * oi.price) AS revenue FROM order_item oi "
    "JOIN orders o ON o.id = oi.order_id "
    "WHERE o.status <> 'cancelled'"
)

CASE = ExtractCase.authored(
    name="t",
    question="what was revenue last quarter?",
    sql=SQL,
    findings="orders.status takes pending, paid, cancelled and refunded",
    filed={"revenue": "revenue is quantity times price on orders that were not cancelled"},
)
CASE = ExtractCase(**{**CASE.__dict__, "baseline_tokens_out": 300})


def recipe(name: str, claim: str, fragment: str) -> store.CacheEntry:
    return store.CacheEntry(
        kind="recipe",
        name=name,
        claim=claim,
        sql_fragment=fragment,
        verified=grounded_in(fragment, SQL),
    )


def fact(name: str, claim: str) -> store.CacheEntry:
    return store.CacheEntry(kind="schema_fact", name=name, claim=claim)


def scored(*entries: store.CacheEntry, tokens_out: int = 300, error: str | None = None):
    return metric.score(
        Replayed(case=CASE, entries=list(entries), tokens_out=tokens_out, error=error)
    )


GOOD = (
    recipe(
        "revenue",
        "revenue is quantity times price on order lines, counting only orders "
        "whose status is not cancelled",
        "sum(oi.qty * oi.price) FROM order_item oi JOIN orders o ON o.id = oi.order_id "
        "WHERE o.status <> 'cancelled'",
    ),
    fact("orders.status", "orders.status takes the values pending, paid, cancelled and refunded"),
    fact("order_item", "order_item joins to orders on order_id and holds one row per line"),
)


def test_a_good_extraction_scores_near_the_top():
    s = scored(*GOOD)
    assert s.value > 0.9, (s.value, s.terms, s.text())
    assert s.text() == "No problems found."


# --------------------------------------------------- the ways to cheat it


def test_emitting_nothing_does_not_win():
    """The cheapest possible output, and the first thing a cost term rewards.

    This caught a real hole. Scored term-by-term, an empty extraction gets
    census 1.0, names 1.0 and cost 1.0 — three of five terms are vacuously
    perfect when there is nothing to be wrong about — and the total came to
    0.6. Emptiness is now a gate rather than a good score on four terms.
    """
    s = scored(tokens_out=5)
    assert s.value == 0.0, s.terms
    assert "declining to do its one job" in s.text()


def test_a_fragment_too_short_to_be_wrong_does_not_win():
    """`grounded_in` accepts `count(*)` against any query that counts. If this
    starts passing, the verified flag has stopped meaning anything and
    PLAN_SYSTEM is granting authority to guesses."""
    cheap = recipe("revenue", "the revenue of the business", "FROM order_item")
    assert cheap.verified, "production really does accept this — hence the clamp"

    s = scored(cheap, *GOOD[1:])
    assert s.terms["grounding"] < 1.0
    assert "too short to be wrong" in s.text()


def test_the_clamp_catches_trivial_fragments_and_not_thin_ones():
    """The limit of what a token count can claim, stated rather than implied.

    `sum(oi.qty * oi.price)` names real columns, so it clears the clamp — even
    though it drops `WHERE o.status <> 'cancelled'`, which is the whole trap.
    No token-count heuristic distinguishes "carries the load-bearing predicate"
    from "does not"; that is what the grounding probe and a human reading the
    winning prompt are for. Pretending otherwise would be a clamp that agrees
    with itself.
    """
    thin = recipe("revenue", "quantity times price", "sum(oi.qty * oi.price)")
    assert scored(thin, *GOOD[1:]).terms["grounding"] == 1.0


def test_a_recipe_that_claims_what_the_query_did_not_is_penalised():
    """The 16% revenue gap in grounded_in's docstring, as a score."""
    s = scored(
        recipe(
            "revenue",
            "revenue excludes cancelled and pending orders",
            "WHERE o.status NOT IN ('cancelled', 'pending')",
        ),
        *GOOD[1:],
    )
    assert s.terms["grounding"] == 0.0
    assert "never did" in s.text()


def test_a_census_is_penalised_in_proportion():
    s = scored(GOOD[0], fact("soft deletes", "160 of 2,000 customer rows are soft-deleted"))
    assert s.terms["census"] == pytest.approx(0.5)
    assert "goes stale the instant a row changes" in s.text()


def test_overwriting_a_general_rule_with_a_thinner_claim_is_penalised():
    s = scored(recipe("revenue", "west region revenue", "sum(oi.qty * oi.price)"), *GOOD[1:])
    assert s.terms["names"] < 1.0
    assert "much shorter claim" in s.text()


def test_a_paraphrase_of_a_filed_name_is_penalised():
    s = scored(fact("revenues", "much the same thing"), *GOOD)
    assert s.terms["names"] < 1.0
    assert "paraphrase" in s.text()


def test_the_same_name_twice_in_one_batch_is_penalised():
    s = scored(fact("dup", "one claim"), fact("dup", "another claim"), GOOD[0])
    assert s.terms["names"] < 1.0
    assert "emitted twice" in s.text()


def test_one_giant_entry_does_not_beat_several_good_ones():
    """Games the count band and the cost term at once."""
    giant = scored(recipe("everything", "x " * 200, GOOD[0].sql_fragment))
    assert giant.value < scored(*GOOD).value


def test_flooding_the_cache_is_penalised():
    """The cache is sent in full every turn, so entries are a recurring bill."""
    many = scored(*GOOD, *[fact(f"n{i}", f"claim {i}") for i in range(12)])
    assert many.terms["shape"] < 1.0
    assert many.value < scored(*GOOD).value


# ------------------------------------------------------------------- the cost


def test_cost_is_one_sided():
    """Cheaper than baseline earns 1.0 and no more. An unbounded reward for
    terseness is the same failure mode as 'fewer entries is better'."""
    assert scored(*GOOD, tokens_out=10).terms["cost"] == 1.0
    assert scored(*GOOD, tokens_out=300).terms["cost"] == 1.0
    assert scored(*GOOD, tokens_out=450).terms["cost"] == pytest.approx(0.5)
    assert scored(*GOOD, tokens_out=9_000).terms["cost"] == 0.0


def test_a_case_with_no_baseline_is_not_scored_on_cost():
    """An authored probe never ran in production, so there is nothing to
    compare against and inventing a target would be scoring noise."""
    no_baseline = Replayed(case=CASE.__class__.authored(
        name="p", question="q", sql=SQL, findings="f"
    ), entries=list(GOOD), tokens_out=99_999)
    assert metric.score(no_baseline).terms["cost"] == 1.0


# ------------------------------------------------------------------- plumbing


def test_a_failed_call_scores_zero_rather_than_raising():
    s = scored(error="APIError: 500")
    assert s.value == 0.0
    assert "did not produce usable output" in s.text()


def test_the_weights_are_a_partition_of_one():
    """Not arithmetic hygiene — it is what makes `value` readable as a
    proportion, so a reader can argue with 0.35 rather than decode it."""
    assert sum(metric.WEIGHTS.values()) == pytest.approx(1.0)


def test_the_feedback_names_the_entry_and_quotes_the_text():
    """GEPA's reflection step proposes a targeted mutation from this prose. A
    bare number gives it nothing to aim at."""
    s = scored(fact("soft deletes", "160 of 2,000 rows are soft-deleted"), GOOD[0])
    text = s.text()
    assert "soft deletes" in text
    assert "160 of 2,000" in text
    assert "write the rule, not the number" in text
