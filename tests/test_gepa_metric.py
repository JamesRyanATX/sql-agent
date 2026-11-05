"""The metric, and the ways an optimiser would try to cheat it.

Every test here is a degenerate prompt someone reasoned their way to before the
metric existed. If one of them starts passing, the metric has a hole and the
next GEPA run will find it — that is the whole reason they are written as tests
rather than as a comment about gaming.

Free: no model, no database. The metric is a pure function of replayed output.
"""

from __future__ import annotations

from app import store
from app.graph import grounded_in
from tools.gepa import metric_extract as metric
from tools.gepa.cases import ExtractCase
from tools.gepa.replay import Replayed

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
CASE = ExtractCase(
    **{**CASE.__dict__, "baseline_tokens_out": 300, "baseline_tokens_in": 700}
)


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


def scored(
    *entries: store.CacheEntry,
    tokens_out: int = 300,
    tokens_in: int = 700,
    error: str | None = None,
):
    return metric.score(
        Replayed(
            case=CASE,
            entries=list(entries),
            tokens_out=tokens_out,
            tokens_in=tokens_in,
            error=error,
        )
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


def test_a_good_extraction_scores_one_with_no_feedback():
    s = scored(*GOOD)
    assert s.value == 1.0, s.text()
    assert s.text() == "No problems found."


def test_the_extract_score_has_no_parts():
    """The number is one rule, not a weighted sum, so there is nothing to put
    in `terms`. Only `value` reaches GEPA."""
    assert scored(*GOOD).terms == {}


# --------------------------------------------------- the ways to cheat it


def test_emitting_nothing_does_not_win():
    """The cheapest possible output.

    This caught a real hole when the score was weighted terms: an empty
    extraction got census 1.0, names 1.0 and cost 1.0, because there was
    nothing to be wrong about, and the total came to 0.6. Emptiness is a gate.
    """
    s = scored(tokens_out=5)
    assert s.value == 0.0
    assert "declining to do its one job" in s.text()


def test_a_fragment_too_short_to_be_wrong_does_not_win():
    """`grounded_in` accepts `count(*)` against any query that counts. If this
    starts passing, the verified flag has stopped meaning anything and
    PLAN_SYSTEM is granting authority to guesses."""
    cheap = recipe("revenue", "the revenue of the business", "FROM order_item")
    assert cheap.verified, "production really does accept this — hence the clamp"

    s = scored(cheap, *GOOD[1:])
    assert s.value == 0.0
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
    thin = recipe("line total", "quantity times price", "sum(oi.qty * oi.price)")
    assert scored(thin, *GOOD[1:]).value == 1.0


def test_a_recipe_that_claims_what_the_query_did_not_scores_zero():
    """The 16% revenue gap in grounded_in's docstring, as a score."""
    s = scored(
        recipe(
            "revenue",
            "revenue excludes cancelled and pending orders",
            "WHERE o.status NOT IN ('cancelled', 'pending')",
        ),
        *GOOD[1:],
    )
    assert s.value == 0.0
    assert "never did" in s.text()


def test_a_clean_case_scores_the_fraction_of_its_recipes_that_are_grounded():
    """No census and no naming problem, so the number is grounding alone: one
    of two recipes quotes the SQL that ran."""
    s = scored(
        GOOD[0],
        recipe(
            "pending orders",
            "pending orders are excluded alongside cancelled ones",
            "WHERE o.status NOT IN ('cancelled', 'pending')",
        ),
        *GOOD[1:],
    )
    assert s.value == 0.5
    assert "1 of 2 recipes are not properly grounded" in s.text()


# ---------------------------------------------------------------- the gates


def test_a_census_note_scores_zero():
    """One count among good notes is enough. A count is a failure that shows
    up turns later, and a share of the score would let a candidate trade it
    for grounding elsewhere."""
    s = scored(GOOD[0], fact("soft deletes", "160 of 2,000 customer rows are soft-deleted"))
    assert s.value == 0.0
    assert "goes stale the instant a row changes" in s.text()


def test_overwriting_a_general_rule_with_a_thinner_claim_scores_zero():
    s = scored(recipe("revenue", "west region revenue", "sum(oi.qty * oi.price)"), *GOOD[1:])
    assert s.value == 0.0
    assert "much shorter claim" in s.text()


def test_a_paraphrase_of_a_filed_name_scores_zero():
    s = scored(fact("revenues", "much the same thing"), *GOOD)
    assert s.value == 0.0
    assert "paraphrase" in s.text()


def test_the_same_name_twice_in_one_batch_scores_zero():
    s = scored(fact("dup", "one claim"), fact("dup", "another claim"), GOOD[0])
    assert s.value == 0.0
    assert "emitted twice" in s.text()


def test_a_case_scored_zero_by_a_gate_still_carries_every_other_note():
    """The reflection model reads the feedback, not the number. A case the
    census gate scored 0 that also has an ungrounded recipe has two things to
    fix, and the feedback says both."""
    s = scored(
        recipe(
            "pending orders",
            "pending orders are excluded alongside cancelled ones",
            "WHERE o.status NOT IN ('cancelled', 'pending')",
        ),
        fact("soft deletes", "160 of 2,000 customer rows are soft-deleted"),
    )
    assert s.value == 0.0
    text = s.text()
    assert "goes stale the instant a row changes" in text
    assert "never did" in text


# ---------------------------------------------------------------- the notes


def test_too_many_entries_is_a_note_and_does_not_change_the_score():
    """The cache is sent in full every turn, so entries are a recurring bill,
    and the feedback says so. How many is right depends on the query, so the
    number is left to grounding."""
    many = scored(*GOOD, *[fact(f"n{i}", f"claim {i}") for i in range(12)])
    assert many.value == scored(*GOOD).value == 1.0
    assert "a bill that recurs" in many.text()


def test_a_paragraph_sized_entry_is_a_note_and_does_not_change_the_score():
    giant = scored(recipe("everything", "x " * 200, GOOD[0].sql_fragment))
    assert giant.value == 1.0
    assert "a note, not a paragraph" in giant.text()
    assert "Only 1 entries" in giant.text()


def test_the_size_of_the_prompt_does_not_change_the_score():
    """Size is a limit the adapter applies before the call, not a term here.
    See `ExtractAdapter`'s `max_chars`."""
    assert scored(*GOOD, tokens_in=700, tokens_out=300).value == 1.0
    assert scored(*GOOD, tokens_in=7_000, tokens_out=3_000).value == 1.0
    assert "tokens" not in scored(*GOOD, tokens_in=7_000).text()


# ------------------------------------------------------------------- plumbing


def test_a_failed_call_scores_zero_rather_than_raising():
    s = scored(error="APIError: 500")
    assert s.value == 0.0
    assert "did not produce usable output" in s.text()


def test_the_feedback_names_the_entry_and_quotes_the_text():
    """GEPA's reflection step proposes a targeted mutation from this prose. A
    bare number gives it nothing to aim at."""
    s = scored(fact("soft deletes", "160 of 2,000 rows are soft-deleted"), GOOD[0])
    text = s.text()
    assert "soft deletes" in text
    assert "160 of 2,000" in text
    assert "write the rule, not the number" in text
