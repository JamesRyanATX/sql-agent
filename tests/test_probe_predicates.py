"""Does the gate work? Asserted in both directions, and never against a model.

[tests/test_prompt_invariants.py](tests/test_prompt_invariants.py) spends tokens
asking whether the prompt honours its invariants. This asks the prior question —
whether a violation would actually be caught — against hand-built output, for
free.

Both directions, in the register of `tests/test_capabilities.py`: a predicate
that never fires reports a healthy prompt forever, and a predicate that always
fires gets switched off the first time it blocks a good candidate. The first
failure mode is the dangerous one, because it agrees with itself.
"""

from __future__ import annotations

import pytest

from app import store
from tools.gepa import detect, probes
from tools.gepa.cases import ExtractCase
from tools.gepa.replay import Replayed

CASE = ExtractCase.authored(
    name="t",
    question="how many customers do we have?",
    sql="SELECT count(*) FROM customer WHERE deleted_at IS NULL AND lower(region) = 'west'",
    findings="customer.deleted_at is the soft-delete flag",
    filed={
        "active customer": "an active customer is one whose deleted_at is null",
        # The name from the live incident, so the paraphrase of it can be tested
        # against the thing that actually happened.
        "active customer count": "how many customers have a null deleted_at",
    },
)


def replayed(*entries: store.CacheEntry, error: str | None = None) -> Replayed:
    return Replayed(case=CASE, entries=list(entries), error=error)


def recipe(name: str, claim: str, fragment: str) -> store.CacheEntry:
    from app.graph import grounded_in

    return store.CacheEntry(
        kind="recipe",
        name=name,
        claim=claim,
        sql_fragment=fragment,
        verified=grounded_in(fragment, CASE.sql),
    )


def fact(name: str, claim: str) -> store.CacheEntry:
    return store.CacheEntry(kind="schema_fact", name=name, claim=claim)


# ------------------------------------------------------------------ the census


@pytest.mark.parametrize(
    "claim",
    [
        "160 of 2,000 customer rows are soft-deleted",
        "there are 1,840 active customers",
        "there are 1840 active customers",
        "12.5% of orders are cancelled",
        "the west region (currently 460 customers) is the largest",
        "roughly 8 percent of rows are soft-deleted",
    ],
)
def test_a_census_is_caught_however_it_is_spelled(claim):
    assert detect.census_hits(claim), claim
    ok, _ = probes.PREDICATES["no_census"](replayed(fact("x", claim)), {})
    assert not ok


@pytest.mark.parametrize(
    "claim",
    [
        "deleted_at is a nullable soft-delete flag",
        "orders.status takes 4 values: pending, paid, cancelled, refunded",
        "order_item holds one row per line item",
        "the timestamp column is `created`, not `created_at`",
        "region casing varies: west, West and WEST all occur",
    ],
)
def test_a_rule_about_shape_is_not_a_census(claim):
    """The floor of 100 is load-bearing. Small integers appear in legitimate
    claims about cardinality and enum sizes often enough that catching them
    would train the prompt to stop describing shape at all."""
    assert not detect.census_hits(claim), claim
    ok, _ = probes.PREDICATES["no_census"](replayed(fact("x", claim)), {})
    assert ok


def test_a_literal_in_a_fragment_is_never_a_census():
    """Over `claim` only — a recipe's whole job is to carry the filter that ran."""
    entry = recipe("west", "customers in the west region", "lower(region) = 'west'")
    ok, _ = probes.PREDICATES["no_census"](replayed(entry), {})
    assert ok


# ---------------------------------------------------------------- the grounding


def test_a_fragment_the_query_never_ran_is_caught():
    entry = recipe(
        "active", "excludes soft-deleted and pending", "deleted_at IS NULL AND status <> 'pending'"
    )
    assert not entry.verified
    ok, reason = probes.PREDICATES["recipes_are_grounded"](replayed(entry), {})
    assert not ok
    assert "never did" in reason


def test_a_fragment_too_short_to_say_anything_is_caught():
    """`grounded_in` accepts a token subsequence, so `count(*)` verifies against
    any query that counts. Without this clamp an optimiser told to raise the
    verified rate finds that in about four generations, and every entry ends up
    carrying the authority PLAN_SYSTEM grants a verified recipe."""
    entry = recipe("customers", "how many customers there are", "count(*)")
    assert entry.verified, "the production gate does accept this — that is the point"

    ok, reason = probes.PREDICATES["recipes_are_grounded"](replayed(entry), {})
    assert not ok
    assert "too short" in reason


def test_emitting_no_recipe_at_all_is_not_a_pass():
    """Vacuous truth is the first thing an optimiser finds."""
    ok, reason = probes.PREDICATES["recipes_are_grounded"](
        replayed(fact("customer", "the table describing people who buy things")), {}
    )
    assert not ok
    assert "no recipe" in reason


def test_a_recipe_copied_from_the_query_passes():
    entry = recipe(
        "active customer",
        "an active customer is one whose deleted_at is null",
        "FROM customer WHERE deleted_at IS NULL",
    )
    ok, _ = probes.PREDICATES["recipes_are_grounded"](replayed(entry), {})
    assert ok


# --------------------------------------------------------------- the scope creep

SCOPE = {"name": "revenue", "must_not_mention": ["region", "west"]}


def test_a_general_rule_overwritten_by_a_special_case_is_caught():
    entry = fact("revenue", "revenue for the west region excludes cancelled orders")
    ok, reason = probes.PREDICATES["no_scope_creep"](replayed(entry), SCOPE)
    assert not ok
    assert "inherits the special case" in reason


def test_reusing_a_name_to_refine_the_same_concept_is_allowed():
    """The upsert exists to serve this. A predicate that forbade every reuse
    would be blocking the documented behaviour, not defending an invariant."""
    entry = fact("revenue", "revenue is quantity times price, excluding cancelled orders")
    ok, _ = probes.PREDICATES["no_scope_creep"](replayed(entry), SCOPE)
    assert ok


def test_the_special_case_under_its_own_name_is_allowed():
    entry = fact("revenue by region", "revenue for the west region excludes cancelled")
    ok, _ = probes.PREDICATES["no_scope_creep"](replayed(entry), SCOPE)
    assert ok


# ------------------------------------------------------------- the near collision


@pytest.mark.parametrize(
    "name",
    [
        # The live incident, verbatim: "active customer count" was filed on one
        # turn and this arrived on the next.
        "active customers count",
        "active customers",
        "active_customer",
        "Active Customer",
    ],
)
def test_a_paraphrase_of_a_filed_name_is_caught(name):
    ok, reason = probes.PREDICATES["no_near_collision"](
        replayed(fact(name, "some claim")), {}
    )
    assert not ok
    assert "two entries saying one thing" in reason


def test_reusing_the_filed_name_exactly_is_not_a_collision():
    """Exact reuse is how an entry gets refined; the upsert merges it."""
    ok, _ = probes.PREDICATES["no_near_collision"](
        replayed(fact("active customer", "refined claim")), {}
    )
    assert ok


def test_a_qualifier_that_changes_the_scope_earns_its_own_name():
    """The limit of what this predicate can claim, and it is deliberate.

    EXTRACT_SYSTEM says a narrower or broader concept — "revenue" against
    "revenue by region", "active customer" against "active customer in a
    region" — *should* get its own name. So an added qualifier word is the
    documented right answer, not a collision, and a detector that flagged it
    would be enforcing the opposite of the invariant. What is left to catch is
    a paraphrase at the same scope: a plural, a typo, punctuation. That is also
    exactly the failure that was observed.
    """
    for name in ("active customer in a region", "revenue by region"):
        ok, _ = probes.PREDICATES["no_near_collision"](
            replayed(fact(name, "some claim")), {}
        )
        assert ok, name


def test_an_unrelated_name_is_not_a_collision():
    ok, _ = probes.PREDICATES["no_near_collision"](
        replayed(fact("revenue", "qty x price")), {}
    )
    assert ok


# ------------------------------------------------------------------- plumbing


def test_a_failed_call_fails_every_probe_rather_than_raising():
    """GEPA's adapter contract: a candidate that crashes scored badly, it did
    not end the run."""
    for probe in probes.load("extract"):
        ok, reason = probes.check(probe, replayed(error="APIError: 500"))
        assert not ok
        assert "the call failed" in reason


def test_every_probe_names_an_invariant_a_predicate_and_a_reason():
    for probe in probes.load("extract"):
        assert probe.predicate in probes.PREDICATES, probe.name
        assert probe.invariant.strip(), probe.name
        assert probe.cites.strip(), probe.name
        # The `why` is the failure that motivated the probe. A probe without one
        # is a rule nobody can evaluate the removal of.
        assert len(probe.why) > 80, probe.name
        assert probe.case.user_message.startswith("Question: ")


def test_a_probe_case_is_built_by_the_function_production_uses():
    """So a probe exercises the real message assembly, not an imitation."""
    probe = next(p for p in probes.load("extract") if p.case.filed)
    assert "Already filed" in probe.case.user_message
    for name in probe.case.filed:
        assert f"- {name}: " in probe.case.user_message
