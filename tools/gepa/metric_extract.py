"""What a candidate `extract` prompt is worth, with no ground truth.

Nobody labels whether a cache entry was right. What is computable offline is
whether it is *self-consistent* — grounded in the query that ran, free of counts
that go stale, filed under a name that will not collide — which is what the
invariants in `config/prompts/extract.md` are about.

Two things this does not do. It does not measure correctness: a prompt can score
1.0 on a beautifully grounded recipe for the wrong business concept. And it does
not measure the flywheel, because whether the cache makes turn N+1 cheaper is a
property of a sequence and one `extract` call is not one.

The weights are in one dict so a reader can argue with them, which is the only
defence a number like 0.35 has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.gepa import detect
from tools.gepa.probes import graph_tokens
from tools.gepa.replay import Replayed

WEIGHTS = {
    # The production gate, not a re-implementation of it, clamped so that
    # satisfying it cheaply is not free. See `_grounding`.
    "grounding": 0.35,
    # A prose-only invariant with a delayed failure: a recorded count is right
    # today and wrong forever after, and nothing revisits it.
    "census": 0.25,
    # Overwrites and paraphrases. The upsert reports success either way, so
    # nothing else will ever tell you.
    "names": 0.20,
    # A band, not a direction. "Fewer entries is better" rewards deleting the
    # job, which is the cheapest way to satisfy every other term here.
    "shape": 0.10,
    # One-sided against the recorded baseline, for the same reason.
    "cost": 0.10,
}

# 2-6 entries is what a good turn produces. Outside the band is a mild penalty
# rather than a cliff: a genuinely rich query can establish seven things.
ENTRY_BAND = (2, 6)
# A claim is a note a colleague reads aloud, not a paragraph.
CLAIM_LIMIT = 200


@dataclass
class Score:
    """A scalar for the optimiser and prose for the reflection step.

    The scalar decides selection; the feedback is what lets the reflection model
    propose a targeted mutation rather than a random one.
    """

    value: float
    terms: dict[str, float] = field(default_factory=dict)
    feedback: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n\n".join(self.feedback) if self.feedback else "No problems found."


def score(r: Replayed) -> Score:
    """One case, one candidate.

    Two gates before the weighted terms. The empty-extraction gate is the
    load-bearing one: census, names and cost are all vacuously perfect when
    nothing was recorded, so without it three of five terms reward a prompt for
    doing nothing.
    """
    # A candidate whose output does not fit EXTRACT_SCHEMA scores zero rather
    # than ending the run. That is GEPA's adapter contract.
    if r.error:
        return Score(0.0, {}, [f"The call did not produce usable output: {r.error}"])

    if not r.entries:
        return Score(
            0.0,
            {},
            [
                "Nothing was recorded. `extract` runs only after a query has "
                "already succeeded, so an empty result is the node declining to "
                "do its one job — and the next turn pays full price to "
                "rediscover what this one had in hand."
            ],
        )

    terms: dict[str, float] = {}
    feedback: list[str] = []
    for name, fn in (
        ("grounding", _grounding),
        ("census", _census),
        ("names", _names),
        ("shape", _shape),
        ("cost", _cost),
    ):
        terms[name], note = fn(r)
        if note:
            feedback.append(note)

    value = sum(WEIGHTS[k] * v for k, v in terms.items())
    return Score(value, terms, feedback)


# ----------------------------------------------------------------------- terms


def _grounding(r: Replayed) -> tuple[float, str]:
    """Recipes must be copied from the SQL that ran, and must say something.

    The second half is not optional: `grounded_in` accepts an order-preserving
    token subsequence, so `count(*)` is grounded against any query that counts.
    Optimising the metric derived from the gate is how you destroy the gate.
    """
    if not r.recipes:
        return 0.0, (
            "No recipe was recorded. The SQL that ran expresses at least one "
            "business concept, and a turn that files only schema facts has "
            "learned nothing reusable — the next question re-derives the query."
        )

    good, notes = 0, []
    for e in r.recipes:
        if not e.verified:
            notes.append(
                f'  name="{e.name}"  fragment={e.sql_fragment!r}\n'
                f"    is not a subsequence of the SQL that ran:\n"
                f"    {r.case.sql}\n"
                f"    so it claims something the query never did."
            )
        elif not detect.informative(e.sql_fragment, graph_tokens(e.sql_fragment)):
            notes.append(
                f'  name="{e.name}"  fragment={e.sql_fragment!r}\n'
                f"    is grounded only because it is too short to be wrong. A "
                f"fragment has to carry the filters and joins that make the "
                f"concept what it is, not just name a table or a function."
            )
        else:
            good += 1

    if not notes:
        return 1.0, ""
    return good / len(r.recipes), (
        f"{len(notes)} of {len(r.recipes)} recipes are not properly grounded.\n"
        + "\n".join(notes)
    )


def _census(r: Replayed) -> tuple[float, str]:
    if not r.entries:
        return 1.0, ""
    offenders = [(e, hits) for e in r.entries if (hits := detect.census_hits(e.claim))]
    if not offenders:
        return 1.0, ""

    notes = "\n".join(
        f'  name="{e.name}"  {hits}\n    {e.claim}' for e, hits in offenders
    )
    return 1 - len(offenders) / len(r.entries), (
        f"{len(offenders)} of {len(r.entries)} claims read as a census.\n{notes}\n"
        "    A count or a percentage is this query's answer, not a fact about "
        "the schema. It goes stale the instant a row changes and nothing ever "
        "revisits it — write the rule, not the number it produced today."
    )


def _names(r: Replayed) -> tuple[float, str]:
    """Destructive overwrite, near-collision, self-collision.

    All three are silent in production — `write_entries` upserts and reports
    success — so the only symptom is a later question composing the wrong recipe.
    """
    if not r.entries:
        return 1.0, ""

    problems: list[str] = []
    penalised: set[int] = set()

    for i, e in enumerate(r.entries):
        old = r.case.filed.get(e.name or "")
        # A much shorter claim landing on a filed name is the general rule being
        # overwritten by a special case, destroying it for every later question
        # that composed it.
        if old and len(e.claim) < 0.6 * len(old):
            penalised.add(i)
            problems.append(
                f'  "{e.name}" overwrote a filed entry with a much shorter claim\n'
                f"    was: {old}\n"
                f"    now: {e.claim}"
            )

    for new, filed_name in detect.near_collisions(r.names, list(r.case.filed)):
        penalised.update(i for i, e in enumerate(r.entries) if e.name == new)
        problems.append(
            f'  "{new}" is a paraphrase of the filed "{filed_name}". The upsert '
            f"keys on the name, so these become two entries saying one thing, "
            f"and both are loaded into every later prompt."
        )

    seen: dict[str, int] = {}
    for i, e in enumerate(r.entries):
        if e.name and e.name in seen:
            penalised.add(i)
            problems.append(
                f'  "{e.name}" was emitted twice in one batch — the second write '
                f"silently replaces the first."
            )
        elif e.name:
            seen[e.name] = i

    if not problems:
        return 1.0, ""
    return 1 - len(penalised) / len(r.entries), (
        f"{len(problems)} naming problems.\n" + "\n".join(problems)
    )


def _shape(r: Replayed) -> tuple[float, str]:
    low, high = ENTRY_BAND
    n = len(r.entries)
    if n < low:
        count_score = n / low
        note = (
            f"Only {n} entries from a query this substantial — the next turn "
            f"will have to rediscover what this one already established."
        )
    elif n > high:
        count_score = max(0.0, 1 - (n - high) / (high * 2))
        note = (
            f"{n} entries. The cache is sent in full on every turn, so each one "
            f"is a bill that recurs — record the conventions, not every "
            f"observation."
        )
    else:
        count_score, note = 1.0, ""

    long = [e for e in r.entries if len(e.claim) > CLAIM_LIMIT]
    if long:
        note = (note + "\n" if note else "") + "\n".join(
            f'  name="{e.name}" is {len(e.claim)} characters — a note, not a paragraph.'
            for e in long
        )
    length_score = 1 - (len(long) / len(r.entries) if r.entries else 0)

    return (count_score + length_score) / 2, note


def _cost(r: Replayed) -> tuple[float, str]:
    """One-sided. Cheaper than the baseline earns 1.0, never more.

    An unbounded reward for terseness is the same failure as "fewer entries is
    better", and this is the term most likely to delete a paragraph whose payoff
    no metric here can see.
    """
    baseline = r.case.baseline_tokens_out
    if not baseline:
        return 1.0, ""
    over = (r.tokens_out - baseline) / baseline
    if over <= 0:
        return 1.0, ""
    return max(0.0, 1 - over), (
        f"{r.tokens_out} output tokens against a recorded {baseline} — "
        f"{over:.0%} more for this case."
    )
