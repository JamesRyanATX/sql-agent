"""What a candidate `extract` prompt is worth, with no ground truth.

Nobody labels whether a cache entry was right. What is computable offline is
whether it is *self-consistent* — grounded in the query that ran, free of counts
that go stale, filed under a name that will not collide — which is what the
invariants in `config/prompts/extract.md` are about.

Two things this does not do. It does not measure correctness: a prompt can score
1.0 on a beautifully grounded recipe for the wrong business concept. And it does
not measure the flywheel, because whether the cache makes turn N+1 cheaper is a
property of a sequence and one `extract` call is not one.

The score is one number per case, and it can be said in one sentence: 0 when
the call failed, nothing was recorded, no recipe was recorded, a note records
a count, or a note overwrites or duplicates a filed name; otherwise the
fraction of recipes whose SQL fragment is copied from the query that ran and
carries more than a bare table or function name. It used to be six weighted terms. Two of them, `census`
and `names`, were 1.00 for every candidate in every run, so in practice the
search compared grounding against prompt length, and the committed winner led
by 0.0004. A count in a note and an overwritten name are failures rather than
preferences, so they are gates here, and the invariant probes check them again
on every candidate that survives.

The size of the prompt is not part of the score. `ExtractAdapter` refuses a
prompt longer than a fixed multiple of the current one before any model call
is made; see `targets.MAX_GROWTH`.
"""

from __future__ import annotations

from tools.gepa import detect
from tools.gepa.probes import graph_tokens
from tools.gepa.replay import Replayed

# Shared with `metric_turn`, and re-exported here because every caller in the
# tree already reads it as `metric_extract.Score`.
from tools.gepa.score import Score  # noqa: F401

# 2-6 entries is what a good turn produces. Outside the band is a note in the
# feedback and does not change the score: a genuinely rich query can establish
# seven things.
ENTRY_BAND = (2, 6)
# A claim is a note a colleague reads aloud, not a paragraph.
CLAIM_LIMIT = 200


def score(r: Replayed) -> Score:
    """One case, one candidate.

    Five things score the case 0: the call failed, nothing was recorded, no
    recipe was recorded, a note records a count, or a note overwrites or
    duplicates a filed name. Otherwise the case scores the fraction of its
    recipes whose SQL fragment is copied from the query that ran and carries
    more than a bare table or function name. The no-recipe case is
    `_grounding` returning 0, not a gate of its own.

    The empty-extraction gate comes before the other two because census and
    names are both satisfied when nothing was recorded, so without it a prompt
    that records nothing would pass them.

    The feedback carries every note that applies, including on a case the
    census or names gate scored 0, and including the note about the number and
    size of the entries, which does not change the score. The reflection model
    reads the feedback, so it should see every problem with a case and not only
    the one that decided the number.
    """
    # A candidate whose output does not fit EXTRACT_SCHEMA scores zero rather
    # than ending the run. That is GEPA's adapter contract.
    if r.error:
        return Score(0.0, feedback=[f"The call did not produce usable output: {r.error}"])

    if not r.entries:
        return Score(
            0.0,
            feedback=[
                "Nothing was recorded. `extract` runs only after a query has "
                "already succeeded, so an empty result is the node declining to "
                "do its one job — and the next turn pays full price to "
                "rediscover what this one had in hand."
            ],
        )

    grounding, grounding_note = _grounding(r)
    census, census_note = _census(r)
    names, names_note = _names(r)
    feedback = [
        note for note in (grounding_note, census_note, names_note, _shape(r)) if note
    ]

    if census < 1.0 or names < 1.0:
        return Score(0.0, feedback=feedback)
    return Score(grounding, feedback=feedback)


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


def _shape(r: Replayed) -> str:
    """A note about how many entries there are and how long each one is.

    Feedback only: it does not change the score. The number of entries that is
    right depends on the query, so this says what it sees and leaves the
    judgement to the reflection model.
    """
    low, high = ENTRY_BAND
    n = len(r.entries)
    if n < low:
        note = (
            f"Only {n} entries from a query this substantial — the next turn "
            f"will have to rediscover what this one already established."
        )
    elif n > high:
        note = (
            f"{n} entries. The cache is sent in full on every turn, so each one "
            f"is a bill that recurs — record the conventions, not every "
            f"observation."
        )
    else:
        note = ""

    long = [e for e in r.entries if len(e.claim) > CLAIM_LIMIT]
    if long:
        note = (note + "\n" if note else "") + "\n".join(
            f'  name="{e.name}" is {len(e.claim)} characters — a note, not a paragraph.'
            for e in long
        )
    return note
