"""Detectors for the invariants that live only as prose.

CLAUDE.md's invariant list mixes two kinds. "A human's `pinned` entry is never
overwritten" is enforced by a `WHERE` clause in `store.write_entries`' upsert and
already has a test. "Neither kind is a census" exists only as five lines inside
`EXTRACT_SYSTEM` — a string a candidate prompt is free to delete, whose failure
surfaces turns later when a stale count is read back as fact.

This module is the code half of those prose rules. Pure functions over model
output, no imports from `app`, shared by the probe predicates (a hard 0/1 gate)
and the trainset metric (proportional, because a cliff makes the gradient noise).
"""

from __future__ import annotations

import re

# A count, a percentage, or a parenthetical. Each pattern is one of the examples
# EXTRACT_SYSTEM gives, which is deliberate: the detector should recognise the
# thing the prompt names, so that deleting the paragraph and failing the check
# are the same event.
#
# The floor of 100 is doing real work. "one row per line item" and "3 statuses"
# are shape, not census; "1,840 active customers" and "160 of 2,000" are the
# answer to today's question. Small integers appear in legitimate claims about
# cardinality and enum sizes often enough that catching them would train the
# prompt to stop describing shape.
_CENSUS = (
    # 100+, with or without thousands separators. The demo's own gate spells it
    # `1,?840` for the same reason: the model writes it both ways.
    re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"\bpercent\b", re.I),
    # "160 of 2,000" — the literal example in the prompt, and it slips the
    # threshold above whenever both numbers are small.
    re.compile(r"\b\d[\d,]*\s+of\s+\d[\d,]*\b", re.I),
    re.compile(r"\((?:currently|today|as of|at present)\b[^)]*\)", re.I),
)


def census_hits(claim: str) -> list[str]:
    """The substrings of a claim that read as a census, or an empty list.

    Over `claim` only — never `sql_fragment`, where `WHERE status <> 'cancelled'`
    and a literal threshold are exactly what a recipe is supposed to carry.
    """
    hits: list[str] = []
    for pattern in _CENSUS:
        hits.extend(m.group(0) for m in pattern.finditer(claim or ""))
    return hits


# ---------------------------------------------------------------- name hygiene


def normalise(name: str) -> str:
    """Fold a cache-entry name for collision detection.

    Crude on purpose — deterministic and readable beats linguistically correct.
    A wrong fold makes a probe report a collision that a human can see is not
    one, which is a five-second dismissal; a clever one that is wrong on a
    Tuesday is an afternoon.
    """
    folded = re.sub(r"[^a-z0-9 ]+", " ", (name or "").casefold())
    words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in folded.split()]
    return " ".join(words)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein, iterative, two rows. Names are short; this is not hot."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def near_collisions(
    emitted: list[str], filed: list[str], *, max_distance: int = 2
) -> list[tuple[str, str]]:
    """(new name, filed name) pairs that are variants rather than the same key.

    The observed failure this defends: a live run produced "active customer
    count" on one turn and "active customers count" on the next. The upsert
    keys on `(connection_id, name)`, so those are two entries saying one thing,
    and the second does not refine the first — it sits beside it and both get
    loaded into every later prompt.

    An *exact* match is not a collision. Reusing a name is the documented way to
    refine an entry, and the upsert exists to make it work.
    """
    folded_filed = {normalise(f): f for f in filed if f}
    found: list[tuple[str, str]] = []
    for name in emitted:
        if not name or name in filed:
            continue
        candidate = normalise(name)
        for folded, original in folded_filed.items():
            if candidate == folded or edit_distance(candidate, folded) <= max_distance:
                found.append((name, original))
                break
    return found


# ---------------------------------------------------------------- fragment size

# The clamp that stops the grounding term being free. `grounded_in` accepts an
# order-preserving token subsequence, so `count(*)` is grounded against any query
# that counts and `customer` against any query that reads the table. An optimiser
# told to maximise the verified rate finds that in about four generations, and
# the result is a cache where every entry carries the authority PLAN_SYSTEM
# grants a verified recipe and none of them earned it.
MIN_FRAGMENT_TOKENS = 4

_STRUCTURAL = {"select", "from", "where", "count", "*", "(", ")", ",", "sum", "as"}


def informative(fragment: str | None, tokens: list[str]) -> bool:
    """Does this fragment say enough to be worth verifying?

    `tokens` comes from `graph._tokens`, so the tokenisation is the gate's own.

    What this catches: `customer`, `count(*)`, `SELECT count(*) FROM` — spans
    made only of structure, which verify against anything of their shape.

    What it does not, and cannot: a fragment that names real columns but drops
    the predicate that makes the concept what it is. `sum(oi.qty * oi.price)`
    clears this bar while omitting `WHERE status <> 'cancelled'`, which is the
    entire trap. No token count separates those two, so the honest boundary is
    here rather than in a cleverer threshold — the grounding probe and a human
    reading the winning prompt are what cover the rest.
    """
    if not fragment:
        return False
    distinct = {t for t in tokens}
    return len(distinct) >= MIN_FRAGMENT_TOKENS and not distinct <= _STRUCTURAL
