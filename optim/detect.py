"""Detectors for the invariants that live only as prose.

Some invariants are enforced by code and already have tests. Others exist only
as lines inside `config/prompts/extract.md` — which a candidate prompt is free
to delete, and whose failure surfaces turns later when a stale count is read
back as fact.

This module is the code half of those prose rules. Pure functions over model
output, no imports from `app`, shared by the probe predicates (a hard 0/1 gate)
and the trainset metric (proportional, because a cliff makes the gradient noise).
"""

from __future__ import annotations

import re

# A count, a percentage, or a parenthetical. Each pattern is one of the examples
# `extract.md` gives, so that deleting the paragraph and failing the check are
# the same event.
#
# The floor of 100 is load-bearing: "one row per line item" and "3 statuses" are
# shape, while "1,840 active customers" is the answer to today's question. Small
# integers appear in legitimate claims about cardinality often enough that
# catching them would train the prompt to stop describing shape.
_CENSUS = (
    # 100+, with or without thousands separators — the model writes it both ways.
    re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"\bpercent\b", re.I),
    # "160 of 2,000" slips the threshold above whenever both numbers are small.
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

    Crude on purpose: deterministic and readable beats linguistically correct,
    because a wrong fold here only makes a probe report a collision a human can
    dismiss at a glance.
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

    The failure this defends: "active customer count" on one turn and "active
    customers count" on the next. The upsert keys on `(connection_id, name)`, so
    those are two entries saying one thing, both loaded into every later prompt.

    An *exact* match is not a collision — reusing a name is how an entry is
    refined, and the upsert exists to make it work.
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
# that counts. An optimiser maximising the verified rate finds that in about four
# generations, and every entry ends up carrying authority none of them earned.
MIN_FRAGMENT_TOKENS = 4

_STRUCTURAL = {"select", "from", "where", "count", "*", "(", ")", ",", "sum", "as"}


def informative(fragment: str | None, tokens: list[str]) -> bool:
    """Does this fragment say enough to be worth verifying?

    `tokens` comes from `graph._tokens`, so the tokenisation is the gate's own.

    Catches spans made only of structure — `customer`, `count(*)`, `SELECT
    count(*) FROM` — which verify against anything of their shape. It cannot
    catch a fragment that names real columns but drops the predicate: `sum(oi.qty
    * oi.price)` clears this bar while omitting `WHERE status <> 'cancelled'`.
    No token count separates those, so the probe and a human reading the winning
    prompt cover the rest.
    """
    if not fragment:
        return False
    distinct = {t for t in tokens}
    return len(distinct) >= MIN_FRAGMENT_TOKENS and not distinct <= _STRUCTURAL
