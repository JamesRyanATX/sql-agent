"""Is this result set the answer? Comparison rules, and nothing else.

The whole-turn metric needs one judgement no prompt can make for it: the
candidate wrote its own SQL, ran it, and got rows back — are those the rows?
Everything hard about that question is in here, kept out of `metric_turn` so
that file reads as gates, terms and weights the way `metric_extract` does, and
so `tests/test_golden_corpus.py` can check a corpus without importing a metric.

**Column names are discarded at the door.** What a query calls its columns is
the model's to choose; `customer_count` and `n` and `total` are the same answer.
How many values a row carries is not a choice, and neither are the values.

Two rules here are relaxations, and both exist because a strict reading would
score correct answers wrong:

*Money is compared to the cent.* `numeric(10,2)` plus a `SUM` means two correct
queries differ in the third decimal depending on whether one wrapped the sum in
`ROUND`. Quantising both sides is a rule a reader can check. A tolerance
constant is a number nobody chose.

*Ties may be ordered either way.* An ordered case is ordered by its measure, and
where two rows share a measure no correct query has an opinion about which comes
first. Comparing the sequence of measures rather than the sequence of rows is
exactly "sorted correctly, ties anywhere".

Pure. No database, no clock, no model.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from itertools import combinations
from typing import Any, Sequence

Cell = Any
Row = tuple[Cell, ...]

CENTS = Decimal("0.01")

# Every value right and the sequence wrong is half the work. It is gated out
# either way — `exact` stays False — so this only ranks failures against each
# other, which is what lets the feedback distinguish "wrong rows" from "right
# rows, wrong order". Those are different mutations.
ORDER_CREDIT = 0.5

# A candidate may return one column the answer does not ask for — the id beside
# the name — and still be right. One, not several: the projection is a search
# over subsets, and an unbounded one is a permissive comparison wearing a
# helpful hat.
MAX_EXTRA_COLUMNS = 1

# NaN is never equal to itself, which would make a Counter behave in a way
# nothing else here does. Nothing in the demo corpus produces one; this is so
# the next corpus does not discover it the hard way.
NAN = "<nan>"


@dataclass(frozen=True)
class Match:
    """What the comparison found, in enough detail to write prose from.

    `missing` and `extra` are the point. "12 rows came back; the answer has 4"
    tells a reflection model that something is wrong. The three spellings of
    west that came back and the one row that did not tell it what.
    """

    exact: bool
    similarity: float
    arity: tuple[int, int]  # (candidate, reference)
    counts: tuple[int, int]  # (candidate, reference)
    missing: list[Row] = field(default_factory=list)
    extra: list[Row] = field(default_factory=list)
    order_ok: bool = True
    first_disorder: int | None = None
    # Which candidate columns were used, when one had to be dropped to match.
    projection: tuple[int, ...] | None = None


def compare(
    candidate: Sequence[Any], reference: Sequence[Any], *, ordered: bool
) -> Match:
    """One candidate result set against one reference result set.

    Both sides may be rows of dicts, as the graph emits them, or rows of
    sequences, as a golden case's written-down answer is.
    """
    left, right = rows_of(candidate), rows_of(reference)
    left_arity, right_arity = _arity(left), _arity(right)

    if left_arity == right_arity:
        return _match(left, right, ordered=ordered, projection=None)

    # A one-column answer takes a one-column result, always. Otherwise
    # `SELECT count(*), count(*) FILTER (WHERE deleted_at IS NULL)` matches
    # 1,840 on its second column while the answer above it says 2,000.
    droppable = (
        right_arity >= 2
        and left_arity > right_arity
        and left_arity - right_arity <= MAX_EXTRA_COLUMNS
    )
    if droppable:
        for keep in combinations(range(left_arity), right_arity):
            projected = [tuple(row[i] for i in keep) for row in left]
            found = _match(projected, right, ordered=ordered, projection=keep)
            if found.exact:
                return found

    return Match(
        exact=False,
        similarity=0.0,
        arity=(left_arity, right_arity),
        counts=(len(left), len(right)),
        missing=list(right),
        extra=list(left),
    )


def rows_of(raw: Sequence[Any]) -> list[Row]:
    """Rows as tuples of normalised values, whatever shape they arrived in.

    A dict's keys are dropped here and nowhere else, so there is one place to
    read to know that column names never reach a comparison.
    """
    rows = []
    for row in raw:
        values = row.values() if isinstance(row, dict) else row
        rows.append(tuple(normalise(v) for v in values))
    return rows


def normalise(value: Cell) -> Cell:
    """One value, in the spelling everything downstream compares.

    A `bool` becomes 1 or 0, which is what Python already thinks it is:
    `True == Decimal(1)` and the two hash alike, so keeping them apart would
    take a wrapper type that every renderer of these rows then has to know
    about. Postgres does distinguish them, and no case in `demo/golden/` returns
    a boolean cell. The test that needs the distinction is the one that should
    introduce it.
    """
    if value is None:
        return value
    if isinstance(value, bool):
        return Decimal(int(value))
    if isinstance(value, (int, float, Decimal)):
        return _number(value)
    # datetime before date: datetime is a subclass of it.
    if isinstance(value, datetime):
        moment = value.astimezone(timezone.utc) if value.tzinfo else value
        return moment.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip().casefold()
    return str(value)


def _number(value: int | float | Decimal) -> Cell:
    """Integers exactly, fractions to the cent.

    Only fractions are quantised. An integer left alone cannot raise on a value
    too large for the decimal context, and `Decimal("1840") == Decimal("1840.00")`
    anyway — Decimal compares and hashes by value, which is what lets these sit
    in a Counter together.

    `str()` first, so a float arrives as the number that was written rather than
    as 0.1000000000000000055.
    """
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    if number.is_nan():
        return NAN
    if math.isinf(float(number)):
        return str(number)
    if number == number.to_integral_value():
        return number
    return number.quantize(CENTS, rounding=ROUND_HALF_UP)


def _arity(rows: list[Row]) -> int:
    """Widest row wins. A ragged result is already a mismatch; this only has to
    report a number rather than pick a fight over which row is typical."""
    return max((len(row) for row in rows), default=0)


def _match(
    candidate: list[Row],
    reference: list[Row],
    *,
    ordered: bool,
    projection: tuple[int, ...] | None,
) -> Match:
    left, right = Counter(candidate), Counter(reference)

    # Duplicate rows are counted, not collapsed. Two identical rows are two
    # rows, and a query producing one where the answer has two has a missing
    # join condition, not a formatting difference.
    overlap = sum((left & right).values())
    total = max(len(candidate), len(reference))
    same_rows = left == right

    order_ok, first_disorder = True, None
    if ordered and same_rows:
        order_ok, first_disorder = _in_order(candidate, reference)

    similarity = 1.0 if total == 0 else overlap / total
    if not order_ok:
        similarity *= ORDER_CREDIT

    return Match(
        exact=same_rows and order_ok,
        similarity=similarity,
        arity=(_arity(candidate), _arity(reference)),
        counts=(len(candidate), len(reference)),
        missing=list((right - left).elements()),
        extra=list((left - right).elements()),
        order_ok=order_ok,
        first_disorder=first_disorder,
        projection=projection,
    )


def _in_order(candidate: list[Row], reference: list[Row]) -> tuple[bool, int | None]:
    """The sequence, with ties free.

    An `ORDER BY` sorts on the measure, so the measure sequence is the ordering
    claim and the row sequence is not. The top-ten-customers case has four
    pairs of customers on equal revenue; comparing rows would score half the
    correct answers wrong on which of two equal rows came first.

    Where no row carries a measure — an ordering by a text column — this
    degenerates to true, so it falls back to comparing the rows themselves. No
    case in the demo corpus needs that; it is here so the next one does not
    silently stop being checked.
    """
    left = [_measure(row) for row in candidate]
    right = [_measure(row) for row in reference]
    if not any(left) and not any(right):
        left, right = candidate, reference

    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return False, i
    return len(left) == len(right), None


def _measure(row: Row) -> Row:
    """The numeric cells of a row, which is what an ORDER BY here sorts on.

    This is why an ordered case's reference must project the label and the
    measure and nothing else: an id column would join the sort key and demand a
    tiebreak the candidate has no reason to use.
    """
    return tuple(cell for cell in row if isinstance(cell, Decimal))
