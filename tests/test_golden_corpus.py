"""Every golden case, executed against a freshly seeded demo database.

A corpus of reference queries is a corpus of claims, and an unrun claim is true
until the day it matters. Three things go wrong quietly here, and a search is
the worst place to find any of them:

A reference that no longer parses, because somebody renamed a column in
`demo.sql`. That ends a paid run at minute forty.

A reference that parses and comes back empty. Every candidate then scores the
same on that case, and a corpus with a dead case in it looks exactly like GEPA
finding nothing.

A reference that drifted away from the number written beside it. This is the one
where two queries are wrong together — the reference was subtly wrong when it
was written, or `demo.sql` moved and both sides moved with it — and the literal
number is the only witness there is.

The connection is `reader_conn` on purpose: the SELECT-only role the agent
connects as. A reference needing a privilege the agent does not have would pass
as the owner here and fail inside a rollout.

Requires `make up && make migrate && make seed`. No model calls.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from sqlalchemy import inspect

from app import db
from tools.gepa import golden, reference, resultset
from tests.test_coldpath import pool  # noqa: F401 — opens the engines

ROOT = Path(__file__).resolve().parents[1]

CASES = golden.load()
WITH_EXPECT = [c for c in CASES if c.expect is not None]

# Below this, GEPA fits whichever questions happen to be in the corpus.
# `tools/gepa/cli.py` warns at it; this file refuses under it.
THIN = 12
FLOOR = 15


def ids(case: golden.GoldenCase) -> str:
    return case.name


def questions() -> set[str]:
    """`demo/questions.txt` as `read_questions` reads it, without importing the
    CLI — these tests must not need the client package on the path."""
    lines = (ROOT / "demo" / "questions.txt").read_text(encoding="utf-8").splitlines()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


async def run(conn, case: golden.GoldenCase):
    return [tuple(row) for row in (await conn.exec_driver_sql(case.reference_sql)).fetchall()]


# ------------------------------------------------------------- case by case


@pytest.mark.parametrize("case", CASES, ids=ids)
async def test_the_reference_query_runs_and_returns_rows(case, reader_conn):
    """Parses, executes as the agent's own role, and answers something."""
    rows = await run(reader_conn, case)
    assert rows, "an empty reference scores every candidate identically"


@pytest.mark.parametrize("case", WITH_EXPECT, ids=ids)
async def test_the_reference_agrees_with_the_number_beside_it(case, reader_conn):
    """The only check here that fails loudly.

    Compared through the same comparator the metric uses, so the corpus and the
    metric cannot come to disagree about what equal means.
    """
    rows = await run(reader_conn, case)

    match = resultset.compare(case.expect, rows, ordered=case.ordered)

    assert match.exact, (
        f"{case.name}: wrote down {case.expect}, query returned "
        f"{[list(r) for r in resultset.rows_of(rows)]}"
    )


@pytest.mark.parametrize("case", CASES, ids=ids)
async def test_every_table_it_claims_to_touch_exists(case, reader_conn):
    """`tables` is what a split holds an area out by. A typo there silently
    stops that working, and nothing else would ever mention it."""
    real = set(await reader_conn.run_sync(lambda c: inspect(c).get_table_names()))

    assert set(case.tables) <= real, f"{case.name} names a table that is not there"
    assert case.tables, "a case that touches no table cannot be held out"


@pytest.mark.parametrize("case", CASES, ids=ids)
async def test_an_ordered_case_actually_orders(case):
    """A case claiming row order is part of the answer, whose reference does not
    say so, is a case gated on whatever the planner felt like that day."""
    if not case.ordered:
        return
    assert "order by" in case.reference_sql.lower(), case.name


# --------------------------------------------------------- the corpus itself


def test_there_are_enough_cases_to_hold_some_out():
    assert len(CASES) >= FLOOR, f"CHALLENGE asks for {FLOOR}"
    assert len(CASES) > THIN, f"under {THIN} GEPA fits the corpus, not the task"


def test_the_names_and_the_questions_are_both_unique():
    """A duplicated name silently halves the corpus: the gate and the
    reflective dataset key on it."""
    assert len({c.name for c in CASES}) == len(CASES)
    assert len({c.question for c in CASES}) == len(CASES)


def test_every_question_is_one_the_demo_actually_asks():
    """Binds this corpus to `demo/questions.txt`. Without it the two drift into
    asking subtly different things, and the corpus measures a question nobody
    ever saw the agent answer."""
    missing = {c.question for c in CASES} - questions()
    assert not missing, f"not in demo/questions.txt: {sorted(missing)}"


def test_enough_cases_carry_a_literal_answer():
    """Reference-only is a comparison between two queries, and two queries can
    be wrong together. Half the corpus is reference-only because
    `order_item.price` carries a now()-dependent discount, so the other half has
    to hold the line."""
    assert len(WITH_EXPECT) >= 8


def test_at_least_one_case_makes_row_order_part_of_the_answer():
    assert [c for c in CASES if c.ordered]


def test_every_written_answer_round_trips_through_json():
    """`expect` is only written where every cell is an exact integer, a short
    label or null. A money literal would arrive back from JSON as a float, and
    the gate would report drift on a value that never moved."""
    for case in WITH_EXPECT:
        for row in case.expect:
            for cell in row:
                assert isinstance(cell, (int, str)) or cell is None, case.name


def test_the_reference_is_a_query_and_not_a_statement():
    """These run as a role with SELECT and nothing else, so anything else would
    fail anyway. It fails here instead, where the message says which case."""
    for case in CASES:
        assert case.reference_sql.lstrip().upper().startswith(("SELECT", "WITH")), case.name


# ----------------------------------------------------- resolving them at once


async def test_the_whole_corpus_resolves_in_one_transaction(pool):
    """What a rollout batch does before it scores anything.

    One transaction, so every reference sees one `now()`. Without that the
    date-windowed questions would each be answered at a different moment and a
    candidate could be marked wrong for the gap between two of them.
    """
    resolved = await reference.resolve(CASES)

    assert set(resolved) == {case.name for case in CASES}
    assert all(r.rows for r in resolved.values())
    assert len({r.at for r in resolved.values()}) == 1


async def test_a_reference_that_lost_its_answer_stops_the_run(pool):
    """Drift raises rather than scoring zero, and this is the difference that
    matters: a candidate falling over must not end a run, because one bad
    rollout out of a hundred and fifty is noise. A corpus falling over must,
    because every rollout after it is measuring nothing.
    """
    wrong = dataclasses.replace(CASES[0], expect=[[1]])

    with pytest.raises(reference.ReferenceDrift, match="disagrees"):
        await reference.resolve([wrong])


async def test_a_reference_that_returns_nothing_stops_the_run(pool):
    """An empty reference scores every candidate the same, which reads as GEPA
    finding nothing rather than as a broken case."""
    empty = dataclasses.replace(
        CASES[0],
        reference_sql="SELECT 1 WHERE false",
        expect=None,
    )

    with pytest.raises(reference.ReferenceDrift, match="came back empty"):
        await reference.resolve([empty])


async def test_resolving_cannot_write(pool):
    """`resolve` runs authored SQL from a tracked directory, which is the least
    dangerous SQL in this repo. It still goes through the read-only transaction,
    so a case file that grew a DELETE is refused rather than executed."""
    from app import dialects

    delete = dataclasses.replace(
        CASES[0], reference_sql="DELETE FROM customer", expect=None
    )

    with pytest.raises(Exception) as e:
        await reference.resolve([delete])
    assert dialects.is_read_only_error(e.value)
