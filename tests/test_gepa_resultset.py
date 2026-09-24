"""When two result sets are the same answer.

This is the only judgement in the whole-turn metric with a right answer behind
it, so everything else that metric does rests on these rules being the ones a
person would agree with. Two failure directions, both invisible in a score:

Too strict, and correct candidates are scored wrong. Then `correct` never
reaches 1.0, the cost and tool-call terms are gated off on every case, and a
search has nothing to optimise but a term it cannot move. That looks exactly
like GEPA finding nothing.

Too loose, and a wrong answer scores 1.0. Then a promoted tool description was
chosen on evidence that was not evidence, and the first person to check the
number on stage finds out.

Pure. No database, no model.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from tools.gepa.resultset import compare, normalise


def rows(*values):
    """Rows as the graph emits them: dicts, with names nobody agreed on."""
    return [{f"c{i}": v for i, v in enumerate(row)} for row in values]


# ------------------------------------------------------------------- values


def test_every_spelling_of_the_same_number_is_the_same_number():
    """psycopg hands back `int` for a count and `Decimal` for a `numeric`, and
    a golden case writes `1840` as JSON. Three spellings, one answer."""
    for spelling in (1840, 1840.0, Decimal("1840"), Decimal("1840.00")):
        assert compare(rows((spelling,)), rows((1840,)), ordered=False).exact


def test_money_agrees_to_the_cent():
    """`SUM(qty * price)` and `ROUND(SUM(qty * price), 2)` are the same answer
    and differ in the third decimal. Half a cent apart is not."""
    assert compare(rows((768.6297,)), rows((768.63,)), ordered=False).exact
    assert not compare(rows((768.62,)), rows((768.63,)), ordered=False).exact


def test_null_is_itself_and_is_not_zero():
    """A region nobody filled in and a region with no customers are different
    facts, and a comparison that conflates them hides the second one."""
    assert compare(rows((None,)), rows((None,)), ordered=False).exact
    assert not compare(rows((None,)), rows((0,)), ordered=False).exact
    assert not compare(rows((None,)), rows(("",)), ordered=False).exact


def test_a_boolean_compares_as_the_number_python_says_it_is():
    """Pinned because it is a choice, not an accident. Postgres distinguishes
    `true` from `1`; Python does not, and separating them would need a wrapper
    type that every renderer of these rows then has to handle. No case in
    `demo/golden/` returns a boolean cell, so the cost buys nothing today.

    If a case ever does, this test is where the distinction gets argued.
    """
    assert compare(rows((True,)), rows((1,)), ordered=False).exact
    assert compare(rows((False,)), rows((0,)), ordered=False).exact
    # Not a string, though: a text column holding the word is a different
    # answer, and that collision is the one a naive sentinel would have made.
    assert not compare(rows((True,)), rows(("true",)), ordered=False).exact


def test_labels_ignore_surrounding_space_and_case():
    """The model picks how it spells its own output labels. It does not pick
    the rows — `test_mixed_casing_is_caught_by_the_counts` is the other half of
    this, and it is the one that matters."""
    assert compare(rows(("  West ",)), rows(("west",)), ordered=False).exact


def test_a_date_matches_its_own_iso_spelling():
    """`date_trunc(...)::date` comes back as a `date`; a golden case writes a
    string. Same day."""
    assert compare(rows((date(2026, 1, 1),)), rows(("2026-01-01",)), ordered=False).exact


def test_two_timestamps_at_one_instant_match():
    """Same moment, two zones. The database's session zone is not part of the
    answer."""
    utc = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    other = utc.astimezone(timezone.max)
    assert normalise(utc) == normalise(other)


def test_a_number_the_graph_wrote_as_a_string_is_still_the_number():
    """The graph serialises rows with `json.dumps(default=str)`, so a Postgres
    `numeric` reaches the metric as `"1803600.84"` while the reference fetched
    the `Decimal`. Same answer. This is the mismatch that scored every revenue
    case in the config run at zero."""
    assert compare(rows(("1803600.84",)), rows((Decimal("1803600.84"),)), ordered=False).exact
    assert compare(rows(("1840",)), rows((1840,)), ordered=False).exact
    assert compare(rows(("-5",)), rows((-5,)), ordered=False).exact
    # The cents rule still applies once it is a number again.
    assert compare(rows(("768.6297",)), rows((768.63,)), ordered=False).exact
    assert not compare(rows(("768.62",)), rows((768.63,)), ordered=False).exact


def test_a_label_that_is_not_a_plain_number_stays_a_label():
    """`Decimal()` would take these. A label is not a number because a parser
    is generous."""
    for text in ("12 west", "1_000", "inf", "nan", "1,840", "$5"):
        assert normalise(text) == text.casefold()
        assert not compare(rows((text,)), rows((1840,)), ordered=False).exact


def test_a_timestamp_the_graph_wrote_as_a_string_is_still_the_instant():
    """`str(datetime)` uses a space where `isoformat()` uses a T. The graph
    writes the first; the reference side has the object."""
    aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    # Naive on purpose: a `timestamp without time zone` column comes back
    # without one, and its string has no offset to parse.
    naive = datetime(2026, 1, 1, 12, 0)
    assert compare(rows((str(aware),)), rows((aware,)), ordered=False).exact
    assert compare(rows((str(naive),)), rows((naive,)), ordered=False).exact
    # And a bare date is a date, not midnight.
    assert compare(rows(("2026-01-01",)), rows((date(2026, 1, 1),)), ordered=False).exact
    assert not compare(rows(("2026-01-01",)), rows((naive.replace(hour=0),)), ordered=False).exact


def test_a_label_shaped_like_a_date_but_not_one_stays_a_label():
    assert normalise("2026-Q1 plan") == "2026-q1 plan"
    assert normalise("2026-13-45") == "2026-13-45"


# --------------------------------------------------------------------- rows


def test_column_names_are_not_part_of_the_answer():
    """The whole rule, in one assertion. `customer_count` and `n` are the same
    answer, and insisting otherwise scores a candidate on a word it was free to
    choose."""
    assert compare(
        [{"active_customers": 1840}], [{"n": 1840}], ordered=False
    ).exact


def test_duplicate_rows_are_counted_rather_than_collapsed():
    """Two identical rows are two rows. A query returning one where the answer
    has two has lost a join condition, and a set comparison would call that
    correct."""
    assert not compare(rows(("west", 500)), rows(("west", 500), ("west", 500)),
                       ordered=False).exact


def test_mixed_casing_is_caught_by_the_counts():
    """TRAP 2, as the metric sees it. Grouping on the raw `region` column
    reports west three times with the count split between them. Casefolding the
    labels does not disarm this: what is wrong is the row count and the
    numbers, not the spelling of the word.
    """
    naive = rows(("west", 350), ("West", 100), ("WEST", 50), ("east", 500))
    answer = rows(("west", 500), ("east", 500))

    found = compare(naive, answer, ordered=False)

    assert not found.exact
    assert found.counts == (4, 2)
    assert (Decimal("500"),) not in found.missing  # the rows carry two cells
    assert any(row[0] == "west" and row[1] == 500 for row in found.missing)


def test_what_was_missing_and_what_was_extra_are_both_reported():
    """The feedback is built out of these. "4 rows came back; the answer has 2"
    says something is wrong; these say what."""
    found = compare(
        rows(("west", 350), ("north", 500)),
        rows(("west", 500), ("north", 500)),
        ordered=False,
    )

    assert [tuple(r) for r in found.missing] == [("west", Decimal("500"))]
    assert [tuple(r) for r in found.extra] == [("west", Decimal("350"))]


# ------------------------------------------------------------------- order


def test_ties_may_be_ordered_either_way():
    """`top_customers_by_revenue` has four pairs of customers on equal revenue.
    No correct query has an opinion about which of two equal rows comes first,
    so comparing rows positionally would score half the right answers wrong.
    What is compared is the sequence of measures.
    """
    answer = rows(("ana", 10), ("bo", 9), ("cy", 9), ("di", 8))
    swapped = rows(("ana", 10), ("cy", 9), ("bo", 9), ("di", 8))

    assert compare(swapped, answer, ordered=True).exact


def test_order_is_checked_when_the_measures_arrived_as_strings():
    """The graph writes a `numeric` revenue as a string, and the order check
    picks its measures by `Decimal`. Before strings were read back, a
    candidate's money column carried no measures at all, so an ordered money
    case could never have its order checked."""
    answer = rows(("ana", Decimal("10.50")), ("bo", Decimal("9.25")), ("di", Decimal("8.00")))
    right = rows(("ana", "10.50"), ("bo", "9.25"), ("di", "8.00"))
    reversed_ = rows(("di", "8.00"), ("bo", "9.25"), ("ana", "10.50"))

    assert compare(right, answer, ordered=True).exact
    found = compare(reversed_, answer, ordered=True)
    assert not found.exact
    assert not found.order_ok
    assert found.first_disorder == 0


def test_a_genuinely_wrong_order_is_still_wrong():
    """The relaxation above must not become "order is not checked"."""
    answer = rows(("ana", 10), ("bo", 9), ("cy", 9), ("di", 8))
    sorted_up = rows(("di", 8), ("bo", 9), ("cy", 9), ("ana", 10))

    found = compare(sorted_up, answer, ordered=True)

    assert not found.exact
    assert not found.order_ok
    assert found.first_disorder == 0


def test_right_rows_in_the_wrong_order_earn_half():
    """A different mutation from the wrong rows, so a different number. It can
    only rank failures: `exact` is False either way."""
    answer = rows(("ana", 10), ("bo", 9))
    found = compare(rows(("bo", 9), ("ana", 10)), answer, ordered=True)

    assert found.similarity == 0.5
    assert not found.exact


def test_order_is_not_checked_when_the_case_does_not_ask_for_it():
    """Most questions do not name an ordering, and inventing one would score a
    candidate on a claim the question never made."""
    assert compare(
        rows(("north", 500), ("west", 500)),
        rows(("west", 500), ("north", 500)),
        ordered=False,
    ).exact


def test_ordering_by_a_text_column_falls_back_to_comparing_rows():
    """No row carries a measure, so the measure comparison would be vacuously
    true. Nothing in the demo corpus reaches this; it exists so the next corpus
    does not quietly stop being checked."""
    answer = rows(("ana",), ("bo",), ("cy",))

    assert compare(answer, answer, ordered=True).exact
    assert not compare(rows(("bo",), ("ana",), ("cy",)), answer, ordered=True).exact


# ------------------------------------------------------------------- arity


def test_an_extra_column_beside_the_answer_is_allowed():
    """`SELECT c.id, c.name, SUM(...)` against an answer of name and revenue.
    The id is not wrong, it is surplus, and the rows underneath it are right."""
    found = compare(
        rows((7, "ana", 10), (9, "bo", 9)),
        rows(("ana", 10), ("bo", 9)),
        ordered=False,
    )

    assert found.exact
    assert found.projection == (1, 2)


def test_a_scalar_answer_never_projects():
    """The most important rule here. This candidate answered "2,000 customers"
    and put the right number in a second column it did not use. Allowing a
    projection would score it correct on a number its own answer contradicts.
    """
    found = compare(rows((2000, 1840)), rows((1840,)), ordered=False)

    assert not found.exact
    assert found.similarity == 0.0
    assert found.arity == (2, 1)


def test_too_many_extra_columns_is_a_mismatch():
    """`SELECT *` against a two-column answer is not a right answer with
    decoration on it. The projection is bounded so it cannot become a search
    for any reading under which the candidate was correct."""
    found = compare(
        rows((1, "ana", "west", 10)), rows(("ana", 10)), ordered=False
    )

    assert not found.exact
    assert found.arity == (4, 2)


def test_a_narrower_result_than_the_answer_is_a_mismatch():
    """It left a column out. Nothing recovers from that, and the arity is
    reported so the feedback can say which way round it was."""
    found = compare(rows(("ana",)), rows(("ana", 10)), ordered=False)

    assert not found.exact
    assert found.arity == (1, 2)


def test_an_empty_result_against_a_real_answer_scores_nothing():
    """The query ran and the filters removed everything. `metric_turn` gates
    this before it ever gets here; the number is still right."""
    found = compare([], rows(("west", 500)), ordered=False)

    assert not found.exact
    assert found.similarity == 0.0
    assert found.counts == (0, 1)
