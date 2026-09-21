"""What a candidate's whole turn was worth, against an answer we know.

`metric_extract` scores one recorded call with no ground truth, so what it can
measure is self-consistency. This one has ground truth: every case in
`demo/golden/` carries a reference query, and both queries run against the same
database at the same moment, so a question anchored to `now()` stops being a
special case.

Which makes the interesting failure the opposite one. `metric_extract` can be
gamed by a prompt that records nothing. This can be gamed by a **reflection**
that reads the answer out of the feedback and writes it into a tool description
— a description saying "the customer table has 1,840 active rows" scores well on
this corpus and nowhere else, and it would be found on stage. So the feedback
below names the rows that differed and never names the query that got them
right. The rows are a diagnosis. The reference SQL is the answer key, and
`tests/test_gepa_metric_turn.py` asserts it never leaks.

The weights are in one dict so a reader can argue with them, and `score` takes
them as an argument so the sensitivity check — cost at 0.15 and at 0.35, does
the winner change — is a flag rather than an edit. A weight nobody varied is a
guess with a decimal point on it.

Pure and synchronous. The reference has already been run; see `reference.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from tools.gepa import resultset
from tools.gepa.golden import GoldenCase
from tools.gepa.reference import Reference
from tools.gepa.replay_turn import ToolCall, TurnReplayed
from tools.gepa.resultset import Match, Row
from tools.gepa.score import Score

WEIGHTS = {
    # The only term with a right answer behind it. Everything else here is a
    # preference. Gated as well as weighted, below: 0.60 of a wrong answer is
    # still a number a cheap wrong candidate can win on.
    "correct": 0.60,
    # One-sided against what the seed spent on this same question, as
    # `metric_extract._cost` is. Cheaper earns 1.0 and never more — an unbounded
    # reward for terseness buys a candidate that skips the explore loop, and the
    # explore loop is what finds `deleted_at`.
    "cost": 0.25,
    # Fewer introspection calls, one-sided. This is the term the four tool
    # descriptions are aimed at: a description saying what a tool returns is one
    # the model does not call twice to find out. Below `cost` because a tool
    # call is already charged there in tokens, and this exists to make the
    # *shape* visible rather than to bill it twice.
    "tool_calls": 0.15,
}

TERMS = tuple(WEIGHTS)

# How many differing rows to quote before summarising. Enough to see a pattern,
# few enough that a reflection prompt is not mostly result set.
SHOWN = 5

WITHHELD = (
    "Cost and tool-call scores are withheld on a wrong answer. They describe "
    "how a right answer was reached, and there is nothing here to describe. A "
    "candidate cannot buy its way past this with a cheaper turn."
)


@dataclass(frozen=True)
class Baseline:
    """What the seed spent on this question, measured this run.

    Not stored on the case. `metric_extract` keeps its baseline on the case
    because it is a recorded property of the trace the case came from; this one
    is a fact about a model version, and putting it in `demo/golden/` would make
    a committed fixture go stale every time the model changed.
    """

    tokens: int
    tool_calls: int


def score(
    r: TurnReplayed,
    case: GoldenCase,
    reference: Reference,
    *,
    baseline: Baseline | None = None,
    weights: dict[str, float] = WEIGHTS,
) -> Score:
    """One golden case, one candidate, one rollout.

    Four gates before the weighted terms, and a fifth after `correct`. They are
    all the same gate as the empty-extraction one in `metric_extract`: cost and
    tool-call counts are vacuously perfect on a turn that did nothing, so
    without them the cheapest way to score well is to answer without querying.
    """
    if r.error:
        return Score(0.0, _zero(), [_broke(r)])
    if not r.sql:
        return Score(0.0, _zero(), [_no_sql(r)])
    if r.sql_error:
        return Score(0.0, _zero(), [_never_ran(r)])
    if reference.rows and not r.rows:
        return Score(0.0, _zero(), [_empty(r, reference)])

    match = resultset.compare(r.rows, reference.rows, ordered=case.ordered)

    terms, feedback = _zero(), []
    terms["correct"], note = _correct(r, case, match)
    if note:
        feedback.append(note)

    if terms["correct"] < 1.0:
        feedback.append(WITHHELD)
        return Score(weights["correct"] * terms["correct"], terms, feedback)

    for name, term in (("cost", _cost), ("tool_calls", _tool_calls)):
        terms[name], note = term(r, baseline)
        if note:
            feedback.append(note)

    return Score(sum(weights[k] * v for k, v in terms.items()), terms, feedback)


def _zero() -> dict[str, float]:
    """Every term, explicitly zero.

    A fresh dict each call: a module-level constant here would be one mutable
    object shared by every Score in a run. And all three keys on every path,
    including the gates, because `terms` becomes `objective_scores` and a
    missing key reshapes the Pareto front without saying so.
    """
    return dict.fromkeys(TERMS, 0.0)


# ------------------------------------------------------------------- the gates


def _broke(r: TurnReplayed) -> str:
    return (
        f"The turn did not complete: {r.error}\n\n"
        "Nothing after this is measurable — no SQL was written, so nothing ran "
        "and nothing was compared. This is a rollout that fell over rather than "
        "a candidate that answered badly. If it keeps happening to one "
        "candidate, that candidate is producing text the graph cannot use."
    )


def _no_sql(r: TurnReplayed) -> str:
    return (
        "The turn answered without querying the database.\n\n"
        f"{_sequence(r)}\n\n"
        f"and then said: {r.answer!r}\n\n"
        "The question asks for something the database holds. An answer "
        "assembled from the schema is a guess with a confident tone on it."
    )


def _never_ran(r: TurnReplayed) -> str:
    note = (
        f"The SQL never ran. After {_plural(r.fix_attempts, 'fix attempt')} the "
        f"database still said:\n\n    {r.sql_error}\n\n"
        f"The last query tried was:\n\n{_indent(r.sql)}\n\n{_sequence(r)}"
    )
    described = _described_tables(r)
    if r.fix_attempts and described:
        note += (
            f"\n\n`describe_table` was called on {_and_list(described)} and the "
            f"query still did not match what it returned. Each fix attempt is a "
            f"model round trip paid to learn something a tool call had already "
            f"said."
        )
    return note


def _empty(r: TurnReplayed, reference: Reference) -> str:
    note = (
        f"The query ran and came back with no rows. The answer to this question "
        f"has {len(reference.rows)}, so the data is there and the filters "
        f"removed it.\n\nThe SQL that ran:\n\n{_indent(r.sql)}\n\n{_sequence(r)}"
    )
    unsampled = _unsampled_filters(r)
    if unsampled:
        note += (
            f"\n\nAn empty result is not a small error. `sample_column` exists "
            f"to show what a column holds before a filter is written against "
            f"it, and it was not called on {_and_list(unsampled)}."
        )
    return note


# ------------------------------------------------------------------- the terms


def _correct(r: TurnReplayed, case: GoldenCase, m: Match) -> tuple[float, str]:
    if m.exact:
        return 1.0, ""

    parts = [_headline(m)]

    if m.missing or m.extra:
        if m.missing:
            parts.append(
                "  rows the answer has that did not come back:\n"
                + _rows(m.missing)
            )
        if m.extra:
            parts.append(
                "  rows that came back and do not belong:\n" + _rows(m.extra)
            )

    fixed = f", after {_plural(r.fix_attempts, 'fix attempt')}" if r.fix_attempts else ""
    parts.append(f"  the SQL that ran{fixed}:\n{_indent(r.sql, '      ')}")
    parts.append(_sequence(r, indent="  "))

    if case.reading:
        parts.append(f"  how this question is read here: {case.reading}")

    if m.arity[0] > m.arity[1] and m.projection is None:
        parts.append(
            "  a surplus column was tried and dropped, and the rows underneath "
            "it still did not match — so the extra column is not what is wrong."
        )

    return m.similarity, "\n\n".join(parts)


def _headline(m: Match) -> str:
    candidate_arity, reference_arity = m.arity
    if candidate_arity != reference_arity:
        return (
            f"The result has {_plural(candidate_arity, 'column')} where the answer has "
            f"{reference_arity}. Column names are yours to choose; how many "
            f"values a row carries is not."
        )
    if not m.order_ok:
        return (
            f"Every value is right and the order is not; the first difference is "
            f"at position {(m.first_disorder or 0) + 1}. The question names an "
            f"ordering, so the sequence is part of the answer rather than a "
            f"presentation choice."
        )
    if m.counts == (1, 1):
        return "The number is wrong."
    if m.counts[0] != m.counts[1]:
        return f"{_plural(m.counts[0], 'row')} came back; the answer has {m.counts[1]}."
    return f"{_plural(m.counts[0], 'row')} came back, and some of them are wrong."


def _cost(r: TurnReplayed, baseline: Baseline | None) -> tuple[float, str]:
    """One-sided. At or under the seed's tokens for this question earns 1.0.

    Unbounded reward for cheapness is the same failure as "fewer entries is
    better" in `metric_extract._shape`: the cheapest turn is the one that never
    explores. The correctness gate above already refuses that, and keeping this
    one-sided means the gate is not the only thing holding it.
    """
    if baseline is None or not baseline.tokens:
        return 1.0, ""
    over = (r.tokens - baseline.tokens) / baseline.tokens
    if over <= 0:
        return 1.0, ""
    return max(0.0, 1 - over), (
        f"{r.tokens} tokens ({r.tokens_in} in, {r.tokens_out} out) against the "
        f"seed's {baseline.tokens} on this question — {over:.0%} more for the "
        f"same answer, across {_plural(len(r.tools), 'tool call')} and "
        f"{_plural(r.fix_attempts, 'fix attempt')}:\n\n{_sequence(r)}"
    )


def _tool_calls(r: TurnReplayed, baseline: Baseline | None) -> tuple[float, str]:
    """Fewer calls, one-sided, plus the two shapes worth naming whatever the
    count did. A repeat and an error are facts about a description, and they are
    worth reading even on a turn that came in under the seed."""
    notes = []
    term = 1.0

    if baseline is not None and baseline.tool_calls:
        over = (len(r.tools) - baseline.tool_calls) / baseline.tool_calls
        if over > 0:
            term = max(0.0, 1 - over)
            notes.append(
                f"{len(r.tools)} tool calls against the seed's "
                f"{baseline.tool_calls} — {over:.0%} more:\n\n{_sequence(r)}"
            )

    repeats = _repeats(r.tools)
    if repeats:
        notes.append(
            "Called more than once with the same arguments:\n"
            + "\n".join(f"  {call}  x{n}" for call, n in repeats)
            + "\nThe second call asked a question the first had answered. A "
            "description that names what the tool returns is one the model does "
            "not have to call twice to find out."
        )

    errored = [t for t in r.tools if t.error]
    if errored:
        notes.append(
            f"{len(errored)} of {len(r.tools)} calls errored:\n"
            + "\n".join(f"  {_call(t)}" for t in errored)
            + "\nAn errored call is an argument the description did not explain. "
            "The loop recovered, and paid a model round trip to do it."
        )

    return term, "\n\n".join(notes)


# -------------------------------------------------------------------- renderers
#
# Deliberately absent from every one of these: `case.reference_sql`. The
# reflection model reads this text and rewrites a tool description from it, and
# a description carrying the corpus's queries scores well on the corpus and
# nowhere else. The rows a candidate got wrong are a diagnosis; the query that
# gets them right is the answer key. There is a test for this.


def _call(t: ToolCall) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in t.args.items())
    return f"{t.name}({args})" + ("   <- errored" if t.error else "")


def _sequence(r: TurnReplayed, indent: str = "") -> str:
    """Numbered, in order, errors marked.

    Order and repetition are the signal. "It called `sample_column` twice on
    `customer.region`" is a sentence about a tool description; "3 tool calls"
    is not.
    """
    if not r.tools:
        return (
            f"{indent}No tools were called — the turn wrote SQL without looking "
            f"at the schema first."
        )
    lines = "\n".join(f"{indent}  {i}. {_call(t)}" for i, t in enumerate(r.tools, 1))
    return f"{indent}It called {_plural(len(r.tools), 'tool')}, in order:\n{lines}"


def _repeats(tools: list[ToolCall]) -> list[tuple[str, int]]:
    """Identical calls, keyed on name and arguments, in the order first seen.

    `describe_table(customer)` at positions 2 and 7 with another table in
    between is the explore-loop thrash a better `list_tables` description is
    supposed to remove, which is the whole bet behind searching these.
    """
    counts: dict[tuple[str, str], int] = {}
    first: dict[tuple[str, str], ToolCall] = {}
    for t in tools:
        key = (t.name, json.dumps(t.args, sort_keys=True, default=str))
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, t)
    return [(_call(first[key]), n) for key, n in counts.items() if n > 1]


def _plural(n: int, noun: str) -> str:
    """Prose a person would write. "1 tool calls" is the tell that nobody read
    the output this metric produces, and its whole job is to be read."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _cell(value) -> str:
    if value is None:
        return "NULL"
    return repr(value) if isinstance(value, str) else str(value)


def _row(row: Row) -> str:
    if len(row) == 1:
        return _cell(row[0])
    return "(" + ", ".join(_cell(c) for c in row) + ")"


def _rows(rows: list[Row]) -> str:
    shown = "\n".join(f"      {_row(row)}" for row in rows[:SHOWN])
    if len(rows) > SHOWN:
        shown += f"\n      ... and {len(rows) - SHOWN} more"
    return shown


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _and_list(names: list[str]) -> str:
    quoted = [f"`{n}`" for n in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _described_tables(r: TurnReplayed) -> list[str]:
    """Tables the turn described and then wrote SQL against."""
    lowered = r.sql.lower()
    return sorted(
        {
            str(t.args.get("table"))
            for t in r.tools
            if t.name == "describe_table"
            and str(t.args.get("table", "")).lower() in lowered
        }
    )


def _unsampled_filters(r: TurnReplayed) -> list[str]:
    """Columns the SQL filters on that no `sample_column` call looked at.

    A substring scan, not a parse. It is feedback prose, and being approximately
    right about which column to look at is worth more than being exactly right
    about nothing.
    """
    sampled = {
        str(t.args.get("column", "")).lower()
        for t in r.tools
        if t.name == "sample_column"
    }
    where = r.sql.lower().partition(" where ")[2]
    if not where:
        return []
    words = {w.strip("(),;'\"").rsplit(".", 1)[-1] for w in where.split()}
    return sorted(
        word
        for word in words
        if word.isidentifier() and word not in sampled and word not in KEYWORDS
    )


KEYWORDS = frozenset(
    "and or not is null in like between select from group order by limit as on "
    "join where having distinct case when then else end".split()
)
