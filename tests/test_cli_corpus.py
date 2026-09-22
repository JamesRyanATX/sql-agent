"""Nineteen questions, cold each time, certified against a written answer.

What matters here is not the asking — `turn.take` is the same code `ask` runs,
tested in test_cli_feedback.py. It is everything around it that can waste a run:
a preflight that lets the run start when the verdicts have nowhere to go, a
comparison that calls a right answer wrong, a summary that lies about what was
scored.

The comparison is the part with the most room to be subtly wrong, and it is a
deliberate subset of `tools/gepa/resultset.compare` that the CLI may not import.
`test_the_two_comparisons_agree` is what stops the two drifting apart.

Offline. `stream_events`, `post` and `get` are scripted; nothing asks anybody
anything, which is the command's whole point.
"""

from __future__ import annotations

import json

import pytest

from sql_agent import corpus, http
from sql_agent import turn as turn_mod
from tests.test_cli_feedback import ANSWER, Stream

# Two cases: one with a written answer, one without. Enough to exercise both
# branches without a directory that takes a screenful to read.
COUNTED = {
    "name": "customers_active",
    "question": "how many customers do we have?",
    "reference_sql": ["SELECT count(*) FROM customer WHERE deleted_at IS NULL"],
    "tables": ["customer"],
    "expect": [[1840]],
    "ordered": False,
    "reading": "Customers that still exist.",
    "traps": ["soft deletes"],
    "format_version": 1,
}

MOVES = {
    "name": "revenue_total",
    "question": "what was our total revenue?",
    "reference_sql": ["SELECT sum(oi.qty * oi.price) FROM order_item oi"],
    "tables": ["order_item"],
    "expect": None,
    "ordered": False,
    "reading": "Line price at the time of sale.",
    "traps": ["historical price"],
    "format_version": 1,
}

RIGHT = [{"customers": 1840}]
WRONG = [{"customers": 2000}]


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    """A scripted run: what was asked, in order, and what was filed."""
    calls: list[str] = []
    asked: list[dict] = []
    posted: list[dict] = []
    state = {
        "answer": dict(ANSWER),
        "rows": RIGHT,
        "tracing": True,
        "fatal": False,
        "ceiling": 5.0,
    }

    def stream_events(path, payload):
        async def events():
            calls.append(f"ask:{payload['question']}")
            asked.append(payload)
            if state["fatal"]:
                yield {"type": "error", "message": "ReadTimeout", "fatal": True}
            else:
                yield {"type": "rows", "rows": state["rows"], "count": 1}
                yield dict(state["answer"])
            yield {"type": "done"}

        return events()

    async def post(path, json=None, **kw):
        posted.append({"path": path, "json": json})
        return {}

    async def get(path, **kw):
        return {
            "config": {
                "model": {"provider": "openrouter", "model": "google/gemini-2.5-flash"},
                "max_spend": state["ceiling"],
            },
            "overlay": None,
            "overridden": [],
            "tracing": state["tracing"],
        }

    monkeypatch.setattr(turn_mod.http, "stream_events", stream_events)
    monkeypatch.setattr(turn_mod.http, "post", post)
    monkeypatch.setattr(corpus.http, "get", get)

    directory = tmp_path / "golden"
    directory.mkdir()
    write(directory, "01_counted", COUNTED)
    write(directory, "02_moves", MOVES)
    return {
        "calls": calls,
        "asked": asked,
        "posted": posted,
        "state": state,
        "dir": directory,
    }


def write(directory, stem: str, case: dict) -> None:
    (directory / f"{stem}.json").write_text(json.dumps(case), encoding="utf-8")


async def run(scripted) -> None:
    await corpus._record(scripted["dir"], False)


# ------------------------------------------------------------ reading the cases


def test_the_cases_are_read_in_filename_order(scripted):
    cases = corpus.read_cases(scripted["dir"])

    assert [c.name for c in cases] == ["customers_active", "revenue_total"]
    assert cases[0].expect == [[1840]]
    assert cases[1].expect is None


async def test_a_directory_with_no_cases_is_refused(scripted, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(http.ApiError, match="no cases"):
        corpus.read_cases(empty)


async def test_a_bumped_format_version_is_an_error_not_a_misread(scripted):
    """The CLI cannot import `tools/gepa/golden.py`, so the two agree on the
    shape by assertion. A moved field would otherwise arrive as a question that
    is silently None and a verdict filed against nothing."""
    write(scripted["dir"], "03_future", {**COUNTED, "format_version": 2})

    with pytest.raises(http.ApiError, match="format_version"):
        corpus.read_cases(scripted["dir"])


# -------------------------------------------------------------------- the loop


async def test_every_question_is_asked_cold_and_the_memory_is_left_alone(scripted):
    """Cold is what makes the run expensive, and it is the point: a warm second
    question describes an agent nobody starts from, and its `extract` call is
    not a distinct case.

    It gets there with `memory: false` rather than by emptying the memory, which
    is what makes the run safe to start on the machine the demo runs on.
    """
    await run(scripted)

    assert scripted["calls"] == [
        "ask:how many customers do we have?",
        "ask:what was our total revenue?",
    ]
    assert [p["memory"] for p in scripted["asked"]] == [False, False]


async def test_a_matching_answer_is_filed_as_correct(scripted):
    await run(scripted)

    assert scripted["posted"] == [
        {"path": "/turns/7/feedback", "json": {"correct": True, "comment": None}}
    ]


async def test_a_differing_answer_is_filed_with_what_differed(scripted):
    """The comment is the point of a 0. A bare zero says something was wrong;
    this says what was expected, what came back, and which reading of the
    question the answer encodes — which is the likeliest reason a defensible
    answer was marked wrong."""
    scripted["state"]["rows"] = WRONG

    await run(scripted)

    filed = scripted["posted"][0]["json"]
    assert filed["correct"] is False
    assert "1840" in filed["comment"] and "2000" in filed["comment"]
    assert "Customers that still exist" in filed["comment"]


async def test_a_case_with_no_written_answer_is_asked_and_not_scored(scripted, capsys):
    """Its answer moves, so there is nothing to compare against. It is still
    asked, because the trace is what a later harvest reads."""
    await run(scripted)

    assert "ask:what was our total revenue?" in scripted["calls"]
    assert len(scripted["posted"]) == 1, "only the case with an answer is filed"
    assert "1 asked but not scored" in capsys.readouterr().out


async def test_nobody_is_asked_anything(scripted, monkeypatch):
    """The whole reason this exists beside `sql-agent ask`. A prompt here would
    hang `make corpus` in CI and nothing would say why."""
    monkeypatch.setattr(
        turn_mod.render, "choose", lambda *a, **kw: pytest.fail("it asked")
    )
    monkeypatch.setattr(
        corpus.click, "confirm", lambda *a, **kw: pytest.fail("it asked")
    )

    await run(scripted)

    assert scripted["posted"], "and it still filed a verdict"


async def test_it_runs_with_no_terminal_at_all(scripted, monkeypatch):
    """`make corpus > log` and every CI job. The old command refused here,
    because somebody had to answer it; this one has nobody to ask."""
    monkeypatch.setattr(turn_mod.sys, "stdin", Stream(turn_mod.sys.stdin, False))
    monkeypatch.setattr(turn_mod.sys, "stdout", Stream(turn_mod.sys.stdout, False))

    await run(scripted)

    assert len(scripted["calls"]) == 2


async def test_the_summary_counts_what_was_certified(scripted, capsys):
    scripted["state"]["rows"] = WRONG

    await run(scripted)

    assert "1 of 2 turns certified, 0 of them right" in capsys.readouterr().out


async def test_a_failed_turn_costs_that_question_and_not_the_run(scripted, capsys):
    """One timed-out turn out of nineteen is a question to re-ask, not a reason
    to lose the other eighteen verdicts."""
    scripted["state"]["fatal"] = True

    await run(scripted)

    assert scripted["posted"] == []
    out = capsys.readouterr().out
    assert "moving on" in out
    assert "0 of 2 turns certified" in out


async def test_an_untraced_turn_stops_the_run(scripted):
    """Tracing was on at preflight and is not now. Every later verdict would be
    filed nowhere, so the run stops rather than spending nineteen turns to
    discover it."""
    scripted["state"]["answer"] = {**ANSWER, "trace_id": None}

    with pytest.raises(http.ApiError, match="not traced"):
        await run(scripted)


# --------------------------------------------------------------- the preflight


async def test_it_refuses_when_tracing_is_off(scripted):
    """Found before the first turn, because the first turn is the expensive one
    and its verdict would have nowhere to land."""
    scripted["state"]["tracing"] = False

    with pytest.raises(http.ApiError, match="tracing is off"):
        await run(scripted)

    assert scripted["calls"] == []


async def test_it_says_what_it_is_about_to_spend(scripted, capsys):
    """Before the first turn, not after the nineteenth. It does not ask — there
    is no confirmation to give — but a run that spends money in silence is one
    nobody can budget for."""
    await run(scripted)

    out = capsys.readouterr().out
    assert "google/gemini-2.5-flash" in out
    assert "23,000 tokens" in out  # two cases, ~11.5k each, every one cold
    assert "$5.00" in out
    assert "1 carry a written answer" in out


async def test_it_stops_at_the_ceiling(scripted, capsys):
    """A run left going unattended is the case this exists for. It stops
    mid-list and says where rather than finishing the list."""
    scripted["state"]["ceiling"] = 0.05
    scripted["state"]["answer"] = {**ANSWER, "cost": 0.06}

    await run(scripted)

    assert len(scripted["calls"]) == 1
    out = capsys.readouterr().out
    assert "stopping at $0.06" in out
    assert "1 of 2 asked" in out


async def test_no_ceiling_is_said_out_loud(scripted, capsys):
    """`max_spend: 0` is a real choice, and the run should not imply a guard it
    does not have — the more so now that nobody is asked to confirm."""
    scripted["state"]["ceiling"] = 0

    await run(scripted)

    assert "no spend ceiling" in capsys.readouterr().out


async def test_what_it_spent_is_in_the_summary(scripted, capsys):
    scripted["state"]["answer"] = {**ANSWER, "cost": 0.0123}

    await run(scripted)

    assert "$0.0246 spent" in capsys.readouterr().out


# ------------------------------------------------------------- the comparison


# Every rule worth having an opinion about, asserted twice below: once for what
# it decides, and once against `resultset.compare`, which decides the same thing
# for the optimiser. The surplus-column pair is here because the two genuinely
# disagreed on it — a candidate that also returned the id was right to the
# metric and wrong to the CLI, which is the divergence this whole section is
# meant to catch.
RULES = [
    pytest.param([{"n": 1840}], [[1840]], False, True, id="a count"),
    pytest.param([{"n": 2000}], [[1840]], False, False, id="the wrong count"),
    pytest.param(
        [{"r": "West", "n": 500}], [["west", 500]], False, True, id="label casing"
    ),
    pytest.param(
        [{"r": " west ", "n": 500}], [["west", 500]], False, True, id="label spacing"
    ),
    pytest.param(
        [{"r": "east", "n": 1}, {"r": "west", "n": 2}],
        [["west", 2], ["east", 1]],
        False,
        True,
        id="unordered rows in any order",
    ),
    pytest.param(
        [{"r": "east", "n": 1}, {"r": "west", "n": 2}],
        [["west", 2], ["east", 1]],
        True,
        False,
        id="ordered rows in the wrong order",
    ),
    pytest.param([{"n": None}], [[None]], False, True, id="null"),
    pytest.param([], [[1840]], False, False, id="nothing came back"),
    pytest.param([{"n": 1}, {"n": 1}], [[1]], False, False, id="a duplicated row"),
    pytest.param(
        [{"id": 7, "r": "west", "n": 500}],
        [["west", 500]],
        False,
        True,
        id="it also returned the id",
    ),
    pytest.param(
        [{"total": 2000, "active": 1840}], [[1840]], False, False, id="two counts, one answer"
    ),
    pytest.param(
        [{"a": 1, "b": 2, "c": 3, "d": 4}], [[1, 2]], False, False, id="two columns surplus"
    ),
]


@pytest.mark.parametrize("rows, expect, ordered, same", RULES)
def test_the_comparison_rules(rows, expect, ordered, same):
    assert corpus.matches(rows, expect, ordered=ordered) is same


@pytest.mark.parametrize("rows, expect, ordered, same", RULES)
def test_the_two_comparisons_agree_on_every_rule(rows, expect, ordered, same):
    """The guard on a deliberate duplication.

    `corpus.matches` is a short subset of `tools/gepa/resultset.compare`, because
    the CLI may not import the optimiser. If the two disagree, a verdict on a
    trace and a score in a search stop meaning the same thing, and nothing else
    would ever say so — a search would simply optimise against a corpus labelled
    by different rules than the ones scoring it.
    """
    from tools.gepa import resultset

    assert resultset.compare(rows, expect, ordered=ordered).exact is same


def test_the_two_comparisons_agree_on_every_real_case():
    """The rules above are the ones I thought of. This is the corpus itself."""
    from tools.gepa import golden, resultset

    checked = 0
    for case in golden.load():
        if case.expect is None:
            continue
        checked += 1
        rows = [dict(enumerate(row)) for row in case.expect]

        assert corpus.matches(rows, case.expect, ordered=case.ordered) is True
        assert resultset.compare(rows, case.expect, ordered=case.ordered).exact is True

        # And the same answer with one cell moved must be wrong to both.
        broken = [list(row) for row in case.expect]
        broken[0][-1] = "nonsense-that-is-in-no-answer"
        mine = corpus.matches(
            [dict(enumerate(r)) for r in broken], case.expect, ordered=case.ordered
        )
        theirs = resultset.compare(broken, case.expect, ordered=case.ordered).exact
        assert mine is theirs is False, case.name

    assert checked == 9, "the nine cases with a written answer"
