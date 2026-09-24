"""The harvest: which recorded calls become a corpus, and which are dropped.

There were no tests here, which is how the first version shipped with a defect
that only appeared in use: it joined each recorded call to `turn.trace_id` to
find out which prompt had produced it, and `make reset` empties that table by
design. A reset turned every earlier recording into debris, data intact and
provenance gone.

So the first test below is the regression: a case whose turn row is gone must
still harvest. The rest are the drop reasons, each asserted to fire *and* not to
fire, because a filter that silently eats good cases and a filter that lets bad
ones through are equally invisible in a score.

No Langfuse and no database: `tracing.observations` is stubbed.
"""

from __future__ import annotations

import pytest

from app import graph, tracing
from tools.gepa import harvest
from tools.gepa.replay import NODE as REPLAY_NODE

SQL = "SELECT count(*) FROM customer WHERE deleted_at IS NULL"


def message(question: str = "how many customers do we have?", *, filed: str = "") -> str:
    return graph.extract_message(
        question=question,
        sql=SQL,
        findings="customer.deleted_at is the soft-delete flag",
        cache=[{"name": filed, "claim": "a filed claim"}] if filed else [],
    )


def generation(trace_id: str, *, name: str = "extract", content: str | None = None,
               tokens_out: int = 120, obs_id: str = "obs") -> dict:
    return {
        "id": obs_id,
        "trace_id": trace_id,
        "name": name,
        "start_time": None,
        "level": None,
        "input": {"system": "the prompt", "messages": [
            {"role": "user", "content": message() if content is None else content}
        ]},
        "output": {"text": "{}"},
        "metadata": {},
        "usage": {"output": tokens_out},
    }


def turn_span(trace_id: str, *, extract_fp: str = "abc12345") -> dict:
    return {
        "id": f"span-{trace_id}",
        "trace_id": trace_id,
        "name": "turn",
        "start_time": None,
        "level": None,
        "input": {"question": "q"},
        "output": None,
        "metadata": {"prompts": {"extract": extract_fp, "plan": "deadbeef"}},
        "usage": {},
    }


@pytest.fixture
def recorded(monkeypatch):
    """Stub the one function that reads Langfuse."""
    store: dict[str, list[dict]] = {"extract": [], "turn": []}

    def fake(*, name, kind="GENERATION", since=None, until=None, page=100):
        yield from store.get(name, [])

    monkeypatch.setattr(tracing, "observations", fake)
    return store


# ---------------------------------------------------- the reason this exists


def test_a_case_survives_the_loss_of_its_turn_row(recorded):
    """The regression. Nothing here consults Postgres, so `make reset` — which
    empties the `turn` table and leaves the trace store untouched — cannot
    orphan a recorded call any more."""
    recorded["extract"] = [generation("t1")]
    recorded["turn"] = [turn_span("t1")]

    result = harvest.extract_cases()

    assert len(result.cases) == 1
    assert result.cases[0].trace_id == "t1"
    assert result.cases[0].prompt_fp == "abc12345"


def test_each_case_carries_its_own_turn_spans_fingerprint(recorded):
    """Two calls recorded under two different `extract` prompts. The span is
    per-turn, so the fingerprints must not be shared between them."""
    recorded["extract"] = [
        generation("first", obs_id="a"),
        generation("second", obs_id="b", content=message("and the west region?")),
    ]
    recorded["turn"] = [
        turn_span("first", extract_fp="1111aaaa"),
        turn_span("second", extract_fp="2222bbbb"),
    ]

    result = harvest.extract_cases()

    assert {c.trace_id: c.prompt_fp for c in result.cases} == {
        "first": "1111aaaa",
        "second": "2222bbbb",
    }


def test_a_call_with_no_turn_span_is_dropped_rather_than_assumed(recorded):
    """No span means no fingerprint, and no fingerprint means round two cannot
    tell its own output from the prose it started with. Dropping it is how the
    corpus stays evidence about the prompt a human actually ran."""
    recorded["extract"] = [generation("orphan")]
    recorded["turn"] = []

    result = harvest.extract_cases()

    assert result.cases == []
    assert result.unscoped == 1
    assert "which prose produced it" in result.report()


def test_the_prompt_fingerprint_rides_along(recorded):
    """What keeps round two of an optimisation off round one's output."""
    recorded["extract"] = [generation("t1")]
    recorded["turn"] = [turn_span("t1", extract_fp="9f9f9f9f")]

    assert harvest.extract_cases().cases[0].prompt_fp == "9f9f9f9f"


def test_what_the_recorded_call_cost_rides_along_too(recorded):
    """The cost term scores a candidate against this. Losing it does not fail —
    a zero baseline reads as "do not score cost", so a fifth of the metric just
    stops existing. It was lost exactly that way in a rewrite, and every case in
    a real corpus came back with `base=0` before anyone noticed."""
    recorded["extract"] = [generation("t1", tokens_out=1387)]
    recorded["turn"] = [turn_span("t1")]

    assert harvest.extract_cases().cases[0].baseline_tokens_out == 1387


def test_a_call_langfuse_has_no_usage_for_still_harvests(recorded):
    """Zero is the honest answer, and the metric knows to skip cost for it."""
    missing = generation("t1")
    missing["usage"] = {}
    recorded["extract"] = [missing]
    recorded["turn"] = [turn_span("t1")]

    assert harvest.extract_cases().cases[0].baseline_tokens_out == 0


# ------------------------------------------------------------ the drop reasons


def test_the_harnesss_own_calls_never_become_corpus(recorded):
    recorded["extract"] = [generation("t1", name=REPLAY_NODE)]
    recorded["turn"] = [turn_span("t1")]

    result = harvest.extract_cases()

    assert result.cases == []
    assert result.contaminated == 1


def test_a_message_whose_sql_will_not_parse_is_dropped_and_counted(recorded):
    """Never silently: a corpus that quietly halved is a corpus whose scores
    are about a different population."""
    recorded["extract"] = [generation("t1", content="Question: q\n\nno anchor here")]
    recorded["turn"] = [turn_span("t1")]

    result = harvest.extract_cases()

    assert result.cases == []
    assert result.no_sql == 1
    assert "SQL would not parse" in result.report()


def test_an_input_with_no_user_turn_is_dropped(recorded):
    bad = generation("t1")
    bad["input"] = {"system": "s", "messages": [{"role": "assistant", "content": "x"}]}
    recorded["extract"] = [bad]
    recorded["turn"] = [turn_span("t1")]

    assert harvest.extract_cases().no_message == 1


def test_the_same_question_twice_is_kept_once(recorded):
    """A repeated question produces byte-identical extract inputs, and a corpus
    holding one case five times tunes the prompt for that case five times."""
    recorded["extract"] = [generation("t1", obs_id="a"), generation("t2", obs_id="b")]
    recorded["turn"] = [turn_span("t1"), turn_span("t2")]

    result = harvest.extract_cases()

    assert len(result.cases) == 1
    assert result.duplicate == 1


def test_different_questions_are_both_kept(recorded):
    """The other direction: dedupe must not eat a genuinely distinct case."""
    recorded["extract"] = [
        generation("t1", obs_id="a"),
        generation("t2", obs_id="b", content=message("revenue by region?")),
    ]
    recorded["turn"] = [turn_span("t1"), turn_span("t2")]

    assert len(harvest.extract_cases().cases) == 2


# ------------------------------------------------------------------ the report


def test_the_report_accounts_for_every_generation_it_saw(recorded):
    """Kept plus dropped must equal seen, or the numbers are decoration."""
    recorded["extract"] = [
        generation("keep", obs_id="a"),
        generation("same-again", obs_id="b"),
        generation("orphan", obs_id="c"),
        generation("harness", obs_id="d", name=REPLAY_NODE),
    ]
    recorded["turn"] = [
        turn_span("keep"),
        turn_span("same-again"),
        turn_span("harness"),
    ]

    r = harvest.extract_cases()
    dropped = (r.no_message + r.no_sql + r.unscoped
               + r.contaminated + r.duplicate)

    assert r.seen == 4
    assert len(r.cases) + dropped == r.seen


def test_a_clean_harvest_says_nothing_about_drops(recorded):
    recorded["extract"] = [generation("t1")]
    recorded["turn"] = [turn_span("t1")]

    assert "dropped" not in harvest.extract_cases().report()


def test_the_label_is_readable_and_unique(recorded):
    recorded["extract"] = [
        generation("aaaaaaaabbbb", obs_id="a"),
        generation("ccccccccdddd", obs_id="b", content=message("revenue by region?")),
    ]
    recorded["turn"] = [turn_span("aaaaaaaabbbb"), turn_span("ccccccccdddd")]

    names = [c.name for c in harvest.extract_cases().cases]

    assert names == ["aaaaaaaa-how-many-customers-do-we-have", "cccccccc-revenue-by-region"]


def test_tracing_off_yields_an_empty_harvest_rather_than_an_error(monkeypatch):
    """Off has to stay free, here as everywhere else."""
    monkeypatch.setattr(tracing, "client", lambda: None)
    result = harvest.extract_cases()
    assert result.cases == [] and result.seen == 0


# ------------------------------------------------------------- whole turns
#
# `turn_cases`: a turn is a case when its trace carries a reference query, or
# a verdict that the query it ran was right. Both arrive as scores, so the
# second stubbed function is `tracing.scores`, keyed by score name like the
# observations store is keyed by observation name.

RAN = "SELECT count(*) AS customers FROM customer"
GOLDEN = "SELECT count(*) AS customers FROM customer WHERE deleted_at IS NULL"


def finished_turn(trace_id: str, *, question: str = "how many customers do we have?",
                  sql: str | None = RAN, start: int = 1) -> dict:
    span = turn_span(trace_id)
    span["input"] = {"question": question}
    span["output"] = {"type": "answer", "text": "2,000 customers.", "sql": sql}
    span["start_time"] = start
    return span


def verdict(trace_id: str, value: float, *, comment: str | None = None, at: int = 1) -> dict:
    return {"trace_id": trace_id, "name": "correct", "value": value,
            "string_value": None, "comment": comment, "timestamp": at, "source": "API"}


def reference(trace_id: str, sql: str = GOLDEN, *, reading: str = "deleted customers are not customers",
              at: int = 1) -> dict:
    return {"trace_id": trace_id, "name": "reference", "value": None,
            "string_value": sql, "comment": reading, "timestamp": at, "source": "API"}


@pytest.fixture
def scored(recorded, monkeypatch):
    """The observations store, plus a scores store keyed by score name."""
    store: dict[str, list[dict]] = {"correct": [], "reference": []}

    def fake(*, name, since=None, page=100):
        yield from store.get(name, [])

    monkeypatch.setattr(tracing, "scores", fake)
    return recorded, store


def test_a_reference_score_makes_the_turn_a_case(scored):
    """The answer key's query, filed by `make corpus`, or a person's correction
    from the UI: the reference is what the turn *should* have run, whatever it
    did run, and the reading rides in the comment."""
    turns, scores = scored
    turns["turn"] = [finished_turn("t1")]
    scores["reference"] = [reference("t1")]

    result = harvest.turn_cases()

    assert len(result.cases) == 1
    case = result.cases[0]
    assert case.question == "how many customers do we have?"
    assert case.reference_sql == GOLDEN
    assert case.reading == "deleted customers are not customers"
    assert case.name.startswith("t1-how-many-customers")
    assert case.expect is None and case.tables == []


def test_a_correct_verdict_makes_the_turns_own_query_the_reference(scored):
    """Nobody wrote a reference, but somebody said the answer was right, so
    the query that produced it is one."""
    turns, scores = scored
    turns["turn"] = [finished_turn("t1", sql=GOLDEN)]
    scores["correct"] = [verdict("t1", 1.0)]

    result = harvest.turn_cases()

    assert [c.reference_sql for c in result.cases] == [GOLDEN]
    assert result.cases[0].reading == ""


def test_a_reference_wins_over_a_correct_turns_own_query(scored):
    turns, scores = scored
    turns["turn"] = [finished_turn("t1", sql=RAN)]
    scores["correct"] = [verdict("t1", 1.0)]
    scores["reference"] = [reference("t1", GOLDEN)]

    assert harvest.turn_cases().cases[0].reference_sql == GOLDEN


def test_a_turn_judged_wrong_with_no_reference_is_dropped_and_counted(scored):
    """There is nothing to score a candidate against. The report names it,
    because the fix is a person writing the query down."""
    turns, scores = scored
    turns["turn"] = [finished_turn("t1")]
    scores["correct"] = [verdict("t1", 0.0, comment="expected 1840, got 2000")]

    result = harvest.turn_cases()

    assert result.cases == []
    assert result.wrong_unreferenced == 1
    assert "nobody wrote what right is" in result.report()


def test_a_turn_judged_wrong_with_a_reference_is_a_case(scored):
    """The valuable kind: a question the seed gets wrong is where a candidate
    has room to win, and the reference says what winning looks like."""
    turns, scores = scored
    turns["turn"] = [finished_turn("t1")]
    scores["correct"] = [verdict("t1", 0.0)]
    scores["reference"] = [reference("t1")]

    assert len(harvest.turn_cases().cases) == 1


def test_an_unjudged_turn_is_dropped_and_counted(scored):
    turns, _ = scored
    turns["turn"] = [finished_turn("t1")]

    result = harvest.turn_cases()

    assert result.cases == [] and result.unjudged == 1
    assert "no verdict and no reference" in result.report()


def test_a_turn_that_never_ran_sql_is_dropped(scored):
    turns, scores = scored
    turns["turn"] = [finished_turn("t1", sql=None)]
    scores["reference"] = [reference("t1")]

    result = harvest.turn_cases()

    assert result.cases == [] and result.incomplete == 1


def test_a_rollout_leaves_no_turn_span_and_so_is_never_seen(scored):
    """The contamination guard, stated from this side: scores on a trace with
    no turn span are not a case, because a case is keyed on the span and
    `replay_turn` opens none."""
    turns, scores = scored
    turns["turn"] = []
    scores["correct"] = [verdict("rollout", 1.0)]

    assert harvest.turn_cases().seen == 0


def test_a_repeated_question_keeps_the_newest_turn(scored):
    """T1 asked five times is one case, weighted once. The newest, because
    that is the one under the prose on disk."""
    turns, scores = scored
    turns["turn"] = [
        finished_turn("old", sql="SELECT 1", start=1),
        finished_turn("new", sql="SELECT 2", start=2),
    ]
    scores["correct"] = [verdict("old", 1.0), verdict("new", 1.0)]

    result = harvest.turn_cases()

    assert [c.reference_sql for c in result.cases] == ["SELECT 2"]
    assert result.duplicate == 1


def test_a_repeated_question_prefers_the_turn_with_a_reference(scored):
    turns, scores = scored
    turns["turn"] = [
        finished_turn("referenced", sql=RAN, start=1),
        finished_turn("newer", sql="SELECT 2", start=2),
    ]
    scores["correct"] = [verdict("newer", 1.0)]
    scores["reference"] = [reference("referenced", GOLDEN)]

    assert [c.reference_sql for c in harvest.turn_cases().cases] == [GOLDEN]


def test_the_latest_verdict_on_a_trace_is_the_one_meant(scored):
    turns, scores = scored
    turns["turn"] = [finished_turn("t1", sql=GOLDEN)]
    scores["correct"] = [verdict("t1", 1.0, at=1), verdict("t1", 0.0, at=2)]

    assert harvest.turn_cases().wrong_unreferenced == 1


def test_order_matters_when_the_reference_orders_at_the_top_level(scored):
    turns, scores = scored
    turns["turn"] = [finished_turn("a", question="who signed up first?"),
                     finished_turn("b", question="how many in each region?")]
    scores["reference"] = [
        reference("a", "SELECT name FROM customer ORDER BY signed_up LIMIT 1"),
        reference("b", "SELECT region, count(*) FROM (SELECT * FROM customer ORDER BY id) c GROUP BY 1"),
    ]

    by_question = {c.question: c.ordered for c in harvest.turn_cases().cases}

    assert by_question == {"who signed up first?": True, "how many in each region?": False}


def test_the_turn_report_accounts_for_every_turn_it_saw(scored):
    turns, scores = scored
    turns["turn"] = [finished_turn("kept"), finished_turn("wrong", question="q2"),
                     finished_turn("unjudged", question="q3"), finished_turn("kept", start=0)]
    scores["correct"] = [verdict("kept", 1.0), verdict("wrong", 0.0)]

    result = harvest.turn_cases()

    dropped = result.incomplete + result.unjudged + result.wrong_unreferenced + result.duplicate
    assert result.seen == 4
    assert len(result.cases) + dropped == result.seen
    assert result.report().startswith("1 cases from 4 recorded turns")


def test_tracing_off_yields_an_empty_turn_harvest_rather_than_an_error(monkeypatch):
    monkeypatch.setattr(tracing, "client", lambda: None)
    result = harvest.turn_cases()
    assert result.cases == [] and result.seen == 0
