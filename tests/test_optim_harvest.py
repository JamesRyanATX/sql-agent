"""The harvest: which recorded calls become a corpus, and which are dropped.

There were no tests here, which is how the first version shipped with a defect
that only appeared in use: it joined each recorded call to `turn.trace_id` to
find out which warehouse it was about, and `make reset` empties that table by
design. A reset turned every earlier recording into debris — data intact,
ownership unprovable.

So the first test below is the regression: a case whose turn row is gone must
still harvest. The rest are the drop reasons, each asserted to fire *and* not to
fire, because a filter that silently eats good cases and a filter that lets bad
ones through are equally invisible in a score.

No Langfuse and no database: `tracing.observations` is stubbed.
"""

from __future__ import annotations

import pytest

from app import graph, tracing
from optim import harvest
from optim.replay import NODE as REPLAY_NODE

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


def turn_span(trace_id: str, *, connection_id: str = "default",
              extract_fp: str = "abc12345") -> dict:
    return {
        "id": f"span-{trace_id}",
        "trace_id": trace_id,
        "name": "turn",
        "start_time": None,
        "level": None,
        "input": {"question": "q", "connection_id": connection_id},
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

    result = harvest.extract_cases(connection_id="default")

    assert len(result.cases) == 1
    assert result.cases[0].trace_id == "t1"
    assert result.cases[0].connection_id == "default"


def test_the_scope_comes_from_the_turn_span(recorded):
    """One warehouse's recordings are not evidence about another's, and the
    turn span's own input is the only surviving record of which is which."""
    recorded["extract"] = [generation("mine", obs_id="a"), generation("theirs", obs_id="b")]
    recorded["turn"] = [
        turn_span("mine", connection_id="default"),
        turn_span("theirs", connection_id="warehouse-2"),
    ]

    result = harvest.extract_cases(connection_id="default")

    assert [c.trace_id for c in result.cases] == ["mine"]
    assert result.other_connection == 1


def test_a_call_with_no_turn_span_is_dropped_rather_than_assumed(recorded):
    """Unscoped is not the same as ours. Guessing here is how one customer's
    recordings tune the prompt used on another's warehouse."""
    recorded["extract"] = [generation("orphan")]
    recorded["turn"] = []

    result = harvest.extract_cases(connection_id="default")

    assert result.cases == []
    assert result.unscoped == 1
    assert "which warehouse" in result.report()


def test_the_prompt_fingerprint_rides_along(recorded):
    """What keeps round two of an optimisation off round one's output."""
    recorded["extract"] = [generation("t1")]
    recorded["turn"] = [turn_span("t1", extract_fp="9f9f9f9f")]

    assert harvest.extract_cases(connection_id="default").cases[0].prompt_fp == "9f9f9f9f"


def test_what_the_recorded_call_cost_rides_along_too(recorded):
    """The cost term scores a candidate against this. Losing it does not fail —
    a zero baseline reads as "do not score cost", so a fifth of the metric just
    stops existing. It was lost exactly that way in a rewrite, and every case in
    a real corpus came back with `base=0` before anyone noticed."""
    recorded["extract"] = [generation("t1", tokens_out=1387)]
    recorded["turn"] = [turn_span("t1")]

    assert harvest.extract_cases(connection_id="default").cases[0].baseline_tokens_out == 1387


def test_a_call_langfuse_has_no_usage_for_still_harvests(recorded):
    """Zero is the honest answer, and the metric knows to skip cost for it."""
    missing = generation("t1")
    missing["usage"] = {}
    recorded["extract"] = [missing]
    recorded["turn"] = [turn_span("t1")]

    assert harvest.extract_cases(connection_id="default").cases[0].baseline_tokens_out == 0


# ------------------------------------------------------------ the drop reasons


def test_the_harnesss_own_calls_never_become_corpus(recorded):
    recorded["extract"] = [generation("t1", name=REPLAY_NODE)]
    recorded["turn"] = [turn_span("t1")]

    result = harvest.extract_cases(connection_id="default")

    assert result.cases == []
    assert result.contaminated == 1


def test_a_message_whose_sql_will_not_parse_is_dropped_and_counted(recorded):
    """Never silently: a corpus that quietly halved is a corpus whose scores
    are about a different population."""
    recorded["extract"] = [generation("t1", content="Question: q\n\nno anchor here")]
    recorded["turn"] = [turn_span("t1")]

    result = harvest.extract_cases(connection_id="default")

    assert result.cases == []
    assert result.no_sql == 1
    assert "SQL would not parse" in result.report()


def test_an_input_with_no_user_turn_is_dropped(recorded):
    bad = generation("t1")
    bad["input"] = {"system": "s", "messages": [{"role": "assistant", "content": "x"}]}
    recorded["extract"] = [bad]
    recorded["turn"] = [turn_span("t1")]

    assert harvest.extract_cases(connection_id="default").no_message == 1


def test_the_same_question_twice_is_kept_once(recorded):
    """A repeated question produces byte-identical extract inputs, and a corpus
    holding one case five times tunes the prompt for that case five times."""
    recorded["extract"] = [generation("t1", obs_id="a"), generation("t2", obs_id="b")]
    recorded["turn"] = [turn_span("t1"), turn_span("t2")]

    result = harvest.extract_cases(connection_id="default")

    assert len(result.cases) == 1
    assert result.duplicate == 1


def test_different_questions_are_both_kept(recorded):
    """The other direction: dedupe must not eat a genuinely distinct case."""
    recorded["extract"] = [
        generation("t1", obs_id="a"),
        generation("t2", obs_id="b", content=message("revenue by region?")),
    ]
    recorded["turn"] = [turn_span("t1"), turn_span("t2")]

    assert len(harvest.extract_cases(connection_id="default").cases) == 2


# ------------------------------------------------------------------ the report


def test_the_report_accounts_for_every_generation_it_saw(recorded):
    """Kept plus dropped must equal seen, or the numbers are decoration."""
    recorded["extract"] = [
        generation("keep", obs_id="a"),
        generation("other", obs_id="b"),
        generation("orphan", obs_id="c"),
        generation("mine", obs_id="d", name=REPLAY_NODE),
    ]
    recorded["turn"] = [
        turn_span("keep"),
        turn_span("other", connection_id="warehouse-2"),
        turn_span("mine"),
    ]

    r = harvest.extract_cases(connection_id="default")
    dropped = (r.no_message + r.no_sql + r.unscoped
               + r.other_connection + r.contaminated + r.duplicate)

    assert r.seen == 4
    assert len(r.cases) + dropped == r.seen


def test_a_clean_harvest_says_nothing_about_drops(recorded):
    recorded["extract"] = [generation("t1")]
    recorded["turn"] = [turn_span("t1")]

    assert "dropped" not in harvest.extract_cases(connection_id="default").report()


def test_the_label_is_readable_and_unique(recorded):
    recorded["extract"] = [
        generation("aaaaaaaabbbb", obs_id="a"),
        generation("ccccccccdddd", obs_id="b", content=message("revenue by region?")),
    ]
    recorded["turn"] = [turn_span("aaaaaaaabbbb"), turn_span("ccccccccdddd")]

    names = [c.name for c in harvest.extract_cases(connection_id="default").cases]

    assert names == ["aaaaaaaa-how-many-customers-do-we-have", "cccccccc-revenue-by-region"]


def test_tracing_off_yields_an_empty_harvest_rather_than_an_error(monkeypatch):
    """Off has to stay free, here as everywhere else."""
    monkeypatch.setattr(tracing, "client", lambda: None)
    result = harvest.extract_cases(connection_id="default")
    assert result.cases == [] and result.seen == 0
