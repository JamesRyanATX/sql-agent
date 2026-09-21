"""Twenty questions, cold each time, judged as they land.

What matters here is not the asking — `turn.take` is the same code `ask` runs,
tested in test_cli_feedback.py. It is everything around it that can waste an
run: a cache left warm, a preflight that lets the run start when the
verdicts have nowhere to go, a summary that lies about what was judged.

Offline. `stream_events`, `delete` and `post` are scripted, and the menu answers
itself.
"""

from __future__ import annotations

import click
import pytest

from sql_agent import corpus, http
from sql_agent import turn as turn_mod
from tests.test_cli_feedback import ANSWER, Stream

QUESTIONS = """\
# The counts demo.sql pins
how many customers do we have?

# TRAP 2, the casing one
how many customers are in the west region?
"""


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    """A scripted run: what was called, in order, and what was posted."""
    calls: list[str] = []
    posted: list[dict] = []
    state = {"answer": dict(ANSWER), "tracing": True, "fatal": False}

    def stream_events(path, payload):
        async def events():
            calls.append(f"ask:{payload['question']}")
            if state["fatal"]:
                yield {"type": "error", "message": "ReadTimeout", "fatal": True}
            else:
                yield dict(state["answer"])
            yield {"type": "done"}

        return events()

    async def delete(path, **kw):
        calls.append(f"clear:{path}")
        return {}

    async def post(path, json=None, **kw):
        posted.append({"path": path, "json": json})
        return {}

    async def get(path, **kw):
        return {"config": {}, "overlay": None, "overridden": [],
                "tracing": state["tracing"]}

    monkeypatch.setattr(turn_mod.http, "stream_events", stream_events)
    monkeypatch.setattr(turn_mod.http, "post", post)
    monkeypatch.setattr(corpus.http, "delete", delete)
    monkeypatch.setattr(corpus.http, "get", get)
    monkeypatch.setattr(turn_mod.render, "choose", lambda *a, **kw: state.get("choice", 0))
    monkeypatch.setattr(turn_mod.click, "prompt", lambda *a, **kw: "wrong table")

    path = tmp_path / "questions.txt"
    path.write_text(QUESTIONS)
    return {"calls": calls, "posted": posted, "state": state, "path": path}


def streams(monkeypatch, *, stdin: bool = True, stdout: bool = True) -> None:
    """From the test body, never a fixture — see tests/test_cli_feedback.py."""
    monkeypatch.setattr(turn_mod.sys, "stdin", Stream(turn_mod.sys.stdin, stdin))
    monkeypatch.setattr(corpus.sys, "stdout", Stream(corpus.sys.stdout, stdout))
    monkeypatch.setattr(corpus.sys, "stdin", Stream(corpus.sys.stdin, stdin))
    monkeypatch.setattr(turn_mod.sys, "stdout", Stream(turn_mod.sys.stdout, stdout))


async def run(scripted, connection="golden") -> None:
    await corpus._record(scripted["path"], connection, False)


# ------------------------------------------------------------- reading the file


def test_the_file_is_questions_and_the_rest_is_for_people(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("# a heading\n\nhow many customers?\n   # indented note\nand west?\n")

    assert corpus.read_questions(path) == ["how many customers?", "and west?"]


async def test_a_file_of_nothing_but_comments_is_refused(scripted, monkeypatch):
    streams(monkeypatch)
    scripted["path"].write_text("# all notes\n\n# and no questions\n")

    with pytest.raises(http.ApiError, match="no questions"):
        await run(scripted)


# -------------------------------------------------------------------- the loop


async def test_every_question_is_asked_cold(scripted, monkeypatch):
    """The cache is cleared *before each* question, not once at the start.

    This is the whole reason the run is expensive: a warm second question
    describes an agent nobody starts from, and its `extract` call is not a
    distinct case.
    """
    streams(monkeypatch)

    await run(scripted)

    assert scripted["calls"] == [
        "clear:/connections/golden/cache",
        "ask:how many customers do we have?",
        "clear:/connections/golden/cache",
        "ask:how many customers are in the west region?",
    ]


async def test_each_answer_is_judged_and_filed(scripted, monkeypatch):
    streams(monkeypatch)

    await run(scripted)

    assert [p["path"] for p in scripted["posted"]] == [
        "/connections/golden/turns/7/feedback",
        "/connections/golden/turns/7/feedback",
    ]
    assert all(p["json"] == {"correct": True, "comment": None} for p in scripted["posted"])


async def test_the_summary_counts_what_was_judged(scripted, monkeypatch, capsys):
    """Two questions, one of them wrong. "How many were right" is the first
    thing anyone asks after a run."""
    streams(monkeypatch)
    scripted["state"]["choice"] = 1  # every answer marked Not OK

    await run(scripted)

    out = capsys.readouterr().out
    assert "2 of 2 turns judged, 0 of them right" in out


async def test_a_failed_turn_costs_that_question_and_not_the_run(
    scripted, monkeypatch, capsys
):
    """One timed-out turn out of twenty is a question to re-ask, not a reason to
    lose the other nineteen verdicts."""
    streams(monkeypatch)
    scripted["state"]["fatal"] = True

    await run(scripted)

    assert scripted["posted"] == []
    out = capsys.readouterr().out
    assert "moving on" in out
    assert "0 of 2 turns judged" in out


async def test_an_untraced_turn_stops_the_run(scripted, monkeypatch):
    """Tracing was on at preflight and is not now. Every later verdict would be
    filed nowhere, so the run stops rather than spending twenty turns to
    discover it."""
    streams(monkeypatch)
    scripted["state"]["answer"] = {**ANSWER, "trace_id": None}

    with pytest.raises(http.ApiError, match="not traced"):
        await run(scripted)


# --------------------------------------------------------------- the preflight


async def test_it_refuses_when_tracing_is_off(scripted, monkeypatch):
    """Found before the first turn, because the first turn is the expensive
    one and its verdict would have nowhere to land."""
    streams(monkeypatch)
    scripted["state"]["tracing"] = False

    with pytest.raises(http.ApiError, match="tracing is off"):
        await run(scripted)

    assert scripted["calls"] == []


async def test_it_refuses_without_a_terminal(scripted, monkeypatch):
    """`make corpus > log` and every CI job: nobody is there to answer, so the
    run would ask twenty questions into a pipe."""
    streams(monkeypatch, stdin=False)

    with pytest.raises(http.ApiError, match="somebody at the keyboard"):
        await run(scripted)

    assert scripted["calls"] == []


async def test_the_demo_connection_asks_first(scripted, monkeypatch, capsys):
    """`default` is what `sql-agent turns` charts. Twenty corpus turns in it
    buries the demo, and the cache clearing wipes what the demo taught it."""
    streams(monkeypatch)
    asked = {}

    def confirm(text, **kw):
        asked["text"] = text
        raise click.Abort()

    monkeypatch.setattr(corpus.click, "confirm", confirm)

    with pytest.raises(click.Abort):
        await run(scripted, connection="default")

    assert "Use it anyway?" in asked["text"]
    assert scripted["calls"] == []
