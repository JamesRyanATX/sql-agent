"""Being asked what the answer was worth, at the one moment somebody knows.

The menu itself is tested in test_cli_render.py. What is tested here is when it
appears at all, which is the harder half: a prompt that fires into a pipe hangs
`make demo`, and one that never fires leaves the corpus empty. Both failures are
silent, so the conditions are pinned one at a time.

No server and no model: `http.stream_events` is replaced by a scripted turn.
"""

from __future__ import annotations

import pytest

from sql_agent import main

ANSWER = {
    "type": "answer",
    "text": "1,840 active customers.",
    "tokens_in": 215,
    "tokens_out": 190,
    "total_tokens": 405,
    "latency_ms": 6600,
    "explored": True,
    "turn_id": 7,
    "trace_id": "0123456789abcdef" * 2,
}


@pytest.fixture
def turn(monkeypatch):
    """One scripted turn, and whatever the CLI posts afterwards."""
    posted: list[dict] = []

    def stream_events(path, payload):
        async def events():
            yield {"type": "rows", "count": 1, "rows": [{"n": 1840}], "capped": False}
            yield dict(ANSWER)
            yield {"type": "done"}

        return events()

    async def post(path, json=None, **kw):
        posted.append({"path": path, "json": json})
        return {}

    monkeypatch.setattr(main.http, "stream_events", stream_events)
    monkeypatch.setattr(main.http, "post", post)
    return posted


class Stream:
    """The real stream with a chosen answer to `isatty`.

    Wrapping rather than replacing, because `click.echo` still writes to
    whatever this stands in for, and pytest is still capturing it.
    """

    def __init__(self, real, tty: bool) -> None:
        self._real, self._tty = real, tty

    def isatty(self) -> bool:
        return self._tty

    def __getattr__(self, name):
        return getattr(self._real, name)


def streams(monkeypatch, *, stdin: bool, stdout: bool) -> None:
    """Called from the test body, never from a fixture.

    pytest's capture reinstalls `sys.stdout` between the setup phase and the
    call phase, so a stream patched in a fixture is silently replaced before the
    test runs — and these tests would then all assert the *unasked* path while
    reading as though they cover the asked one.
    """
    monkeypatch.setattr(main.sys, "stdin", Stream(main.sys.stdin, stdin))
    monkeypatch.setattr(main.sys, "stdout", Stream(main.sys.stdout, stdout))


@pytest.fixture
def answers(monkeypatch):
    """The person at the keyboard: which line they took, and what they typed."""
    said = {"choice": 0, "comment": "counted the cancelled orders"}
    monkeypatch.setattr(main.render, "choose", lambda *a, **kw: said["choice"])
    monkeypatch.setattr(main.click, "prompt", lambda *a, **kw: said["comment"])
    return said


async def ask(**kwargs) -> None:
    await main._ask("how many customers do we have?", "default", False, False, **kwargs)


async def test_an_approved_answer_files_a_verdict_and_no_prose(turn, answers, monkeypatch):
    """The common case is one keystroke. Nothing is asked about an answer that
    was right, because there is nothing a correct answer needs explained."""
    streams(monkeypatch, stdin=True, stdout=True)

    await ask()

    assert turn == [
        {
            "path": "/connections/default/turns/7/feedback",
            "json": {"correct": True, "comment": None},
        }
    ]


async def test_a_rejected_answer_carries_the_prose(turn, answers, monkeypatch):
    """This text is the side information a later optimisation reads, which is
    why it is asked for here and not inferred from the verdict."""
    streams(monkeypatch, stdin=True, stdout=True)
    answers["choice"] = 1

    await ask()

    assert turn[0]["json"] == {
        "correct": False,
        "comment": "counted the cancelled orders",
    }


async def test_an_empty_comment_is_no_comment(turn, answers, monkeypatch):
    """Enter on the prose prompt is a person declining to elaborate, not an
    empty string worth storing."""
    streams(monkeypatch, stdin=True, stdout=True)
    answers["choice"] = 1
    answers["comment"] = ""

    await ask()

    assert turn[0]["json"] == {"correct": False, "comment": None}


async def test_nothing_is_asked_when_stdin_is_not_a_terminal(turn, monkeypatch):
    """`echo | sql-agent ask` and every CI job. A prompt here waits forever on
    input that is not coming."""
    streams(monkeypatch, stdin=False, stdout=True)

    await ask()

    assert turn == []


async def test_nothing_is_asked_when_stdout_is_redirected(turn, monkeypatch):
    """`sql-agent ask q > answer.txt` is a pipeline, not a conversation — and
    the menu redraws with cursor movement, which click does not strip from a
    file."""
    streams(monkeypatch, stdin=True, stdout=False)

    await ask()

    assert turn == []


async def test_no_feedback_skips_it(turn, answers, monkeypatch):
    """What `make customer-count` passes, because VHS records through a pty and
    a pty is a terminal."""
    streams(monkeypatch, stdin=True, stdout=True)

    await ask(no_feedback=True)

    assert turn == []


async def test_a_turn_that_was_never_traced_is_not_asked_about(turn, answers, monkeypatch):
    """Tracing was off, so the verdict has nowhere to go. Asking anyway spends
    a keystroke to earn a 409."""
    streams(monkeypatch, stdin=True, stdout=True)

    def stream_events(path, payload):
        async def events():
            yield {**ANSWER, "trace_id": None}
            yield {"type": "done"}

        return events()

    monkeypatch.setattr(main.http, "stream_events", stream_events)

    await ask()

    assert turn == []


async def test_a_failed_turn_is_not_asked_about(turn, answers, monkeypatch):
    """There is no answer to judge, and the command exits 1."""
    streams(monkeypatch, stdin=True, stdout=True)

    def stream_events(path, payload):
        async def events():
            yield {"type": "error", "message": "ReadTimeout", "fatal": True}
            yield {"type": "done"}

        return events()

    monkeypatch.setattr(main.http, "stream_events", stream_events)

    with pytest.raises(SystemExit):
        await ask()

    assert turn == []


async def test_json_output_is_never_interrupted_by_a_menu(turn, answers, monkeypatch):
    """`--json` is one parseable object per line for another program to read,
    and that program cannot answer a question."""
    streams(monkeypatch, stdin=True, stdout=True)

    await main._ask("how many customers?", "default", False, True)

    assert turn == []
