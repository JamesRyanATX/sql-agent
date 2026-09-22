"""One turn, and the verdict that follows it.

Its own module because two commands take turns — `ask` and `corpus` — and a
corpus recorded through a second code path would be a corpus of something else.
`main` cannot hold it: `main` imports the command modules at the bottom, so a
command importing `main` back is a cycle.

The same reasoning applies to the verdict. `ask` asks a person; `corpus`
compares against an answer somebody wrote down. Both go through `file_verdict`,
so what lands on a trace is one shape whichever produced it — a later reader
cannot tell them apart, and should not have to.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import click

from sql_agent import events, http, render

VERDICTS = ("OK", "Not OK")


@dataclass(frozen=True)
class Taken:
    """What one turn produced, for whoever has to decide what happens next.

    `rows` is here because `corpus` compares them against a written-down
    answer. They were always on the wire — `events.show` renders them — and
    were simply dropped on the floor.
    """

    answered: dict | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    fatal: bool = False


async def take(
    question: str,
    *,
    verbose: bool = False,
    as_json: bool = False,
    memory: bool = True,
) -> Taken:
    """Ask, rendering as it happens.

    `ask` exits 1 on a fatal turn; `corpus` notes it and moves to the next
    question. Both need the answer event, and only one needs the rows.
    """
    if not as_json:
        click.secho(question, fg="yellow", bold=True)

    fatal = False
    answered: dict | None = None
    rows: list[dict[str, Any]] = []
    # No session_id: the server mints one per turn, which is what a one-shot
    # question wants. Every turn reads the same memory regardless.
    async for ev in http.stream_events(
        "/ask", {"question": question, "memory": memory}
    ):
        if as_json:
            events.raw(ev)
        else:
            events.show(ev, verbose=verbose)
        fatal = fatal or bool(ev.get("type") == "error" and ev.get("fatal"))
        if ev.get("type") == "rows":
            # The last one wins: the fix loop can run a query more than once,
            # and what was compared has to be what was finally answered from.
            rows = ev.get("rows") or []
        if ev.get("type") == "answer":
            answered = ev
    return Taken(answered=answered, rows=rows, fatal=fatal)


def askable(answered: dict | None) -> bool:
    """Whether there is anything to ask about, and anywhere to put the answer.

    No `trace_id` means the turn ran with tracing off, so the verdict has
    nowhere to go and the question would be a keystroke spent on nothing.

    Both streams, not just stdin: the menu redraws with cursor movement, which
    `click.echo` does not strip from a redirected stdout. `sql-agent ask q >
    answer.txt` is a pipeline, not a conversation.
    """
    return bool(
        answered
        and answered.get("turn_id")
        and answered.get("trace_id")
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


async def judge(answered: dict) -> bool:
    """Ask what that answer was worth, and file it against the turn's trace.

    Returns the verdict, which `sql-agent corpus` counts: how many of twenty
    answers were right is the first thing anyone asks after a corpus run.

    The moment is the point. Asked here, the person still has the answer in
    front of them and is the one who wanted it; asked later, in another tool,
    against twenty traces, it is a chore nobody does. A recipe learned from a
    turn nobody judged is a guess the next turn inherits.
    """
    click.echo()
    correct = render.choose("Was that right?", VERDICTS) == 0
    comment = None
    if not correct:
        # Prose, not a menu of reasons: this text is read by the optimiser as
        # side information, and "wrong table" chosen from a list says less than
        # the sentence the person would have typed anyway.
        comment = click.prompt("What could be improved", default="", show_default=False)
    await file_verdict(answered, correct, comment)
    click.echo(render.dim("  thanks — filed on this turn's trace"))
    return correct


async def file_verdict(answered: dict, correct: bool, comment: str | None) -> None:
    """Put a verdict on a turn's trace.

    Split out because two things file one: a person choosing from the menu
    above, and `corpus` comparing against an answer somebody wrote down. They
    are indistinguishable once filed, which is the point — one shape, so a
    later reader cannot tell them apart and does not have to.
    """
    await http.post(
        f"/turns/{answered['turn_id']}/feedback",
        json={"correct": correct, "comment": comment or None},
    )
