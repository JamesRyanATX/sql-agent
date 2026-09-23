"""One turn, and the verdict `corpus` files after it.

Its own module because two commands take turns — `ask` and `corpus` — and a
corpus recorded through a second code path would be a corpus of something else.
`main` cannot hold it: `main` imports the command modules at the bottom, so a
command importing `main` back is a cycle.

`corpus` compares the rows against an answer somebody wrote down and files the
verdict through `file_verdict`, the same route the API offers anyone. `ask`
used to offer a two-line menu after every answer; it was removed because
nothing read what it filed, and a keystroke that changes nothing is a chore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import click

from sql_agent import events, http


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


async def file_verdict(answered: dict, correct: bool, comment: str | None) -> None:
    """Put a verdict on a turn's trace.

    `corpus` is the one caller now. The endpoint is what a person or another
    tool would use too, and what lands on the trace is one shape whoever
    produced it, so a later reader cannot tell them apart and does not have
    to.
    """
    await http.post(
        f"/turns/{answered['turn_id']}/feedback",
        json={"correct": correct, "comment": comment or None},
    )
