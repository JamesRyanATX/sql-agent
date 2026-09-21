"""One turn, and the question that follows it.

Its own module because two commands take turns — `ask` and `corpus` — and a
corpus recorded through a second code path would be a corpus of something else.
`main` cannot hold it: `main` imports the command modules at the bottom, so a
command importing `main` back is a cycle.
"""

from __future__ import annotations

import sys

import click

from sql_agent import events, http, render

VERDICTS = ("OK", "Not OK")


async def take(
    question: str,
    *,
    verbose: bool = False,
    as_json: bool = False,
    memory: bool = True,
) -> tuple[dict | None, bool]:
    """Ask, rendering as it happens.

    Returns the answer event and whether the turn died, which is everything a
    caller needs to decide what happens next: `ask` exits 1, `corpus` notes it
    and moves to the next question.
    """
    if not as_json:
        click.secho(question, fg="yellow", bold=True)

    fatal = False
    answered: dict | None = None
    # No session_id: the server mints one per turn, which is what a one-shot
    # question wants. Every turn reads the same memory regardless.
    async for ev in http.stream_events(
        "/ask", {"question": question, "memory": memory}
    ):
        if as_json:
            events.raw(ev)
        else:
            events.show(ev, verbose=verbose)
        fatal = fatal or (ev.get("type") == "error" and ev.get("fatal"))
        if ev.get("type") == "answer":
            answered = ev
    return answered, bool(fatal)


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
    await http.post(
        f"/turns/{answered['turn_id']}/feedback",
        json={"correct": correct, "comment": comment or None},
    )
    click.echo(render.dim("  thanks — filed on this turn's trace"))
    return correct
