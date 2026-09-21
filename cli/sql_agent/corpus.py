"""Record a corpus: ask each question cold, and say what the answer was worth.

The optimisation downstream needs turns with a label on them, and the only
person who can supply one is the person who just read the answer. So this asks
a file of questions, one turn each, and waits for you to judge each answer.

**Cold every turn.** The cache is cleared before each question, so no answer
leans on what the last one learned. That is what makes every turn a full
exploration — a distinct `extract` call for the prompt corpus, and a token count
that means something as a baseline. It also makes the run expensive on purpose:
a warm corpus would be cheaper and would describe an agent nobody starts from.

Everything that could waste the run is checked before the first question:
a terminal to answer on, tracing to record onto, and a connection that exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from sql_agent import config, http, render, turn

# Kept off `default`, which is the demo's own warehouse: `sql-agent turns` reads
# that connection's log, and twenty corpus turns in it is the demo chart with
# the demo buried in the middle.
DEMO_CONNECTION = "default"

# What one cold turn costs in tokens, measured (README's T1). Used only to say
# the order of magnitude before a run: a real figure would need a price per
# model, and the run reports what it actually spent as it goes.
COLD_TURN_TOKENS = 11_500


@click.command("corpus")
@click.argument("questions", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@config.option
@click.option("-v", "--verbose", is_flag=True, help="Show planning, exploration and what was learned.")
def record(questions: Path, connection: str | None, verbose: bool) -> None:
    """Ask every question in a file, cold, and judge each answer.

    \b
      make corpus                          the demo's questions, on `golden`
      sql-agent corpus demo/questions.txt  the same thing, spelled out

    One question per line. Blank lines and `#` comments are skipped, so the file
    can say what each question is for.

    Every verdict is filed on that turn's trace, which is what a later harvest
    reads as a label. Nothing is written here.
    """
    http.run(_record(questions, connection, verbose))


def read_questions(path: Path) -> list[str]:
    """The file, minus the parts written for people.

    A question is a whole line: no splitting, no quoting, nothing to escape.
    Anything else would be a format, and a format is a thing to get wrong on the
    morning of a talk.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


async def _record(path: Path, connection: str | None, verbose: bool) -> None:
    cid = config.connection(connection)
    asked = judged = approved = 0
    spent = 0.0

    questions = read_questions(path)
    if not questions:
        raise http.ApiError(f"{path} holds no questions — every line is blank or a comment")

    ceiling = await _preflight(cid, path, questions)

    try:
        for n, question in enumerate(questions, start=1):
            click.echo(render.dim(f"\n[{n}/{len(questions)}] clearing the cache"))
            await http.delete(f"/connections/{cid}/cache")

            answered, fatal = await turn.take(cid, question, verbose=verbose)
            asked += 1
            if fatal or not answered:
                # Not fatal to the whole run: one timed-out turn out of twenty is
                # a question to re-ask later, not a reason to lose the other
                # nineteen verdicts.
                click.secho("  that turn failed — moving on", fg="yellow")
                continue
            if not answered.get("trace_id"):
                raise http.ApiError(
                    "that turn was not traced, so a verdict would have nowhere "
                    "to go — check `sql-agent config`"
                )

            approved += await turn.judge(cid, answered)
            judged += 1

            spent += answered.get("cost") or 0.0
            if ceiling and spent >= ceiling:
                click.secho(
                    f"\nstopping at ${spent:.2f}, the ceiling in config.yaml "
                    f"(max_spend: {ceiling}). {n} of {len(questions)} asked.",
                    fg="yellow",
                )
                break
    except (KeyboardInterrupt, click.Abort):
        # Twenty questions is long enough that it will be interrupted, and the
        # operator needs to know where it stopped rather than guessing.
        click.echo()
        click.secho("stopped early", fg="yellow")
    finally:
        _summary(asked, judged, approved, spent, cid)


async def _preflight(cid: str, path: Path, questions: list[str]) -> float:
    """Everything that would waste the run, checked before the first turn.
    Returns the spend ceiling, or 0 where there is none.

    Each of these is otherwise discovered after a cold turn has been paid for,
    the trace one only at the end when the verdicts turn out to be on nothing,
    and the money one when the statement arrives.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise http.ApiError(
            "`corpus` needs somebody at the keyboard: it asks what each answer "
            "was worth, and there is nobody at a terminal to answer"
        )

    body = await http.get("/config")
    if not body["tracing"]:
        raise http.ApiError(
            "tracing is off, so the verdicts would have nowhere to land — set "
            "both Langfuse keys (make langfuse-up) and restart the server"
        )

    click.echo(
        f"{len(questions)} questions from {path}, against {cid}, "
        f"cold each time. Answer each one as it lands."
    )

    model = body["config"]["model"]
    ceiling = float(body["config"].get("max_spend") or 0)
    click.echo(render.dim(f"  model: {model['model']} via {model['provider']}"))
    click.echo(
        render.dim(
            f"  roughly {len(questions) * COLD_TURN_TOKENS:,} tokens — a cold "
            f"turn is about {COLD_TURN_TOKENS:,}, and every one of these is cold"
        )
    )
    if ceiling:
        click.echo(render.dim(f"  stopping at ${ceiling:.2f} (max_spend)"))
    else:
        click.secho(
            "  no spend ceiling — max_spend is 0 in config.yaml", fg="yellow"
        )

    if cid == DEMO_CONNECTION:
        click.secho(
            f"\n{cid!r} is the connection the demo reads — its turn log is the "
            "chart, and its cache will be cleared before every question.",
            fg="yellow",
        )
        click.confirm("Use it anyway?", abort=True)

    click.confirm("Start?", default=True, abort=True)
    return ceiling


def _summary(asked: int, judged: int, approved: int, spent: float, cid: str) -> None:
    click.echo()
    click.echo(
        render.bold(f"{judged} of {asked} turns judged, {approved} of them right")
    )
    if spent:
        click.echo(render.bold(f"${spent:.4f} spent"))
    click.echo(
        render.dim(
            f"  the verdicts are on the traces; `sql-agent turns -c {cid}` is "
            "what they cost"
        )
    )
