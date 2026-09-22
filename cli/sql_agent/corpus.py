"""Record a corpus: ask each question cold, and file the verdict for it.

The optimisation downstream needs turns with a label on them. A label is
somebody saying whether the answer was right, and for the demo's questions
somebody already has: `demo/golden/` holds a reference query per question and,
where `demo/demo.sql` fixes the number, the answer written down beside it. So
this asks each question and compares what came back, which means it needs
nobody at a keyboard.

**A certified verdict, not a human one.** A person wrote the reference once and
a machine applies it. That distinction matters and is not hedging: what lands on
the trace is identical to what `sql-agent ask` files when you choose from the
menu, deliberately, so a later reader cannot tell them apart. Judging by hand is
still `sql-agent ask` — this command never asks you anything.

**Cold every turn, and nothing kept.** Every question is asked with the memory
off: it ignores what the agent has learned and saves nothing it learns. So each
turn is a full exploration — a distinct `extract` call for the prompt corpus,
and a token count that means something as a baseline — and your memory is
exactly as it was when the run finishes.

**Nine of the nineteen get a verdict.** The rest carry no written-down answer
because theirs moves: every one is revenue, which depends on how old an order
is, or a date window that slides. Writing those numbers down would be a gate
that passes today and fails in November. They are asked anyway, because their
traces are what a later harvest reads, and they are reported as unscored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from sql_agent import http, render, turn

# What one cold turn costs in tokens, measured (README's T1). Used only to say
# the order of magnitude before a run: a real figure would need a price per
# model, and the run reports what it actually spent as it goes.
COLD_TURN_TOKENS = 11_500

# Bumped by hand in `tools/gepa/golden.py` when the case fields move. Checked
# here so a changed format is an error rather than a silent misread — this
# command cannot import that module (the CLI holds no server code), so the two
# agree by assertion and a test, not by sharing a constant.
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Case:
    """The parts of a golden case this command needs.

    Four fields of the seven on disk. `reference_sql` and `tables` are for the
    optimiser, which can run a query; this command only compares against an
    answer already written down.
    """

    name: str
    question: str
    expect: list[list[Any]] | None
    ordered: bool
    reading: str


@click.command("corpus")
@click.argument(
    "cases", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("-v", "--verbose", is_flag=True, help="Show planning, exploration and what was learned.")
def record(cases: Path, verbose: bool) -> None:
    """Ask every question in a case directory, cold, and file each verdict.

    \b
      make corpus                  the demo's questions
      sql-agent corpus demo/golden the same thing, spelled out

    One JSON file per case, holding the question and — where the answer is
    fixed — what it should be. Nothing is asked of you: the verdict comes from
    comparing the rows against that answer. To judge one by hand, use
    `sql-agent ask`.

    Asked with the memory off, so the run leaves what the agent has learned
    exactly as it found it. Every verdict is filed on that turn's trace.
    """
    http.run(_record(cases, verbose))


def read_cases(directory: Path) -> list[Case]:
    """Every case in the directory, in filename order.

    The version is checked before anything is read out of the row, because a
    moved field otherwise shows up as a question that is silently `None`.
    """
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise http.ApiError(f"{directory} holds no cases — no .json files in it")

    cases = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("format_version") != FORMAT_VERSION:
            raise http.ApiError(
                f"{path} is format_version {row.get('format_version')} and this "
                f"command reads {FORMAT_VERSION} — the case fields moved, so "
                f"`sql-agent corpus` needs updating alongside them"
            )
        cases.append(
            Case(
                name=row["name"],
                question=row["question"],
                expect=row.get("expect"),
                ordered=bool(row.get("ordered")),
                reading=row.get("reading") or "",
            )
        )
    return cases


# ------------------------------------------------------------ the comparison
#
# A deliberate subset of `tools/gepa/resultset.compare`, which this command may
# not import — the CLI holds no server code and no development tooling. It gets
# away with being this short because every `expect` value is an exact integer, a
# short label or null, which JSON round-trips exactly; there are no decimals and
# no dates to reconcile.
#
# The risk is the two drifting into disagreeing about what counts as right.
# tests/test_cli_corpus.py checks them against each other on every golden case,
# because a comment cannot.


def _cells(row: Any) -> tuple:
    """One row as bare values. Column names are the model's to choose."""
    values = row.values() if isinstance(row, dict) else row
    return tuple(
        v.strip().casefold() if isinstance(v, str) else v for v in values
    )


def matches(rows: list[dict], expect: list[list[Any]], *, ordered: bool) -> bool:
    """Whether what came back is the answer.

    A multiset unless the question named an ordering, so duplicate rows count
    and row order is not invented as a requirement where none was stated.

    One surplus column is allowed against an answer of two or more, which is
    the model returning the id beside the name. Never against a one-column
    answer: `SELECT count(*), count(*) FILTER (...)` would otherwise match
    1,840 on its second column while the answer above it says 2,000. Both rules
    are `resultset.compare`'s, and the test holds the two to each other.
    """
    want = [_cells(r) for r in expect]
    got = [_cells(r) for r in rows]
    if _same(got, want, ordered=ordered):
        return True

    width = len(want[0]) if want else 0
    if width >= 2 and got and len(got[0]) == width + 1:
        return any(
            _same([_drop(row, i) for row in got], want, ordered=ordered)
            for i in range(width + 1)
        )
    return False


def _drop(row: tuple, i: int) -> tuple:
    return row[:i] + row[i + 1 :]


def _same(got: list[tuple], want: list[tuple], *, ordered: bool) -> bool:
    if ordered:
        return got == want
    return sorted(got, key=repr) == sorted(want, key=repr)


def _difference(case: Case, rows: list[dict]) -> str:
    """What to put on the trace when the answer was wrong.

    The rows both ways, and the reading — which is the sentence saying how an
    ambiguous question was interpreted here, and therefore the most likely
    reason a defensible answer was marked wrong.
    """
    got = [list(_cells(r)) for r in rows]
    note = (
        f"Certified against demo/golden/: expected {case.expect}, "
        f"the query returned {got}."
    )
    if case.reading:
        note += f"\n\nHow this question is read here: {case.reading}"
    return note


# ------------------------------------------------------------------- the run


async def _record(directory: Path, verbose: bool) -> None:
    asked = judged = approved = unscored = 0
    spent = 0.0

    cases = read_cases(directory)
    ceiling = await _preflight(directory, cases)

    try:
        for n, case in enumerate(cases, start=1):
            click.echo(render.dim(f"\n[{n}/{len(cases)}] {case.name}"))
            taken = await turn.take(case.question, verbose=verbose, memory=False)
            asked += 1

            if taken.fatal or not taken.answered:
                # Not fatal to the whole run: one timed-out turn out of twenty
                # is a question to re-ask later, not a reason to lose the other
                # nineteen verdicts.
                click.secho("  that turn failed — moving on", fg="yellow")
                continue
            if not taken.answered.get("trace_id"):
                raise http.ApiError(
                    "that turn was not traced, so a verdict would have nowhere "
                    "to go — check `sql-agent config`"
                )

            spent += taken.answered.get("cost") or 0.0

            if case.expect is None:
                unscored += 1
                click.echo(render.dim("  no written answer — asked, not scored"))
            else:
                correct = matches(taken.rows, case.expect, ordered=case.ordered)
                comment = None if correct else _difference(case, taken.rows)
                await turn.file_verdict(taken.answered, correct, comment)
                judged += 1
                approved += correct
                if correct:
                    click.secho("  correct — filed on this turn's trace", fg="green")
                else:
                    click.secho("  not the answer — filed, with what differed", fg="yellow")

            if ceiling and spent >= ceiling:
                click.secho(
                    f"\nstopping at ${spent:.2f}, the ceiling in config.yaml "
                    f"(max_spend: {ceiling}). {n} of {len(cases)} asked.",
                    fg="yellow",
                )
                break
    except (KeyboardInterrupt, click.Abort):
        # Nineteen turns is long enough that it will be interrupted, and the
        # operator needs to know where it stopped rather than guessing.
        click.echo()
        click.secho("stopped early", fg="yellow")
    finally:
        _summary(asked, judged, approved, unscored, spent)


async def _preflight(directory: Path, cases: list[Case]) -> float:
    """Everything that would waste the run, checked before the first turn.
    Returns the spend ceiling, or 0 where there is none.

    No terminal check and no confirmation: this command asks nobody anything,
    which is the whole reason it exists beside `sql-agent ask`. What is left is
    the one thing that would waste every turn — tracing off, so every verdict
    lands nowhere — and saying what the run is likely to cost.
    """
    body = await http.get("/config")
    if not body["tracing"]:
        raise http.ApiError(
            "tracing is off, so the verdicts would have nowhere to land — set "
            "both Langfuse keys (make langfuse-up) and restart the server"
        )

    scorable = sum(1 for c in cases if c.expect is not None)
    click.echo(
        f"{len(cases)} cases from {directory}, cold each time and none of it "
        f"kept. {scorable} carry a written answer and will be scored."
    )

    model = body["config"]["model"]
    ceiling = float(body["config"].get("max_spend") or 0)
    click.echo(render.dim(f"  model: {model['model']} via {model['provider']}"))
    click.echo(
        render.dim(
            f"  roughly {len(cases) * COLD_TURN_TOKENS:,} tokens — a cold "
            f"turn is about {COLD_TURN_TOKENS:,}, and every one of these is cold"
        )
    )
    if ceiling:
        click.echo(render.dim(f"  stopping at ${ceiling:.2f} (max_spend)"))
    else:
        click.secho(
            "  no spend ceiling — max_spend is 0 in config.yaml", fg="yellow"
        )
    return ceiling


def _summary(
    asked: int, judged: int, approved: int, unscored: int, spent: float
) -> None:
    click.echo()
    click.echo(
        render.bold(f"{judged} of {asked} turns certified, {approved} of them right")
    )
    if unscored:
        click.echo(
            render.dim(
                f"  {unscored} asked but not scored — no written answer, "
                f"because revenue and date windows move"
            )
        )
    if spent:
        click.echo(render.bold(f"${spent:.4f} spent"))
    click.echo(
        render.dim(
            "  the verdicts are on the traces; `sql-agent turns` is what they cost"
        )
    )
