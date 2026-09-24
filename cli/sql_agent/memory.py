"""The three views of what the agent has learned: cache, turns, reset.

`cache` is the product — it is what the model reads on the next turn, in the
same order, tombstones included. `turns` is what each turn cost, which is the
demo. `reset` throws both away.
"""

from __future__ import annotations

from typing import Any

import click

from sql_agent import config, http, render


@click.command()
@click.option(
    "--kind",
    type=click.Choice(["schema_fact", "recipe"]),
    help="Filter the listing. Never changes what the model would see.",
)
def cache(kind: str | None) -> None:
    """What the agent has learned about this database."""
    http.run(_cache(kind))


async def _cache(kind: str | None) -> None:
    body = await http.get(
        "/cache", params={"kind": kind} if kind else None
    )
    summary, entries = body["summary"], body["entries"]

    click.echo(click.style(_header(summary), bold=True) + _trouble(summary))
    if not entries:
        click.echo("cache is empty")
        return

    # Every line starts at column one. The blank line is what separates entries;
    # indenting the body under the name only pushed long claims into wrapping,
    # and a multi-line SQL fragment fell back to the margin anyway.
    for e in entries:
        click.echo()
        click.echo("  ".join(_title(e)))
        if e["kind"] == "recipe" and not e["verified"]:
            # Said only when it is so, and as where the SQL came from rather
            # than as a verdict. Three wordings of "verified" on every recipe
            # each needed a paragraph; a receipt reading "not stolen" on every
            # item. The check is `graph.grounded_in`: the fragment the model
            # typed into the note, against the query the agent executed. It
            # fails when the model writes the query it thinks should have run.
            click.secho(
                "this SQL was not part of any query the agent ran; "
                "it is the model's suggestion",
                fg="yellow",
            )
        click.echo(e["claim"])
        if e["sql_fragment"]:
            first, *rest = e["sql_fragment"].splitlines() or [""]
            click.secho(f"SQL: {first}", fg="cyan")
            for line in rest:
                # Under the first line of SQL, not under "SQL:", so a query
                # reads as the block it was written as.
                click.secho(f"     {line}", fg="cyan")
        if e["tables"]:
            click.echo(render.dim(f"tables: {', '.join(e['tables'])}"))


def _header(summary: dict[str, Any]) -> str:
    """The count, and nothing about the SQL check: "4 entries, 1 verified"
    read as three entries with something wrong, when three were schema facts
    the check does not apply to. The check speaks on the entry, when it
    failed."""
    noun = "entry" if summary["total"] == 1 else "entries"
    return f"{summary['total']} {noun}"


def _trouble(summary: dict[str, Any]) -> str:
    """Stale and disabled counts, only when there are any. Two zeros in every
    header is a line nobody reads, and then the one time it matters it is
    missed."""
    parts = []
    if summary["stale"]:
        parts.append(click.style(f"{summary['stale']} stale", fg="red"))
    if summary["disabled"]:
        parts.append(render.dim(f"{summary['disabled']} disabled"))
    return "  " + ", ".join(parts) if parts else ""


def _title(e: dict[str, Any]) -> list[str]:
    """Name, kind, how much it has been used, then the marks that apply."""
    parts = [click.style(e["name"] or "(unnamed)", bold=True), render.dim(e["kind"])]
    parts.append(render.dim(_usage(e["hits"])))
    if e["origin"] == "human":
        parts.append(click.style("[human]", fg="green"))
    if e["pinned"]:
        parts.append(click.style("[pinned]", fg="green"))
    if e["tombstone"]:
        parts.append(click.style("[tombstone]", fg="yellow"))
    if e["stale"]:
        parts.append(click.style("[STALE]", fg="red"))
    return parts


def _usage(hits: int) -> str:
    """`hits` is how many finished turns leaned on the entry."""
    if hits == 0:
        return "unused"
    return "used in 1 turn" if hits == 1 else f"used in {hits} turns"


@click.command()
@click.option("-n", "--limit", type=int, default=50, show_default=True)
@click.option(
    "--all", "show_all", is_flag=True, help="Include unfinished and failed turns."
)
def turns(limit: int, show_all: bool) -> None:
    """What every turn cost. The number that should be going down."""
    http.run(_turns(limit, show_all))


async def _turns(limit: int, show_all: bool) -> None:
    rows = (
        await http.get(
            "/turns",
            params={"limit": limit, "finished": str(not show_all).lower()},
        )
    )["turns"]
    if not rows:
        click.echo("no turns yet — ask a question")
        return
    # The cost column appears only when something reported one. Every backend
    # but OpenRouter charges somewhere this process cannot see, and a column of
    # blanks would suggest the turns were free.
    priced = any(t.get("cost") is not None for t in rows)
    headers = ["id", "question", "explored", "tools", "cached", "tokens", "secs"]
    if priced:
        headers.append("cost")

    render.table(
        headers,
        [
            [
                t["id"],
                t["question"][:38],
                "yes" if t["explored"] else "no",
                t["tool_calls"],
                t["cache_entries"],
                f"{t['tokens']:,}",
                f"{t['latency_ms'] / 1000:.0f}" if t["latency_ms"] else "",
                *([_dollars(t.get("cost"))] if priced else []),
            ]
            for t in rows
        ],
        right=frozenset({"id", "tools", "cached", "tokens", "secs", "cost"}),
        footer=_total(rows) if priced else None,
    )


def _dollars(cost: float | None) -> str:
    """Four decimal places: a cheap turn costs a fraction of a cent, and two
    would round every one of them to nothing."""
    return "" if cost is None else f"${float(cost):.4f}"


def _total(rows: list[dict]) -> str:
    """What the rows on screen came to. The number somebody paying for this
    wants, and the one nobody wants to add up by hand.

    Keeps the `(N rows)` shape: `demo/demo.tape` waits on the closing paren to
    know the command has finished printing, so a footer that drops it hangs the
    recording for forty minutes.
    """
    total = sum(float(t["cost"]) for t in rows if t.get("cost") is not None)
    plural = "" if len(rows) == 1 else "s"
    return f"({len(rows)} row{plural} — ${total:.4f})"


@click.command()
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation.")
def reset(yes: bool) -> None:
    """Forget everything the agent has learned.

    The database it queries is untouched — this is "forget what you learned",
    not "forget the database". Its address is in the environment and nothing
    here can reach it.
    """
    if not yes:
        click.confirm(
            "wipe everything learned — cache, turns and checkpoints?", abort=True
        )
    http.run(_reset())


async def _reset() -> None:
    wiped = (await http.delete("/cache"))["wiped"]
    if not any(wiped.values()):
        click.echo("nothing to wipe — the agent had learned nothing")
        return
    for table, rows in sorted(wiped.items()):
        if rows:
            click.echo(f"  wiped {table} ({rows} rows)")
    click.echo("reset complete")
