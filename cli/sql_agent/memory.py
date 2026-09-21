"""The three views of what the agent has learned: cache, turns, reset.

`cache` is the product — it is what the model reads on the next turn, in the
same order, tombstones included. `turns` is what each turn cost, which is the
demo. `reset` throws both away.
"""

from __future__ import annotations

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

    click.echo(
        click.style(
            f"{summary['total']} entries, {summary['verified']} verified", bold=True
        )
        + render.dim(
            f"  ({summary['stale']} stale, {summary['disabled']} disabled)"
        )
    )
    if not entries:
        click.echo("cache is empty")
        return

    for e in entries:
        marks = [click.style("✓", fg="green") if e["verified"] else " "]
        if e["origin"] == "human":
            marks.append(click.style("[human]", fg="green"))
        if e["pinned"]:
            marks.append(click.style("[pinned]", fg="green"))
        if e["tombstone"]:
            marks.append(click.style("[tombstone]", fg="yellow"))
        if e["stale"]:
            marks.append(click.style("[STALE]", fg="red"))

        name = click.style(e["name"] or "(unnamed)", bold=True)
        meta = render.dim(f"{e['kind']}  ×{e['hits']}")
        click.echo()
        click.echo(f"{' '.join(marks)} {name}  {meta}")
        click.echo(f"    {e['claim']}")
        if e["sql_fragment"]:
            click.secho(f"    SQL: {e['sql_fragment']}", fg="cyan")
        if e["tables"]:
            click.echo(render.dim(f"    tables: {', '.join(e['tables'])}"))


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
