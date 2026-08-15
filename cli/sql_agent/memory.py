"""The three views of what the agent has learned: cache, turns, reset.

`cache` is the product — it is what the model reads on the next turn, in the
same order, tombstones included. `turns` is what each turn cost, which is the
demo. `reset` throws both away for one connection.
"""

from __future__ import annotations

import click

from sql_agent import config, http, render


@click.command()
@config.option
@click.option(
    "--kind",
    type=click.Choice(["schema_fact", "recipe"]),
    help="Filter the listing. Never changes what the model would see.",
)
def cache(connection: str | None, kind: str | None) -> None:
    """What the agent has learned about this database."""
    http.run(_cache(connection, kind))


async def _cache(connection: str | None, kind: str | None) -> None:
    cid = config.connection(connection)
    body = await http.get(
        f"/connections/{cid}/cache", params={"kind": kind} if kind else None
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
@config.option
@click.option("-n", "--limit", type=int, default=50, show_default=True)
@click.option(
    "--all", "show_all", is_flag=True, help="Include unfinished and failed turns."
)
def turns(connection: str | None, limit: int, show_all: bool) -> None:
    """What every turn cost. The number that should be going down."""
    http.run(_turns(connection, limit, show_all))


async def _turns(connection: str | None, limit: int, show_all: bool) -> None:
    cid = config.connection(connection)
    rows = (
        await http.get(
            f"/connections/{cid}/turns",
            params={"limit": limit, "finished": str(not show_all).lower()},
        )
    )["turns"]
    if not rows:
        click.echo("no turns yet — ask a question")
        return
    render.table(
        ["id", "question", "explored", "tools", "cached", "tokens", "secs"],
        [
            [
                t["id"],
                t["question"][:38],
                "yes" if t["explored"] else "no",
                t["tool_calls"],
                t["cache_entries"],
                f"{t['tokens']:,}",
                f"{t['latency_ms'] / 1000:.0f}" if t["latency_ms"] else "",
            ]
            for t in rows
        ],
        right=frozenset({"id", "tools", "cached", "tokens", "secs"}),
    )


@click.command()
@config.option
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation.")
def reset(connection: str | None, yes: bool) -> None:
    """Forget everything learned about this database.

    The connection itself stays registered — this is "forget what you learned",
    not "forget the database exists". For that, `connections rm`.
    """
    cid = config.connection(connection)
    if not yes:
        # A wrong -c can select the name this is scoped to, which is worth one
        # keystroke to guard.
        click.confirm(
            f"wipe everything learned about {cid!r} — cache, turns and checkpoints?",
            abort=True,
        )
    http.run(_reset(cid))


async def _reset(cid: str) -> None:
    wiped = (await http.delete(f"/connections/{cid}/cache"))["wiped"]
    if not any(wiped.values()):
        click.echo(f"nothing to wipe — {cid!r} had learned nothing")
        return
    for table, rows in sorted(wiped.items()):
        if rows:
            click.echo(f"  wiped {table} ({rows} rows)")
    click.echo("reset complete")
