"""Rendering the turn stream.

**Default output has to fit one VHS screenful**, and that is an invariant.
`demo/demo.tape` waits on `[explored]`, `[no exploration]` and `tokens (`, and
VHS only sees the first screenful — an awaited line that scrolls out of view
times the recording out at forty minutes. `render.CELL` bounds the width; the
row count is bounded only by `max_rows`.

So by default: the rows, an error if there was one, and the token line. The SQL,
the plan decision, the exploration trace and the answer's own prose are all -v.
"""

from __future__ import annotations

import json
from typing import Any

import click

from sql_agent import render


def cost(ev: dict[str, Any]) -> str:
    """The demo's whole point, in one line. Do not reformat — see the module
    docstring, and tests/test_cli_render.py, which pins it as a golden string."""
    return (
        f"{ev['total_tokens']:,} tokens "
        f"({ev['tokens_in']:,} in / {ev['tokens_out']:,} out) "
        f"in {ev['latency_ms'] / 1000:.1f}s"
        f"{'  [explored]' if ev['explored'] else '  [no exploration]'}"
    )


def show(ev: dict[str, Any], *, verbose: bool = False) -> None:
    kind = ev.get("type")
    if kind == "rows":
        render.result(ev["rows"], capped=ev.get("capped", False))
    elif kind == "error":
        # A silent failure with no explanation is worse than the noise it costs.
        click.secho(f"error: {ev['message']}", fg="red")
    elif kind == "answer":
        # One event across both views: the cost line *is* the turn and is always
        # printed, while the prose explains a table the reader is already
        # looking at, so it sits behind -v.
        #
        # Below the table because the order is forced: `rows` comes from
        # `execute` and this from `answer`, with `extract` — a model call — in
        # between. Printing the sentence first would hold the result back for
        # several seconds at the end of a turn the user already waited out.
        if verbose and (text := ev.get("text")):
            click.echo()
            click.echo(text)
        click.echo()
        click.secho(cost(ev), dim=True)
    elif verbose:
        _verbose(ev, kind)


def _verbose(ev: dict[str, Any], kind: str | None) -> None:
    if kind == "plan":
        # `used` is absent on a cold cache — the plan node never ran, so there
        # is nothing it could have used. .get(), not [].
        bits = [
            f"{ev['cache_entries']} entries",
            "sufficient" if ev["sufficient"] else "insufficient",
        ]
        if used := ev.get("used"):
            bits.append(f"used {', '.join(used)}")
        if ev.get("missing"):
            bits.append(f"missing {', '.join(ev['missing'])}")
        click.secho(f"plan: {'; '.join(bits)}", dim=True)

    elif kind == "explore":
        args = ", ".join(f"{k}={v!r}" for k, v in ev["input"].items())
        line = f"{'!' if ev['error'] else ' '} {ev['calls']:>2}. {ev['tool']}({args})"
        click.secho(line, fg="red" if ev["error"] else None, dim=not ev["error"])

    elif kind == "findings":
        click.secho(f"findings after {ev['tool_calls']} calls:", dim=True)
        click.secho(f"  {ev['text']}", dim=True)

    elif kind == "sql":
        click.secho(ev["sql"], fg="cyan")
        for assumption in ev.get("assumptions") or []:
            click.secho(f"  assumes: {assumption}", dim=True)

    elif kind == "fix":
        click.secho(f"fix {ev['attempt']}: {ev['was_wrong']}", fg="yellow")
        click.secho(ev["sql"], fg="cyan")

    elif kind == "learned":
        _learned(ev)

    # `usage` is unrendered even here: it fires per node for a live counter
    # widget, and the `answer` event already carries the totals. --json emits it.


def _learned(ev: dict[str, Any]) -> None:
    """Three shapes, and only one of them carries entries."""
    if ev.get("cached"):
        click.secho("learned: nothing — the cache already covered it", dim=True)
    elif failed := ev.get("failed"):
        click.secho(f"learned: extraction failed — {failed}", fg="yellow")
    else:
        skipped = f", {ev['skipped']} skipped" if ev.get("skipped") else ""
        click.secho(f"learned: {ev['count']} entries{skipped}", dim=True)
        for e in ev.get("entries") or []:
            tick = click.style("✓", fg="green") if e["verified"] else " "
            # Concatenated, not nested: click.style(dim=True) wrapped around an
            # already-styled span puts its reset in the middle, and the rest of
            # the line loses the dim.
            body = click.style(
                f"[{e['kind']}] {e['name'] or '(unnamed)'}: {e['claim']}", dim=True
            )
            click.echo(f"  {tick} {body}")


def raw(ev: dict[str, Any]) -> None:
    click.echo(json.dumps(ev))
