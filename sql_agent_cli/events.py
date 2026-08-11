"""Rendering the turn stream.

**Default output has to fit one VHS screenful**, and that is an invariant rather
than a preference. `demo/demo.tape` waits on `[explored]`, `[no exploration]` and
`tokens (`; VHS can only see the first screenful, so an awaited line that scrolls
out of view times the recording out at forty minutes. That used to be stated as
"byte-identical to what `scripts/ask.py` printed", which was the wrong statement
of it — what the tape needs is output that is *bounded*, not output that is
frozen. `render.CELL` bounds the width; the row count is bounded only by
`max_rows`, so see the tape's own comment for where that ceiling actually bites.

So by default: the rows, an error if there was one, and the token line. What was
asked for and what it cost, and nothing that explains either — the SQL, the plan
decision, the exploration trace and the answer's own prose are all -v, because
they are all the same kind of thing.
"""

from __future__ import annotations

import json
from typing import Any

import click

from sql_agent_cli import render


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
        # One event, split across both views, because it carries two different
        # kinds of thing. The cost line *is* the turn and is always printed. The
        # prose explains a result the reader is already looking at — the table is
        # right there — so it belongs with the SQL, the plan decision and the
        # exploration trace, which is to say behind -v.
        #
        # Below the table rather than above it, and that is forced rather than
        # chosen: `rows` comes from `execute` and this comes from `answer`, with
        # `extract` — a model call — in between. Printing the sentence first
        # would mean holding the result until extract returned, which is several
        # seconds of blank screen at the end of a turn the user already waited
        # 45 seconds for.
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

    # `usage` is deliberately unrendered even here. It fires per node, for a
    # live counter widget; in a scrolling terminal it is noise, and the `answer`
    # event carries the totals. --json still emits it — that is what raw is for.


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
            # Styled substrings concatenated, not nested: click.style(dim=True)
            # wrapped around an already-styled span puts its reset in the middle
            # and the rest of the line loses the dim.
            body = click.style(
                f"[{e['kind']}] {e['name'] or '(unnamed)'}: {e['claim']}", dim=True
            )
            click.echo(f"  {tick} {body}")


def raw(ev: dict[str, Any]) -> None:
    click.echo(json.dumps(ev))
