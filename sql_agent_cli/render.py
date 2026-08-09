"""Terminal output. Nothing here decides anything.

`click.style` rather than `rich`, deliberately. The demo is a VHS-recorded GIF
at a fixed 1200x700, and rich re-flows on COLUMNS and re-measures tables on
every render — a take stops being reproducible across machines. `rich.live` also
emits cursor-movement escapes that a frame differ renders as flicker. The whole
surface here is nine event types and three tables.

The one thing `click.echo` buys over the raw escapes this replaces: it strips
colour when stdout is not a terminal, so `sql-agent cache | less` is readable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import click


def dim(text: str) -> str:
    return click.style(text, dim=True)


def bold(text: str) -> str:
    return click.style(text, bold=True)


def fmt_rows(rows: Sequence[dict[str, Any]]) -> str:
    """The result of a query, as the answer line above it reads it."""
    if not rows:
        return "[]"
    return "[" + ", ".join(
        "(" + ", ".join(str(v) for v in r.values()) + ")" for r in rows
    ) + "]"


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    right: frozenset[str] = frozenset(),
) -> None:
    """Columns sized to their contents, and a psql-shaped row count.

    The `(N rows)` footer earns its place beyond tidiness: `demo/demo.tape`
    waits for `rows)` to know `sql-agent turns` has finished printing, and VHS
    times out on a pattern that never appears. Waiting on a header word instead
    would match the instant the header lands, which is the stale-match failure
    the tape's own comment warns about.
    """
    rows = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]

    def line(cells: Sequence[str]) -> str:
        return " | ".join(
            c.rjust(w) if headers[i] in right else c.ljust(w)
            for i, (c, w) in enumerate(zip(cells, widths))
        ).rstrip()

    click.echo(bold(line(headers)))
    click.echo(dim("-+-".join("-" * w for w in widths)))
    for row in rows:
        click.echo(line(row))
    click.echo()
    click.echo(dim(f"({len(rows)} row{'' if len(rows) == 1 else 's'})"))


def detail(pairs: Sequence[tuple[str, Any]]) -> None:
    """A `key   value` block, for one object."""
    width = max((len(k) for k, _ in pairs), default=0)
    for key, value in pairs:
        click.echo(f"  {dim(key.ljust(width))}  {'' if value is None else value}")
