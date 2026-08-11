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

import re
from collections.abc import Iterable, Sequence
from typing import Any

import click

# Wide enough for a timestamp, an email or a product name; narrow enough that one
# long value cannot destroy the alignment of every row below it or wrap the
# tape's awaited line off screen. `--json` is the untruncated view.
CELL = 40

# What right-aligns. `isinstance(v, int | float)` is not enough on its own:
# app/graph.py round-trips rows through `json.dumps(..., default=str)`, so a
# Postgres numeric arrives as the string "1234.50" and would otherwise sit
# left-aligned next to the integers it is a column with.
_NUMERIC = re.compile(r"-?\d+(\.\d+)?$")


def dim(text: str) -> str:
    return click.style(text, dim=True)


def bold(text: str) -> str:
    return click.style(text, bold=True)


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    right: frozenset[str] = frozenset(),
    footer: str | None = None,
) -> None:
    """Columns sized to their contents, and a psql-shaped row count.

    The `(N rows)` footer earns its place beyond tidiness: `demo/demo.tape`
    waits for `rows)` to know `sql-agent turns` has finished printing, and VHS
    times out on a pattern that never appears. Waiting on a header word instead
    would match the instant the header lands, which is the stale-match failure
    the tape's own comment warns about.

    `footer` overrides that count for a caller that knows something the row list
    does not — see `result()`, where a full page is not a total.
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
    click.echo(
        dim(footer or f"({len(rows)} row{'' if len(rows) == 1 else 's'})")
    )


def _numeric(values: Iterable[Any]) -> bool:
    """Does this column hold nothing but numbers? Nulls do not count against it."""
    seen = False
    for v in values:
        if v is None:
            continue
        seen = True
        if isinstance(v, bool) or not (
            isinstance(v, int | float) or _NUMERIC.fullmatch(str(v))
        ):
            return False
    return seen


def _cell(value: Any) -> str:
    if value is None:
        return ""  # what psql shows for NULL, and what `table` already does
    text = str(value)
    return text if len(text) <= CELL else text[: CELL - 1] + "…"


def result(rows: Sequence[dict[str, Any]], *, capped: bool = False) -> None:
    """A query's rows, the way psql would show them.

    Alignment is decided per column by what is *in* it, not by its name the way
    `table`'s `right=` works — that suits three fixed reports and cannot suit
    arbitrary SQL, where the column list is whatever the model just wrote.

    `capped` means the result filled `max_rows` and the true total is unknown:
    `execute` uses `fetchmany`, which cannot tell a full page from a result that
    happened to be exactly that long. Printing `(50 rows)` there would be a
    number this program does not have.
    """
    if not rows:
        # No row means no keys, so there is no header to print either.
        click.echo(dim("(0 rows)"))
        return

    # `dict(m)` and the JSON round trip both preserve insertion order, so this is
    # the SELECT's own column order rather than a guess at one.
    headers = list(rows[0])
    click.echo()
    table(
        headers,
        [[_cell(r.get(h)) for h in headers] for r in rows],
        right=frozenset(h for h in headers if _numeric(r.get(h) for r in rows)),
        footer=f"({len(rows)} rows — more matched)" if capped else None,
    )


def detail(pairs: Sequence[tuple[str, Any]]) -> None:
    """A `key   value` block, for one object."""
    width = max((len(k) for k, _ in pairs), default=0)
    for key, value in pairs:
        click.echo(f"  {dim(key.ljust(width))}  {'' if value is None else value}")
