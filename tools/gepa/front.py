"""A committed GEPA report, read back for a person: what the score means,
every candidate's score on every validation example, and how each eligible
candidate opens.

`cli._pareto` writes the report to a file and prints it to stderr once, at the
end of a run. The file is what survives: `demo/gepa/*.pareto.json` is tracked
and a run's stderr usually is not. This reads a file back, so the numbers on
a slide can be shown coming out of the file.

    make gepa-extract-pareto                              # the last run's
    python -m tools.gepa.front demo/gepa/extract.pareto.json   # a committed one

The table is the one GEPA selects from. Each row is a candidate and each
column is one validation example. GEPA picks the next candidate to mutate from
the candidates that have the top score on at least one example, which is what
the `best on` column lists.

Each section is headed and starts at column one, so the eye has a left edge to
return to: ABOUT says what is being searched, SCORE says what the number
means, SCORES is the table with the examples numbered under it, and FINALISTS
is how each eligible candidate opens. An overfit run's file adds HELD OUT
between the last two.

stdout, deliberately: this is a display. The search's stdout is the promoted
text, which is why its table went to stderr. Colour when stdout is a
terminal, none when it is a pipe; click decides.
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import click

PREVIEW = 120


def summary(document: dict[str, Any]) -> str:
    pool = document["pool"]
    return (
        f"{_plural(pool['candidates'], 'candidate')} over "
        f"{_plural(len(document['examples']), 'validation example')}, "
        f"{pool['eligible']} best on at least one"
    )


def render(document: dict[str, Any], *, styled: bool = False) -> list[str]:
    """SCORE, SCORES, and HELD OUT if the run kept one, as lines at column
    one. FINALISTS is `finalists`, so the search's own output, which has the
    candidates' text elsewhere, can leave it out.

    Plain by default. Styled for the display: headings bold, the top value in
    each example's column highlighted, the seed row marked.
    """
    lines: list[str] = []
    lines.extend(_score(document, styled))
    lines.extend(_table(document, styled))
    if document.get("holdout"):
        lines.extend(_holdout(document, styled))
    return lines


def _heading(text: str, styled: bool) -> list[str]:
    """A blank line either side, so a section is a block the eye can land on."""
    return ["", _style(text, styled, bold=True), ""]


def _header(text: str, styled: bool) -> list[str]:
    """A table's column names, bright when styled, with a rule underneath in
    either case: the rule is what separates names from numbers at a glance."""
    return [_style(text, styled, fg="bright_white", bold=True), "-" * len(text)]


def _score(document: dict[str, Any], styled: bool) -> list[str]:
    """The one sentence saying what the number means, wrapped. Nothing when
    the file has no sentence."""
    sentence = document.get("score") or ""
    if not sentence:
        return []
    return _heading("SCORE", styled) + textwrap.wrap(sentence, width=78)


def _table(document: dict[str, Any], styled: bool) -> list[str]:
    """One row per candidate: its mean, its score on each example, its length
    and the examples it is best on. Then the examples, numbered.

    A candidate GEPA did not score on an example shows `-` there, and the
    file holds null.
    """
    rows = document["candidates"]
    examples = document["examples"]
    columns = [f"E{n}" for n in range(1, len(examples) + 1)]
    width = max([4] + [len(c) for c in columns])
    name_width = max([4] + [len(_who(row)) for row in rows])

    lines = _heading("SCORES", styled)
    header = "  ".join(f"{c:>{width}}" for c in columns)
    lines.extend(_header(
        f"{'cand':<{name_width}}  {'val':<5}  {header}  {'chars':>6}  best on", styled
    ))

    # The top score in each example's column, so the cell that makes a
    # candidate eligible can be seen without reading the last column.
    top = [
        max((r["scores"][n] for r in rows if r["scores"][n] is not None), default=None)
        for n in range(len(examples))
    ]
    for row in rows:
        cells = []
        for n, value in enumerate(row["scores"]):
            if value is None:
                cells.append(f"{'-':>{width}}")
                continue
            cell = f"{value:>{width}.2f}"
            cells.append(_style(cell, styled and value >= top[n], fg="green", bold=True))
        who = _style(
            f"{_who(row):<{name_width}}", styled and bool(row.get("seed")),
            fg="yellow", bold=True,
        )
        val = f"{row['val']:.3f}" if row.get("val") is not None else "-"
        chars = f"{row['chars']:,}" if row.get("chars") is not None else "-"
        best = _ranges(row.get("best_on") or [])
        lines.append(f"{who}  {val:<5}  {'  '.join(cells)}  {chars:>6}  {best}".rstrip())

    lines.append("")
    label_width = max([len(c) for c in columns] or [0])
    for column, label in zip(columns, examples):
        lines.append(f"{column:<{label_width}}  {label}")
    return lines


def _ranges(numbers: list[int]) -> str:
    """Example numbers as ranges: [1, 2, 3, 5] is `E1-E3, E5`."""
    parts: list[str] = []
    ordered = sorted(numbers)
    start = previous = None
    for n in ordered + [None]:
        if start is not None and n == previous + 1:
            previous = n
            continue
        if start is not None:
            parts.append(f"E{start}" if start == previous else f"E{start}-E{previous}")
        start = previous = n
    return ", ".join(parts)


def _who(row: dict[str, Any]) -> str:
    return "seed" if row.get("seed") else str(row["index"])


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _holdout(document: dict[str, Any], styled: bool) -> list[str]:
    """The overfit run's second table: what the unseen cases said."""
    held = document["holdout"]
    seed_index = document["pool"].get("seed_index")
    lines = _heading("HELD OUT", styled)
    lines.extend(_header(
        f"{'cand':<4}  {'train':<5}  {'holdout':<7}  worse than the seed on {held['cases']}", styled
    ))
    # JSON keys are strings; the search's own dict had ints.
    for key, row in held["candidates"].items():
        i = int(key)
        who = "seed" if i == seed_index else str(i)
        train = f"{row['train']:.3f}" if row.get("train") is not None else "-"
        hold = f"{row['holdout']:.3f}" if row.get("holdout") is not None else "-"
        worse = row.get("worse_than_seed_on")
        if i == seed_index:
            note = ""
        elif worse is None:
            note = "no seed to compare with"
        else:
            note = _style(f"{worse} of {held['cases']}", styled, fg="red", bold=True)
        lines.append(f"{who:<4}  {train:<5}  {hold:<7}  {note}".rstrip())
    return lines


def finalists(
    document: dict[str, Any], *, whole: bool = False, styled: bool = False
) -> list[str]:
    """Each candidate that is best on at least one example, seed first: its
    text whole, or how it opens.

    Whole is for a target whose text is short enough to read on the screen —
    four tool descriptions, a six-line config block — with line breaks kept,
    since a YAML block collapsed onto one line is not YAML. Otherwise the
    first `PREVIEW` characters of each component, whitespace collapsed, on
    the line under the finalist's name: enough to tell a page-long prompt
    from its sibling, not enough to scroll through. The full text is in the
    file, and the diff is in the run's transcript. A candidate with several
    components is labelled per component either way.
    """
    eligible = [row for row in document["candidates"] if row.get("best_on")]
    rows = sorted(eligible, key=lambda r: (not r.get("seed"), r["index"]))
    lines = _heading("FINALISTS", styled)
    for n, row in enumerate(rows):
        if n:
            lines.append("")
        who = "seed" if row.get("seed") else f"candidate {row['index']}"
        val = f"{row['val']:.3f}" if row.get("val") is not None else "-"
        lines.append(_style(f"{who} ({val})", styled, bold=True,
                            fg="yellow" if row.get("seed") else "cyan"))
        components = row.get("components") or {}
        for name, text in components.items():
            label = f"[{name}]" if len(components) > 1 else ""
            if whole:
                if label:
                    lines.append(_style(label, styled, dim=True))
                lines.extend(_whole(text))
            else:
                lines.append(f"{label} {_preview(text)}".lstrip())
    return lines


def _whole(text: str, width: int = 78) -> list[str]:
    """The text as written, each of its lines wrapped to the width."""
    lines: list[str] = []
    for line in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(line, width=width) or [""])
    return lines


def _preview(text: str, limit: int = PREVIEW) -> str:
    """The opening of a prompt, one line. Cut at a word, with an ellipsis
    when there was more."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0] or flat[:limit]
    return cut + " …"


def about(text: str, *, styled: bool = False) -> list[str]:
    """The ABOUT section: two sentences for a newcomer, wrapped."""
    if not text:
        return []
    return _heading("ABOUT", styled) + textwrap.wrap(text, width=78)


def _about_for(target: str | None) -> str:
    from tools.gepa import targets

    chosen = targets.TARGETS.get(target or "")
    return chosen.about if chosen else ""


def _whole_for(target: str | None) -> bool:
    """Whether this target's finalists are printed whole. Unknown target:
    previewed, which is the safe size."""
    from tools.gepa import targets

    chosen = targets.TARGETS.get(target or "")
    return bool(chosen and chosen.whole)


def _style(text: str, on: bool, **kwargs: Any) -> str:
    return click.style(text, **kwargs) if on and text else text


def main(paths: list[str]) -> int:
    """Each file: where it is and when it was written, the summary, then the
    sections. A file that is missing or is not a report gets one line, and
    the exit status says so at the end, after the rest have printed."""
    if not paths:
        click.echo("no report to show: no run of that target has left one in "
                   "tools/gepa/out/ — run the search, or check the target name")
        return 1

    styled = sys.stdout.isatty()
    failed = 0
    for raw in paths:
        path = Path(raw)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            click.echo(f"{raw}: no such file")
            failed += 1
            continue
        except json.JSONDecodeError as e:
            click.echo(f"{raw}: not JSON ({e.msg})")
            failed += 1
            continue
        if isinstance(document, dict) and "candidates" not in document and (
            "front" in document or "objectives" in document
        ):
            # The format before the report listed candidates against
            # validation examples. Its numbers are per term, and there is no
            # way to turn them into the table this prints.
            click.echo(
                f"{raw}: written before the report changed to candidates "
                f"against validation examples — `make gepa-"
                f"{document.get('target') or '<target>'}` writes a new one"
            )
            failed += 1
            continue
        if not isinstance(document, dict) or "candidates" not in document:
            click.echo(f"{raw}: not a report")
            failed += 1
            continue

        # When, because "it ran last night" is a claim the file can back up.
        written = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        click.echo(_style(raw, styled, bold=True) + _style(f"  (written {written})", styled, dim=True))
        click.echo(summary(document))
        for line in about(_about_for(document.get("target")), styled=styled):
            click.echo(line)
        for line in render(document, styled=styled):
            click.echo(line)
        for line in finalists(
            document, whole=_whole_for(document.get("target")), styled=styled
        ):
            click.echo(line)
        click.echo()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
