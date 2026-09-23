"""A committed Pareto front, read back for a person: a legend, the front,
and how each finalist opens.

`cli._pareto` writes the front to a file and prints it to stderr once, at the
end of a run. The file is what survives: `demo/gepa/*.pareto.json` is tracked
and a run's stderr usually is not. This reads a file back, so the numbers on
a slide can be shown coming out of the file.

    make gepa-extract-pareto                              # the last run's
    python -m tools.gepa.front demo/gepa/extract.pareto.json   # a committed one

Three sections, each headed, each starting at column one, so the eye has a
left edge to return to: LEGEND says what the terms are and what they weigh,
SCORES is the table, FINALISTS is how each candidate on it opens. An
overfit run's file adds HELD OUT between the last two.

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
        f"{len(document['objectives'])} objectives over {pool['candidates']} "
        f"candidates, {pool['on_front']} on the front"
    )


def render(
    document: dict[str, Any],
    *,
    legend: dict[str, str] | None = None,
    styled: bool = False,
) -> list[str]:
    """LEGEND, SCORES, and HELD OUT if the run kept one, as lines at
    column one. FINALISTS is `finalists`, so the search's own output, which
    has the candidates' text elsewhere, can leave it out.

    Plain by default. Styled for the display: headings bold, the best value
    in each column highlighted, the seed row marked.
    """
    lines: list[str] = []
    lines.extend(_legend(document, legend or {}, styled))
    lines.extend(_front(document, styled))
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


def _legend(document: dict[str, Any], words: dict[str, str], styled: bool) -> list[str]:
    """One line per term: the term, its weight, what a 1.0 means. A term the
    metric has no words for shows its weight alone, a gap to notice."""
    objectives = tuple(document["objectives"])
    weights = document.get("weights") or {}
    if not weights and not words:
        return []
    width = max(len(o) for o in objectives)
    lines = _heading("LEGEND", styled)
    lines.extend(_header(f"{'term':<{width}}  weight  what a perfect score means", styled))
    for o in objectives:
        weight = f"{weights[o]:.2f}" if o in weights else "-"
        lines.append(f"{o:<{width}}  {weight:>6}  {words.get(o, '')}".rstrip())
    return lines


def _front(document: dict[str, Any], styled: bool) -> list[str]:
    objectives = tuple(document["objectives"])
    front = document["front"]
    width = {o: max(len(o), 5) for o in objectives}
    name_width = max([4] + [len(_who(e)) for e in front])

    lines = _heading("SCORES", styled)
    header = "  ".join(f"{o:>{width[o]}}" for o in objectives)
    lines.extend(_header(
        f"{'cand':<{name_width}}  {'val':<5}  {header}  {'chars':>6}  best on", styled
    ))

    # The best on each term across the front, so the cell that earns a
    # candidate its place can be seen without reading the last column.
    best_on = {o: max((e["scores"][o] for e in front), default=None) for o in objectives}
    for entry in front:
        cells = []
        for o in objectives:
            cell = f"{entry['scores'][o]:>{width[o]}.2f}"
            top = best_on[o] is not None and entry["scores"][o] >= best_on[o]
            cells.append(_style(cell, styled and top, fg="green", bold=True))
        who = _style(f"{_who(entry):<{name_width}}", styled and bool(entry.get("seed")), fg="yellow", bold=True)
        val = f"{entry['val']:.3f}" if entry.get("val") is not None else "-"
        chars = f"{entry['chars']:,}" if entry.get("chars") is not None else "-"
        best = ", ".join(entry.get("tops") or [])
        lines.append(f"{who}  {val:<5}  {'  '.join(cells)}  {chars:>6}  {best}".rstrip())
    # No `best` row: the file keeps `best_per_objective`, and the highlighted
    # cells already say which candidate wins each column.
    return lines


def _who(entry: dict[str, Any]) -> str:
    return "seed" if entry.get("seed") else str(entry["index"])


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


def finalists(document: dict[str, Any], *, styled: bool = False) -> list[str]:
    """How each candidate on the front opens, seed first.

    The first `PREVIEW` characters of each component, whitespace collapsed,
    on the line under the finalist's name: enough to tell them apart, not
    enough to scroll through. The full text is in the file, and the diff is
    in the run's transcript. A candidate with several components — the four
    tool descriptions — gets one line each.
    """
    entries = sorted(document["front"], key=lambda e: (not e.get("seed"), e["index"]))
    lines = _heading("FINALISTS", styled)
    for n, entry in enumerate(entries):
        if n:
            lines.append("")
        who = "seed" if entry.get("seed") else f"candidate {entry['index']}"
        val = f"{entry['val']:.3f}" if entry.get("val") is not None else "-"
        lines.append(_style(f"{who} ({val})", styled, bold=True,
                            fg="yellow" if entry.get("seed") else "cyan"))
        components = entry.get("components") or {}
        for name, text in components.items():
            label = f"[{name}] " if len(components) > 1 else ""
            lines.append(f"{label}{_preview(text)}")
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


def _words_for(target: str | None) -> dict[str, str]:
    """The metric's legend for a file's target, through the registry, so a
    committed file from any target gets its words. Unknown target: none."""
    from tools.gepa import targets

    chosen = targets.TARGETS.get(target or "")
    return chosen.legend() if chosen else {}


def _style(text: str, on: bool, **kwargs: Any) -> str:
    return click.style(text, **kwargs) if on and text else text


def main(paths: list[str]) -> int:
    """Each file: where it is and when it was written, the summary, then the
    sections. A file that is missing or is not a front gets one line, and
    the exit status says so at the end, after the rest have printed."""
    if not paths:
        click.echo("no front to show: no run of that target has left one in "
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
        if not isinstance(document, dict) or "front" not in document:
            click.echo(f"{raw}: not a front")
            failed += 1
            continue

        # When, because "it ran last night" is a claim the file can back up.
        written = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        click.echo(_style(raw, styled, bold=True) + _style(f"  (written {written})", styled, dim=True))
        click.echo(summary(document))
        for line in about(_about_for(document.get("target")), styled=styled):
            click.echo(line)
        for line in render(document, legend=_words_for(document.get("target")), styled=styled):
            click.echo(line)
        for line in finalists(document, styled=styled):
            click.echo(line)
        click.echo()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
