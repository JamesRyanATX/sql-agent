"""A committed Pareto front, as the table the search printed, and the text
of every candidate on it.

`cli._pareto` writes the front to a file and prints it to stderr once, at the
end of a run. The file is what survives: `demo/gepa/*.pareto.json` is tracked
and a run's stderr usually is not. This reads a file back as the same table,
so the numbers on a slide can be shown coming out of the file, and then
prints what each candidate on the front actually says, which the table
cannot.

    make gepa-extract-pareto                              # the last run's
    python -m tools.gepa.front demo/gepa/extract.pareto.json   # a committed one

stdout, deliberately: this is a display. The search's stdout is the promoted
text, which is why its table went to stderr. Colour when stdout is a
terminal, none when it is a pipe; click decides.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click


def render(document: dict[str, Any], *, styled: bool = False) -> list[str]:
    """The front as lines, then the held-out table if the run kept one.

    Plain by default, which is what the search prints to stderr at the end
    of a run: the gate spends green and red on pass and fail, and a table
    where every row is one colour is not saying anything. Styled for the
    display command, where the table is the whole point: the best value in
    each column stands out, and the seed row is marked.
    """
    objectives = tuple(document["objectives"])
    pool = document["pool"]
    front = document["front"]
    lines = [
        f"\npareto    {len(objectives)} objectives over {pool['candidates']} "
        f"candidates, {pool['on_front']} on the front\n"
    ]

    width = {o: max(len(o), 6) for o in objectives}
    header = "  ".join(f"{o:>{width[o]}}" for o in objectives)
    head = f"          {'cand':>4}  {'val':>6}  {header}  {'chars':>6}"
    lines.append(_style(head, styled, bold=True))

    # The best on each term across the front, so the cell that earns a
    # candidate its place can be seen without reading the `tops` note.
    best_on = {
        o: max((e["scores"][o] for e in front), default=None) for o in objectives
    }
    for entry in front:
        cells = []
        for o in objectives:
            cell = f"{entry['scores'][o]:>{width[o]}.2f}"
            top = best_on[o] is not None and entry["scores"][o] >= best_on[o]
            cells.append(_style(cell, styled and top, fg="green", bold=True))
        note = "seed" if entry.get("seed") else ""
        if entry.get("tops"):
            note = (note + "  " if note else "") + "tops " + ", ".join(entry["tops"])
        val = f"{entry['val']:.3f}" if entry.get("val") is not None else "-"
        chars = f"{entry['chars']:,}" if entry.get("chars") is not None else "-"
        row = (
            f"          {entry['index']:>4}  {val:>6}  {'  '.join(cells)}  {chars:>6}  "
            f"{_style(note, styled, fg='yellow' if entry.get('seed') else None, dim=not entry.get('seed'))}"
        ).rstrip()
        lines.append(row)

    best = document.get("best_per_objective") or {}
    if best:
        cells = "  ".join(
            f"{best.get(o, float('nan')):>{width[o]}.2f}" for o in objectives
        )
        lines.append(_style(f"          {'best':>4}  {'-':>6}  {cells}", styled, dim=True))

    if document.get("holdout"):
        lines.extend(_holdout(document["holdout"], pool.get("seed_index"), styled))
    return lines


def _holdout(held: dict[str, Any], seed_index: int | None, styled: bool) -> list[str]:
    """The overfit run's second table: what the unseen cases said."""
    lines = [
        f"\nholdout   {held['cases']} cases the search never saw\n",
        _style(
            f"          {'cand':>4}  {'train':>6}  {'holdout':>7}  worse than the seed on",
            styled, bold=True,
        ),
    ]
    # JSON keys are strings; the search's own dict had ints.
    for key, row in held["candidates"].items():
        i = int(key)
        train = f"{row['train']:.3f}" if row.get("train") is not None else "-"
        hold = f"{row['holdout']:.3f}" if row.get("holdout") is not None else "-"
        worse = row.get("worse_than_seed_on")
        if i == seed_index:
            note = _style("seed", styled, fg="yellow")
        elif worse is None:
            note = "no seed to compare with"
        else:
            note = _style(f"{worse} of {held['cases']}", styled, fg="red", bold=True)
        lines.append(f"          {i:>4}  {train:>6}  {hold:>7}  {note}")
    return lines


def candidates(document: dict[str, Any], *, styled: bool = False) -> list[str]:
    """What each candidate on the front says, in full, seed first.

    The table says a candidate is 64% longer and bought grounding with it;
    this is where the 64% is read. One component is printed as it is; a
    candidate with several — the four tool descriptions — gets a sub-heading
    per component, in the file's order.
    """
    entries = sorted(document["front"], key=lambda e: (not e.get("seed"), e["index"]))
    lines: list[str] = []
    for entry in entries:
        who = "seed" if entry.get("seed") else f"candidate {entry['index']}"
        val = f"val {entry['val']:.3f}" if entry.get("val") is not None else "val -"
        chars = f"{entry['chars']:,} chars" if entry.get("chars") is not None else ""
        title = f" {who}  {val}  {chars} ".rstrip() + " "
        lines.append("")
        lines.append(_style(f"──{title}{'─' * max(0, 76 - len(title))}", styled, bold=True,
                            fg="yellow" if entry.get("seed") else "cyan"))
        components = entry.get("components") or {}
        for name, text in components.items():
            if len(components) > 1:
                lines.append("")
                lines.append(_style(f"  [{name}]", styled, bold=True))
            lines.append("")
            lines.extend(text.rstrip("\n").splitlines())
    return lines


def _style(text: str, on: bool, **kwargs: Any) -> str:
    return click.style(text, **kwargs) if on and text else text


def main(paths: list[str]) -> int:
    """Each file: where it is and when it was written, what it searched and
    at what weights, the tables, then every candidate on the front in full.
    A file that is missing or is not a front gets one line, and the exit
    status says so at the end, after the rest have printed."""
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

        weights = ", ".join(f"{k} {v:.2f}" for k, v in document.get("weights", {}).items())
        # When, because "it ran last night" is a claim the file can back up.
        written = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        click.echo(_style(raw, styled, bold=True) + _style(f"  (written {written})", styled, dim=True))
        click.echo(f"{document.get('target', '?')}: {weights}")
        for line in render(document, styled=styled):
            click.echo(line)
        for line in candidates(document, styled=styled):
            click.echo(line)
        click.echo()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
