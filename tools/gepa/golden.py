"""Questions whose answer we know, written down by hand: the answer key.

Not the training set. `make corpus` asks each of these cold and files its
reference query, and where the fixture fixes it a verdict, on the turn's trace;
the `tools` and `config` searches then read their cases back from the traces
(`harvest.turn_cases`), the same way `extract` reads its calls. So the
directory primes the pump, once, and after that the corpus grows from use: any
turn anyone asks and scores. A golden case is a claim about a fixture, and the
fixture is `demo/demo.sql`, in this repo, so these are authored, and every
reference query is executed before it is committed
(`tests/test_golden_corpus.py`).

`GoldenCase` is also the shape a harvested turn case takes — name, question,
reference query, whether order matters, the reading — so `write_jsonl` and
`read_jsonl` below are what a harvest is cached and reused through.

**A reference query, and where the fixture fixes it, the number too.** Two
queries can be wrong together — the reference was subtly wrong when it was
written, or somebody edited `demo.sql` and both sides moved with it. `expect` is
the only thing here that fails loudly, so it is written down wherever it can be:
where the value is reachable by integer arithmetic over `generate_series` with
no `now()` in the path. That rule excludes every revenue case, because
`order_item.price` carries a discount that depends on how old an order is.

**Why these are committed when `tools/gepa/out/` is not.** That directory is
ignored because a harvested case holds a recorded prompt, and a recorded prompt
holds whatever the database it ran against holds — committing a harvest commits
customer data by construction. Everything here derives from `demo/demo.sql`,
which is fabricated, so that reasoning does not reach it. If you are about to
delete this directory by analogy with the ignore rule: don't.

One file per case rather than one JSONL, following `tests/probes/<node>/*.json`,
which is this repo's existing shape for authored contract data.

**`reference_sql` is a list of lines in the file and a string in here.** JSON
escapes a newline, so a query written as one string arrives as 280 characters on
one line, and the reference query is the claim a reviewer is actually checking —
they need to read the join conditions and the `WHERE` clause, and to see which
line changed when one does. The probe files put their SQL on one line because
there it is an input rather than the assertion.

Imports nothing from `app`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1

GOLDEN = Path(__file__).resolve().parents[2] / "demo" / "golden"


@dataclass(frozen=True)
class GoldenCase:
    """One question, the query that answers it, and what that query returns."""

    # A stable id, not the question. This is what the gate, the split report and
    # the reflective dataset print, and a 60-character question is not a label.
    name: str

    # Verbatim from `demo/questions.txt`. The corpus test asserts that, so the
    # two files cannot drift into asking subtly different things.
    question: str

    reference_sql: str

    # Which schema areas it touches, so a split can hold an area out. Checked
    # against the target by the corpus test.
    tables: list[str]

    # The literal answer, as rows, where `demo.sql` fixes it; None where it
    # slides. Always rows even for a scalar — `[[1840]]`, never `1840` — so one
    # comparison reads this and a live result set the same way.
    #
    # Only written where every cell is an exact integer, a short label or null.
    # JSON round-trips those exactly and does not round-trip a Decimal, and a
    # money literal arriving back as a float is a gate reporting drift where
    # there is none.
    expect: list[list[Any]] | None = None

    # Whether row order is part of the answer.
    ordered: bool = False

    # The reading this reference encodes, in a sentence. Several of these
    # questions are genuinely ambiguous — "revenue", before or after cancelled
    # orders? — and the reference picks one. This goes into the feedback, so a
    # candidate that lost on an ambiguity is told which ambiguity instead of
    # being left to guess.
    reading: str = ""

    # Which of `demo.sql`'s five traps the question is aimed at. Not scored;
    # it is what the talk's before-and-after table is grouped by.
    traps: list[str] = field(default_factory=list)

    format_version: int = FORMAT_VERSION


def load(directory: Path = GOLDEN) -> list[GoldenCase]:
    """Every case in the directory, in filename order.

    The version is checked on the raw row, before the dataclass is built. A
    stale case usually differs by a field, and constructing first turns a
    legible message into `TypeError: unexpected keyword argument`.
    """
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no golden cases in {directory}")

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    stale = {row.get("format_version") for row in rows} - {FORMAT_VERSION}
    if stale:
        raise ValueError(
            f"{directory} holds cases at format_version {sorted(stale)}, and "
            f"golden.py is at {FORMAT_VERSION}. These are authored by hand, not "
            f"harvested — the fields moved, so the files have to be edited."
        )

    for row in rows:
        # Lines on disk, one query in memory. See the module docstring.
        row["reference_sql"] = "\n".join(row["reference_sql"])
    return [GoldenCase(**row) for row in rows]


def write_jsonl(path: Path, cases: list[GoldenCase]) -> None:
    """A harvest, cached beside the run so `--resume` can reuse it. One case
    a line, `reference_sql` as the one string it is in memory: this file is
    read by the next run, not reviewed by a person, so the list-of-lines form
    the committed files use buys nothing here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(asdict(case)) + "\n")


def read_jsonl(path: Path) -> list[GoldenCase]:
    """The version is checked on the raw row first, as `load` does; a stale
    harvest is re-harvested rather than edited, and the message says so."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stale = {row.get("format_version") for row in rows} - {FORMAT_VERSION}
    if stale:
        raise ValueError(
            f"{path} holds cases at format_version {sorted(stale)}, and "
            f"golden.py is at {FORMAT_VERSION}. Re-harvest: run without --resume."
        )
    return [GoldenCase(**row) for row in rows]
