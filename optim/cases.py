"""One replayable `extract` call, however it was obtained.

Two kinds of case, one type. A **harvested** case comes out of Langfuse and
carries the user message the node actually sent, byte for byte. An **authored**
case is a probe: a situation constructed to make one invariant testable, built
through `graph.extract_message` so it exercises the real assembly rather than a
hand-typed imitation of it.

The verbatim rule matters for the harvested half. Re-deriving the message from
its parts would mean this module reimplements graph.py's f-string, and the two
would drift silently — the model would be replayed a message production never
sends, and the score would be reported with a straight face. Only the SQL is
split back out, because the verification gate needs it, and it is split on an
anchor graph.py owns.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app import graph

# Bumped by hand when `graph.extract_message` or this dataclass changes shape,
# with a comment saying what moved. A corpus recorded under an older assembly is
# not wrong so much as answering a different question, and `optimize` refuses a
# stale one: tuning a prompt against a message the product no longer sends is
# the kind of result that reproduces perfectly and means nothing.
#
# 2: dropped `turn_id`, added `prompt_fp`. The harvest stopped joining to the
#    `turn` table — see optim/harvest.py for why — so provenance is the trace,
#    not the row, and the fingerprint is what keeps round two of an
#    optimisation off round one's output.
FORMAT_VERSION = 2


@dataclass(frozen=True)
class ExtractCase:
    """The inputs to one `extract` call, plus what it cost when it really ran."""

    name: str
    user_message: str
    sql: str
    # name -> claim, as shown to the model in the "Already filed" block. Parsed
    # back out of a harvested message so the name-discipline metric knows what
    # reusing a name would overwrite.
    filed: dict[str, str] = field(default_factory=dict)

    # Provenance. All absent on an authored probe, which is how the two are told
    # apart without a second type or a flag.
    obs_id: str | None = None
    trace_id: str | None = None
    connection_id: str | None = None
    # The 8-char fingerprint of the `extract` prompt that produced this case,
    # off the turn span's metadata. Once a candidate ships, new traces come from
    # the thing being optimised — without this a second round trains on the
    # first round's output and the improvement it measures is its own echo.
    prompt_fp: str | None = None
    # The recorded run's output tokens. The cost term is a one-sided penalty
    # against this, so a candidate is measured against what production paid
    # rather than against an absolute nobody chose.
    baseline_tokens_out: int = 0
    format_version: int = FORMAT_VERSION

    @classmethod
    def authored(
        cls,
        *,
        name: str,
        question: str,
        sql: str,
        findings: str,
        filed: dict[str, str] | None = None,
    ) -> ExtractCase:
        """A probe case, assembled by the same function production uses."""
        filed = dict(filed or {})
        return cls(
            name=name,
            user_message=graph.extract_message(
                question=question,
                sql=sql,
                findings=findings,
                cache=[{"name": n, "claim": c} for n, c in filed.items()],
            ),
            sql=sql,
            filed=filed,
        )


def sql_from(user_message: str) -> str | None:
    """Split the executed SQL back out of a recorded message.

    Returns None rather than guessing. A case that will not parse is dropped
    with a printed count — never silently, because a corpus that quietly halved
    is a corpus whose scores are about a different population.
    """
    _, anchor, rest = user_message.partition(graph.EXTRACT_SQL_ANCHOR)
    if not anchor:
        return None
    sql = rest.split("\n\n", 1)[0].strip()
    return sql or None


def filed_from(user_message: str) -> dict[str, str]:
    """The already-filed names and claims, out of a recorded message."""
    _, anchor, rest = user_message.partition(graph.EXTRACT_FILED_ANCHOR)
    if not anchor:
        return {}
    filed: dict[str, str] = {}
    for line in rest.splitlines():
        if line.startswith("- ") and ": " in line:
            name, _, claim = line[2:].partition(": ")
            filed[name] = claim
    return filed


# ------------------------------------------------------------------- JSONL io


def write_jsonl(path: Path, cases: list[ExtractCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[ExtractCase]:
    """Load a corpus, refusing one recorded under a different assembly.

    The version is checked on the raw row, before the dataclass is built. A
    stale corpus usually differs by a *field*, so constructing first turns a
    legible "re-harvest" into an unexpected-keyword TypeError.
    """
    rows = list(_rows(path))
    stale = {r.get("format_version") for r in rows} - {FORMAT_VERSION}
    if stale:
        raise ValueError(
            f"{path} holds cases at format_version {sorted(stale, key=str)}, but "
            f"the harvest is at {FORMAT_VERSION}. Re-harvest — these cases "
            "either replay a message the product no longer sends or are missing "
            "provenance a later stage needs."
        )
    return [ExtractCase(**row) for row in rows]


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
