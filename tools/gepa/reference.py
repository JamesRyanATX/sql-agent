"""Run the reference queries, and refuse to proceed if one has drifted.

Split from `golden.py` because that module is data and imports nothing from
`app`; this one opens a connection.

**Once per batch, in one transaction.** `CHALLENGE.md` asks for the reference to
run inside the same rollout as the candidate, so that a question anchored to
`now()` is not a special case. One transaction is the stronger version of that
promise: in Postgres `now()` is the transaction's start time, so every reference
resolved here sees one instant, where nineteen references resolved inside
nineteen concurrent rollouts would see nineteen. It is also 19 executions per
batch instead of 19 per rollout, several of them three-way joins over 18,000
rows, against a connection pool the rollouts are already using.

What that costs: the references are resolved before the rollouts launch, so a
"last 30 days" window can be one batch older than the SQL a candidate finally
runs. Nothing in `demo/golden/` sits inside that band — `orders.created` is
truncated to the day, so no row is within minutes of a window boundary unless a
batch runs across midnight.

**Drift raises.** A candidate that falls over must not end a run; that is GEPA's
adapter contract and `replay_turn` honours it. A *corpus* that falls over must
end the run, because scoring a hundred and fifty rollouts against a reference
that no longer matches the number written beside it is a result that reproduces
perfectly and means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from app import db
from tools.gepa import resultset
from tools.gepa.golden import GoldenCase
from tools.gepa.resultset import Row


class ReferenceDrift(RuntimeError):
    """A reference query no longer returns the answer written beside it."""


@dataclass(frozen=True)
class Reference:
    """What a case's reference query returned, and when."""

    case: GoldenCase
    rows: list[Row]
    at: datetime


async def resolve(cases: Sequence[GoldenCase]) -> dict[str, Reference]:
    """Every reference query, once, right now, keyed by case name."""
    resolved: dict[str, Reference] = {}
    at = datetime.now(timezone.utc)

    async with db.target_readonly() as conn:
        for case in cases:
            result = await conn.exec_driver_sql(case.reference_sql)
            rows = resultset.rows_of([tuple(row) for row in result.fetchall()])
            _check(case, rows)
            resolved[case.name] = Reference(case=case, rows=rows, at=at)

    return resolved


def _check(case: GoldenCase, rows: list[Row]) -> None:
    """Two ways a reference stops being a reference, both silent otherwise."""
    if not rows:
        raise ReferenceDrift(
            f"{case.name}: the reference query came back empty. Every candidate "
            f"then scores the same on this case, which reads as GEPA finding "
            f"nothing. Fix demo/golden/ or demo/demo.sql.\n\n{case.reference_sql}"
        )

    if case.expect is None:
        return

    match = resultset.compare(case.expect, rows, ordered=case.ordered)
    if not match.exact:
        raise ReferenceDrift(
            f"{case.name}: the reference query disagrees with the answer "
            f"written beside it.\n"
            f"  written down: {case.expect}\n"
            f"  it returned:  {[list(row) for row in rows]}\n"
            f"Either demo.sql moved under the query or the query was wrong when "
            f"it was written. The number is the only thing here that fails "
            f"loudly, so this is it failing loudly.\n\n{case.reference_sql}"
        )
