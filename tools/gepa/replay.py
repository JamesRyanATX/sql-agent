"""Run one `extract` case against one candidate prompt.

The unit of evaluation, and choosing it is the central design decision. A whole
turn costs ~11.5k tokens and minutes of wall clock, while GEPA wants 100-500
scored rollouts — but `extract` is nearly a pure function of three recorded
strings, so one metric call is one model call and needs **no database at all**.

What that unit cannot measure is the flywheel, which is a property of a sequence
of turns. Whole-turn A/B is a final gate over one or two candidates, run over
HTTP, not a metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app import graph, llm, store
from app.config import config
from tools.gepa.cases import ExtractCase

# Not "extract": the generation name is what `tracing.observations()` filters
# on, so sharing it would let round two train on round one's output.
NODE = "extract.replay"


@dataclass
class Replayed:
    """What one candidate did with one case."""

    case: ExtractCase
    entries: list[store.CacheEntry] = field(default_factory=list)
    raw: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    # Set when the call or the parse failed. The metric turns this into score 0
    # with the text as feedback: GEPA's contract is that a candidate which
    # crashes scored badly, not that the run stops.
    error: str | None = None

    @property
    def recipes(self) -> list[store.CacheEntry]:
        return [e for e in self.entries if e.kind == "recipe"]

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.entries if e.name]


async def replay(
    candidate: str, case: ExtractCase, *, effort: str | None = None
) -> Replayed:
    """One model call, then the graph's own post-processing.

    `entries_from` rather than a re-implementation, so the verification gate the
    metric reads is the one production applies. `fallback_tables` is empty
    because inferring them needs a target connection and nothing here scores
    `tables` — which is what keeps this database-free.
    """
    try:
        result = await llm.complete(
            system=candidate,
            messages=[{"role": "user", "content": case.user_message}],
            effort=effort or config().effort_for("extract"),
            schema=graph.EXTRACT_SCHEMA,
            node=NODE,
        )
        raw = result.parsed().get("entries", [])
    except Exception as e:
        return Replayed(case=case, error=f"{type(e).__name__}: {e}")

    try:
        entries = graph.entries_from(raw, case.sql, [])
    except (KeyError, TypeError) as e:
        # The schema is enforced provider-side, so this is nearly unreachable —
        # and a candidate that finds the gap should score 0, not end the run.
        return Replayed(
            case=case,
            raw=raw,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            error=f"output did not fit EXTRACT_SCHEMA: {type(e).__name__}: {e}",
        )

    return Replayed(
        case=case,
        entries=entries,
        raw=raw,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
    )
