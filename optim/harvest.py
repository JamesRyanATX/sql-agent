"""Turn recorded `extract` calls into a replayable corpus.

Langfuse holds the inputs and outputs; Postgres holds the outcome and the
connection scope. The bridge between them is `turn.trace_id` — a column added by
`migrations/004_tracing.sql`, whose own header says the trace "lives in another
system entirely, on the other side of an HTTP exporter, and there is nothing
here to join it to". This is the join it did not have.

Nothing here scores anything. Harvesting and scoring are separate on purpose: a
corpus is expensive to produce and cheap to re-read, and every experiment should
be re-runnable against the same rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from app import db, store, tracing
from optim.cases import ExtractCase, filed_from, sql_from
from optim.replay import NODE as REPLAY_NODE


@dataclass
class Harvest:
    """What came back, and what did not. Printed rather than returned quietly."""

    cases: list[ExtractCase]
    seen: int = 0
    no_sql: int = 0
    no_message: int = 0
    wrong_connection: int = 0
    contaminated: int = 0

    def report(self) -> str:
        dropped = self.seen - len(self.cases)
        lines = [f"{len(self.cases)} cases from {self.seen} generations"]
        if dropped:
            lines.append(f"  dropped {dropped}:")
            for label, n in (
                ("no user message", self.no_message),
                ("SQL would not parse", self.no_sql),
                ("another connection", self.wrong_connection),
                ("written by the harness itself", self.contaminated),
            ):
                if n:
                    lines.append(f"    {n} {label}")
        return "\n".join(lines)


async def extract_cases(
    *,
    connection_id: str,
    days: int = 30,
    since: datetime | None = None,
) -> Harvest:
    """Every `extract` call for one connection, as replayable cases.

    Scoped to one connection for the same reason `load_cache` is: what the agent
    learned about one warehouse is not evidence about another, and a corpus that
    mixes two is a corpus whose scores describe neither.
    """
    since = since or datetime.now(timezone.utc) - timedelta(days=days)
    traces = await _traces_for(connection_id, since)

    harvest = Harvest(cases=[])
    for observation in tracing.generations(node="extract", since=since):
        harvest.seen += 1

        # `replay` writes under `extract.replay`, so the harness's own calls
        # never reach a corpus — but a `name=` filter is one typo from being
        # wrong about that, and the cost of being wrong is round two training on
        # round one's output. Cheap to assert.
        if observation.get("name") == REPLAY_NODE:
            harvest.contaminated += 1
            continue

        trace_id = observation.get("trace_id")
        if trace_id not in traces:
            harvest.wrong_connection += 1
            continue

        message = _user_message(observation.get("input"))
        if not message:
            harvest.no_message += 1
            continue

        sql = sql_from(message)
        if not sql:
            harvest.no_sql += 1
            continue

        turn_id = traces[trace_id]
        harvest.cases.append(
            ExtractCase(
                name=f"turn-{turn_id}",
                user_message=message,
                sql=sql,
                filed=filed_from(message),
                obs_id=observation.get("id"),
                trace_id=trace_id,
                turn_id=turn_id,
                connection_id=connection_id,
                baseline_tokens_out=int(observation.get("usage", {}).get("output") or 0),
            )
        )

    return replace(harvest, cases=_dedupe(harvest.cases))


def _user_message(recorded: object) -> str | None:
    """The one user turn out of a recorded `{"system", "messages"}` input."""
    if not isinstance(recorded, dict):
        return None
    for message in reversed(recorded.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def _dedupe(cases: list[ExtractCase]) -> list[ExtractCase]:
    """One case per message.

    T2-style repeats of a question produce byte-identical extract inputs, and
    a corpus that holds the same case five times weights it five times — which
    would tune the prompt for whichever question the demo happens to ask twice.
    """
    seen: set[str] = set()
    unique = []
    for case in cases:
        if case.user_message in seen:
            continue
        seen.add(case.user_message)
        unique.append(case)
    return unique


async def _traces_for(connection_id: str, since: datetime) -> dict[str, int]:
    """trace_id -> turn id, for one connection. The Postgres half of the join.

    Also the connection filter. Scoping on the observation's own
    `connection:{id}` tag would need the `trace_context` field group and would
    trust a tag; the turn row is where the scope is authoritative, and it is
    the same table the demo chart reads.
    """
    async with db.agent() as conn:
        cur = await conn.execute(
            "SELECT id, trace_id FROM turn "
            "WHERE connection_id = %s AND trace_id IS NOT NULL AND created_at >= %s",
            (connection_id, since),
        )
        rows = await cur.fetchall()
    return {r["trace_id"]: r["id"] for r in rows}
