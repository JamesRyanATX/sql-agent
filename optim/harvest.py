"""Turn recorded `extract` calls into a replayable corpus.

**Langfuse is the record, and this reads only Langfuse.** That is a correction
rather than a convenience. The first version asked Postgres which warehouse each
recorded call was about, by joining `turn.trace_id` — and `make reset` empties
the `turn` table by design, while the trace store keeps everything forever. So a
reset silently turned every recorded call into debris: the data was intact, and
nothing could prove whose it was. Thirteen of fifteen, the first time it
happened.

A reset should reset. What was wrong was depending on a table whose whole job is
to be emptied. The scope comes from the turn span instead, which records
`connection_id` in its own input — and, since `prompts.fingerprint()` went on
it, which prose produced the turn.

Worth knowing for later: Langfuse has the *inputs*, Postgres has the
*outcomes* — whether the SQL errored, how many fix attempts — and only Postgres
gets reset. `extract` does not care, because its score comes entirely from the
recorded call. A `plan` harvest would care a lot: labelling "should the cache
have been enough?" means looking at what happened next.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app import tracing
from optim.cases import ExtractCase, filed_from, sql_from
from optim.replay import NODE as REPLAY_NODE


@dataclass
class Harvest:
    """What came back, and what did not. Printed rather than returned quietly."""

    cases: list[ExtractCase] = field(default_factory=list)
    seen: int = 0
    no_message: int = 0
    no_sql: int = 0
    unscoped: int = 0
    other_connection: int = 0
    contaminated: int = 0
    duplicate: int = 0

    def report(self) -> str:
        lines = [f"{len(self.cases)} cases from {self.seen} recorded extract calls"]
        dropped = [
            ("no user message", self.no_message),
            ("SQL would not parse", self.no_sql),
            ("no turn span, so no way to say which warehouse", self.unscoped),
            ("another connection", self.other_connection),
            ("written by the harness itself", self.contaminated),
            ("identical to a case already kept", self.duplicate),
        ]
        if any(n for _, n in dropped):
            lines.append(f"  dropped {self.seen - len(self.cases)}:")
            lines += [f"    {n} {label}" for label, n in dropped if n]
        return "\n".join(lines)


def extract_cases(
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
    scope = turn_scope(since=since)

    harvest = Harvest()
    seen_messages: set[str] = set()

    for observation in tracing.observations(name="extract", since=since):
        harvest.seen += 1

        # `replay` writes under `extract.replay`, so the harness's own calls
        # should never reach a corpus — but a name filter is one typo from being
        # wrong about that, and being wrong means round two trains on round
        # one's output. Cheap to assert twice.
        if observation.get("name") == REPLAY_NODE:
            harvest.contaminated += 1
            continue

        trace_id = observation.get("trace_id") or ""
        scoped = scope.get(trace_id)
        if scoped is None:
            harvest.unscoped += 1
            continue
        if scoped.connection_id != connection_id:
            harvest.other_connection += 1
            continue

        message = _user_message(observation.get("input"))
        if not message:
            harvest.no_message += 1
            continue

        sql = sql_from(message)
        if not sql:
            harvest.no_sql += 1
            continue

        # T2-style repeats produce byte-identical extract inputs, and a corpus
        # holding the same case five times weights it five times — tuning the
        # prompt for whichever question the demo happens to ask twice.
        if message in seen_messages:
            harvest.duplicate += 1
            continue
        seen_messages.add(message)

        harvest.cases.append(
            ExtractCase(
                name=_label(trace_id, message),
                user_message=message,
                sql=sql,
                filed=filed_from(message),
                obs_id=observation.get("id"),
                trace_id=trace_id,
                connection_id=connection_id,
                prompt_fp=scoped.prompt_fp,
                baseline_tokens_out=_tokens_out(observation),
            )
        )

    return harvest


@dataclass(frozen=True)
class Scope:
    connection_id: str | None
    prompt_fp: str | None


def turn_scope(*, since: datetime | None = None) -> dict[str, Scope]:
    """trace_id -> which warehouse the turn was about, and under which prose.

    The turn span is the only place that says so. `migrations/004_tracing.sql`
    called the trace "another system entirely, on the other side of an HTTP
    exporter, and there is nothing here to join it to" — that is still true, and
    the answer turned out to be not to join at all.
    """
    scope: dict[str, Scope] = {}
    for span in tracing.observations(name="turn", kind="SPAN", since=since):
        prompts = (span.get("metadata") or {}).get("prompts") or {}
        scope[span.get("trace_id") or ""] = Scope(
            connection_id=(span.get("input") or {}).get("connection_id"),
            prompt_fp=prompts.get("extract"),
        )
    return scope


def _tokens_out(observation: dict) -> int:
    """What the recorded call cost, which the metric scores a candidate against.

    Zero when Langfuse has no usage for the call, and the cost term reads zero
    as "no baseline, do not score cost" rather than as "free" — so losing this
    does not fail, it quietly deletes a fifth of the metric. It was lost once
    exactly that way.
    """
    usage = observation.get("usage") or {}
    return int(usage.get("output") or 0)


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


def _label(trace_id: str, message: str) -> str:
    """A short readable id. The trace prefix keeps it unique, the question keeps
    it meaningful in a report someone is reading at speed."""
    question = message.removeprefix("Question: ").split("\n", 1)[0]
    slug = re.sub(r"[^a-z0-9]+", "-", question.casefold()).strip("-")[:36]
    return f"{trace_id[:8]}-{slug}" if slug else trace_id[:8]
