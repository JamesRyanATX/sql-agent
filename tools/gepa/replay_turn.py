"""Run one whole turn under a candidate's text, and report what happened.

`replay.py` scores one `extract` call, which is nearly a pure function of three
recorded strings and needs no database. Tool descriptions and per-node effort
are not like that: what they change is the *shape of a turn* — how many tools
were called, in what order, whether the SQL ran first time. Nothing smaller than
a whole turn can see it.

That makes this the expensive unit, ~11.5k tokens a rollout, and everything here
exists to make sure the tokens buy a measurement rather than a confound:

**Cold every rollout.** The cache is cleared first, so no rollout is answering
from what the last one learned. Tool descriptions only matter on the cold path —
a warm turn never calls a tool — so a shared cache would silently make half the
candidates unmeasurable.

**One scratch connection per concurrent rollout.** Learned state is keyed by
connection id, so two rollouts sharing one would share a cache and clear it
under each other. `default` is never used: it is the demo's warehouse, and its
turn log is a chart somebody presents.

**The graph is driven directly, not through `stream_turn`.** That wrapper opens
a trace span named `turn`, which is the name a later harvest filters on. A
thousand rollouts under that name would become the next round's training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app import db, graph, overrides, store


@dataclass
class ToolCall:
    """One introspection call the explore loop made."""

    name: str
    args: dict[str, Any]
    error: bool = False


@dataclass
class TurnReplayed:
    """What one candidate did with one question.

    The same bargain `replay.Replayed` makes: a rollout that fell over sets
    `error` and is scored zero, rather than ending a run that has already spent
    an hour.
    """

    question: str
    connection_id: str
    answer: str = ""
    sql: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    tools: list[ToolCall] = field(default_factory=list)
    explored: bool = False
    fix_attempts: int = 0
    # The database's complaint about the last failed query, if the turn ended
    # still failing. Distinct from `error`, which means the harness broke.
    sql_error: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def tool_names(self) -> list[str]:
        """The sequence, for feedback prose. Repetition is the signal: three
        `sample_column` calls on one column is a description that did not say
        what the tool was for."""
        return [t.name for t in self.tools]


async def replay_turn(
    question: str,
    *,
    connection_id: str,
    candidate: overrides.Overrides,
) -> TurnReplayed:
    """One cold turn, under this candidate's prompts, tools and efforts.

    The override is set **inside** this coroutine on purpose. `asyncio` copies
    the current context when it creates a task, so a value set here reaches
    every graph node; a value set by a synchronous caller would not survive the
    hop into the optimiser's event-loop thread. Setting it here is also what
    lets rollouts run in parallel without reading each other's candidate.
    """
    replayed = TurnReplayed(question=question, connection_id=connection_id)
    try:
        async with db.agent() as conn:
            await store.reset_learned(conn, connection_id=connection_id)

        with overrides.using(candidate):
            await _drive(question, connection_id, replayed)
    except Exception as e:
        replayed.error = f"{type(e).__name__}: {e}"
    return replayed


async def _drive(question: str, cid: str, out: TurnReplayed) -> None:
    """Run the graph and fold its events into the record.

    No checkpointer: a rollout is one turn and is never resumed, and the
    checkpoint rows would be state to clean up between rollouts. `custom` events
    only — the same stream the CLI renders, so what is measured is what a user
    would have seen.
    """
    compiled = graph.build_graph()
    session = str(uuid4())
    async for mode, chunk in compiled.astream(
        {"session_id": session, "question": question, "connection_id": cid},
        stream_mode=["custom"],
        config={"configurable": {"thread_id": session}},
    ):
        if isinstance(chunk, dict):
            _fold(chunk, out)


def _fold(event: dict[str, Any], out: TurnReplayed) -> None:
    """One event into the record. Unknown types are ignored rather than an
    error: a new event on the stream is not a broken rollout."""
    kind = event.get("type")
    if kind == "explore":
        out.tools.append(
            ToolCall(
                name=event.get("tool", ""),
                args=event.get("input") or {},
                error=bool(event.get("error")),
            )
        )
    elif kind == "sql":
        out.sql = event.get("sql", "")
    elif kind == "fix":
        # The attempt number, not a count of events: `fix` can emit more than
        # once and the node itself is what knows which attempt this was.
        out.fix_attempts = max(out.fix_attempts, int(event.get("attempt", 0)))
        out.sql = event.get("sql", out.sql)
    elif kind == "rows":
        out.rows = event.get("rows") or []
        out.sql_error = ""
    elif kind == "error":
        out.sql_error = event.get("message", "")
    elif kind == "answer":
        out.answer = event.get("text", "")
        out.explored = bool(event.get("explored"))
        out.tokens_in = int(event.get("tokens_in", 0))
        out.tokens_out = int(event.get("tokens_out", 0))


# ------------------------------------------------------------ scratch warehouses


def scratch_id(slot: int) -> str:
    """The connection id for one rollout slot. `gepa-0`, `gepa-1`, …"""
    return f"gepa-{slot}"


async def ensure_scratch(slot: int, *, url: str) -> str:
    """Register the scratch connection for a slot, if it is not already there.

    Points at the same database the demo connection does, and carries its own
    address (`origin="api"`) rather than reading the environment, so it survives
    a `TARGET_DATABASE_URL` that changes mid-run. Its cache starts empty because
    learned state is keyed by connection id — which is most of why a rollout
    gets a scratch id at all rather than borrowing `default` and tidying up.
    """
    cid = scratch_id(slot)
    async with db.agent() as conn:
        if await store.get_connection(conn, cid) is not None:
            return cid
        await store.create_connection(
            conn, store.connection_from_url(url, id=cid, origin="api")
        )
    return cid
