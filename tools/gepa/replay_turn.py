"""Run one whole turn under a candidate's text, and report what happened.

`replay.py` scores one `extract` call, which is nearly a pure function of three
recorded strings and needs no database. Tool descriptions and per-node effort
are not like that: what they change is the *shape of a turn* — how many tools
were called, in what order, whether the SQL ran first time. Nothing smaller than
a whole turn can see it.

That makes this the expensive unit, ~11.5k tokens a rollout, and everything here
exists to make sure the tokens buy a measurement rather than a confound:

**Cold every time, and nothing kept.** The question is asked with memory off:
the turn ignores what the agent has learned and saves nothing it learns. Tool
descriptions only matter on the cold path — a warm turn never calls a tool — so
a candidate measured against a warm cache is not measured at all.

Nothing is *cleared* to achieve that, which is what makes it safe. Your memory
is as it was when the run finishes, and two questions asked at once cannot wipe
each other halfway through.

**The graph is driven directly, not through `stream_turn`.** That wrapper opens
a trace span named `turn`, which is the name a later harvest filters on. A
thousand rollouts under that name would become the next round's training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app import db, graph, overrides, prompts
from app.config import config


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
    # Where the tokens went: node -> (in, out). The totals above are read off
    # the answer event; these are read off each node's own returned delta, so
    # the two are independent records that have to agree, and a test says so.
    per_node: dict[str, tuple[int, int]] = field(default_factory=dict)
    # The effort each model-calling node ran at, resolved with the candidate in
    # force. Recorded rather than recomputed later: what a rollout ran at is a
    # fact about that rollout, and the candidate is gone by the time the
    # feedback is read.
    efforts: dict[str, str] = field(default_factory=dict)
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
    question: str, *, candidate: overrides.Overrides
) -> TurnReplayed:
    """One cold turn, under this candidate's prompts, tools and efforts.

    The override is set **inside** this coroutine on purpose. `asyncio` copies
    the current context when it creates a task, so a value set here reaches
    every graph node; a value set by a synchronous caller would not survive the
    hop into the optimiser's event-loop thread. Setting it here is also what
    lets rollouts run in parallel without reading each other's candidate.
    """
    replayed = TurnReplayed(question=question)
    try:
        with overrides.using(candidate):
            replayed.efforts = _efforts_in_force()
            await _drive(question, replayed)
    except Exception as e:
        replayed.error = f"{type(e).__name__}: {e}"
    return replayed


def _efforts_in_force() -> dict[str, str]:
    """What each model-calling node would run at, under whatever is in force.

    Through `Config.effort_for`, so a candidate's override, the file's block and
    the default all resolve the way the node itself will resolve them a moment
    later. It refuses `none` on a Claude model exactly as the node would, one
    call earlier, which is what turns that candidate into a scored zero rather
    than a rollout that got halfway.
    """
    return {node: config().effort_for(node) for node in prompts.NODES}


async def _drive(question: str, out: TurnReplayed) -> None:
    """Run the graph and fold its events into the record.

    No checkpointer: one turn, never resumed, and the checkpoint rows would be
    state to clean up afterwards. Two streams: `custom` is what the CLI renders,
    so what is measured is what a user would have seen; `updates` is each node's
    returned delta, which is the only place a node's own token count exists
    before the reducer sums it into the turn.

    `memory: False` is the whole point: the turn neither reads the cache nor
    saves to it, so it behaves like a first-ever question every time.

    The agent pool is opened here rather than assumed. A turn writes to the turn
    log whatever its memory setting, so it needs one, and the only other callers
    of `open_pools` are the server's startup and a test fixture — which is how
    this went missing: every test had the fixture, and `make gepa-tools` scored
    nineteen rollouts zero in 1.6 seconds. It is idempotent.
    """
    await db.open_pools()
    compiled = graph.build_graph()
    session = str(uuid4())
    async for mode, chunk in compiled.astream(
        {"session_id": session, "question": question, "memory": False},
        stream_mode=["custom", "updates"],
        config={"configurable": {"thread_id": session}},
    ):
        if not isinstance(chunk, dict):
            continue
        if mode == "updates":
            _fold_update(chunk, out)
        else:
            _fold(chunk, out)


def _fold_update(update: dict[str, Any], out: TurnReplayed) -> None:
    """One node's returned delta into the per-node ledger.

    LangGraph hands these as `{node: delta}`. Only the two token keys are read,
    and only where the node returned them: `load_cache` and `execute` make no
    model call and say nothing, and their absence from the ledger is the right
    record of that. `fix` can run more than once and accumulates.
    """
    for node, delta in update.items():
        if not isinstance(delta, dict):
            continue
        tin = int(delta.get("tokens_in") or 0)
        tout = int(delta.get("tokens_out") or 0)
        if not (tin or tout):
            continue
        was_in, was_out = out.per_node.get(node, (0, 0))
        out.per_node[node] = (was_in + tin, was_out + tout)


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
