"""Observability, and the invariant that it costs nothing when it is off.

Two things are worth testing without a Langfuse container running. First, that
*off* is genuinely off: no client, no import, no branch anywhere else in `app/`
that has to remember to check. Second, that the seams are wired — that
`llm.complete()` really does open a generation, and that what it reports matches
what the model actually returned.

No network and no container: `tracing.generation` is replaced by a recorder, and
the model by the same `httpx.MockTransport` harness test_llm_openai.py uses.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import httpx
import pytest

from app import llm, store, tracing
from app.settings import settings

# The scripted-model harness, rather than a second copy of it: what a cold turn
# does is test_coldpath.py's subject, and this file only adds "and it is traced".
from tests.test_coldpath import (  # noqa: F401 — `pool` is a fixture
    ScriptedModel,
    json_result,
    no_entries,
    pool,
    run,
    text_result,
    tool_result,
)
from tests.test_llm_openai import capture, use_openai

DEFAULT_CONNECTION = "default"


@contextmanager
def keys_set(monkeypatch):
    """A configured-looking environment, undone on the way out.

    The `settings()` lru_cache is process-wide, so a test that changes a setting
    and does not clear it on the way out changes every test after it — the same
    try/finally test_llm_openai.py and test_connections.py use.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    settings.cache_clear()
    try:
        yield
    finally:
        monkeypatch.undo()
        settings.cache_clear()
        # The client is a module singleton keyed on nothing, so it would outlive
        # the settings that built it.
        tracing.shutdown()


# ------------------------------------------------------------------------ off


def test_tracing_is_off_and_builds_nothing():
    """The default, and the reason this file exists.

    `make test`, `make up` and the demo all run here. If any of these three
    assertions stops holding, tracing has started costing something on a path
    that never asked for it.
    """
    assert not tracing.enabled()
    assert tracing.client() is None
    assert tracing._client is None


def test_the_helpers_are_no_ops_when_off():
    """Callers never branch on whether tracing is on, so the handle has to
    absorb every keyword a real one takes."""
    with tracing.turn(session_id="s", question="q", connection_id="default") as t:
        assert t.trace_id is None
        t.update(output={"anything": 1}, level="ERROR", status_message="x")

    with tracing.generation(name="plan", model="m", input={}, metadata={}) as g:
        g.update(output="x", usage_details={"input": 1}, metadata={"a": "b"})

    with tracing.span(name="tool.list_tables", input={}, as_type="tool") as s:
        s.update(output="[]")

    assert tracing._client is None


def test_one_key_is_not_enough(monkeypatch):
    """Half-configured is off, not half-on.

    `Settings` has extra="ignore", so a mistyped key name is silently dropped
    and leaves exactly this state. The lifespan warns about it; `enabled()` is
    what makes the warning true.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    settings.cache_clear()
    try:
        assert not tracing.enabled()
    finally:
        monkeypatch.undo()
        settings.cache_clear()


def test_shutdown_is_safe_when_tracing_never_started():
    """The `client` fixture runs the lifespan on every test in the suite, and
    the lifespan's `finally` calls this."""
    tracing.shutdown()
    tracing.shutdown()
    assert tracing._client is None


# ------------------------------------------------------------- the model seam


class Recorder:
    """Stands in for `tracing.generation`, keeping what it was told."""

    def __init__(self) -> None:
        self.opened: list[dict] = []
        self.updated: list[dict] = []

    @contextmanager
    def __call__(self, **kwargs):
        self.opened.append(kwargs)
        recorder = self

        class _H:
            def update(self, **kw):
                recorder.updated.append(kw)

        yield _H()


OK = {
    "choices": [{"message": {"content": "the answer", "tool_calls": []}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 22},
}


async def test_a_model_call_opens_a_generation(monkeypatch):
    """One seam for both backends.

    `complete()` is already the only function that talks to a model, which is
    what makes it the only place a generation has to be opened — including every
    iteration of the explore loop, whose per-call cost the node sums away before
    anything downstream can see it.
    """
    rec = Recorder()
    monkeypatch.setattr(tracing, "generation", rec)
    use_openai(monkeypatch, model="qwen-test")
    try:
        capture(monkeypatch, httpx.Response(200, json=OK))
        result = await llm.complete(
            system="s",
            messages=[{"role": "user", "content": "q"}],
            effort="low",
            node="plan",
        )
    finally:
        monkeypatch.undo()

    assert len(rec.opened) == 1
    opened = rec.opened[0]
    # Named for the node, so a trace reads as the graph rather than as seven
    # anonymous calls to the same function.
    assert opened["name"] == "plan"
    assert opened["model"] == "qwen-test"
    assert opened["metadata"]["provider"] == "openai_compat"
    assert opened["metadata"]["effort"] == "low"
    assert opened["metadata"]["structured"] is False

    assert len(rec.updated) == 1
    usage = rec.updated[0]["usage_details"]
    # The numbers on the span are the numbers the server reported, not a second
    # count of our own that could drift from the turn table's.
    assert usage["input"] == result.tokens_in == 11
    assert usage["output"] == result.tokens_out == 22
    assert usage["cache_read_input_tokens"] == result.cache_read


async def test_a_failed_model_call_is_marked_on_its_generation(monkeypatch):
    """An `LlmError` is a 400 from a local server, which is control flow
    everywhere else in this codebase. On a trace it has to read as a failure."""
    rec = Recorder()
    monkeypatch.setattr(tracing, "generation", rec)
    use_openai(monkeypatch)
    try:
        capture(monkeypatch, httpx.Response(400, text="no such field"))
        with pytest.raises(llm.LlmError):
            await llm.complete(
                system="s", messages=[], effort="low", node="generate_sql"
            )
    finally:
        monkeypatch.undo()

    # The recorder is not the real context manager, so it cannot mark anything
    # itself — what this pins is that the exception propagates out of the `with`
    # rather than being swallowed by the wrapper.
    assert rec.opened[0]["name"] == "generate_sql"
    assert rec.updated == []


# ------------------------------------------------------------------ the turn


async def test_a_turn_span_wraps_the_stream_and_carries_the_trace_id(monkeypatch):
    """The trace id is minted outside the graph and handed in as state.

    Not read back out of the ambient OpenTelemetry context inside `answer`: the
    turn row's `trace_id` then depends on that context reaching LangGraph's
    tasks, which is a different thing to get right and fails silently when it
    is wrong.
    """
    from app import graph

    seen: dict = {}

    class FakeCompiled:
        async def astream(self, state, **kwargs):
            seen["state"] = state
            seen["kwargs"] = kwargs
            yield "custom", {"type": "answer", "text": "1,840", "total_tokens": 7}

    events = [
        e
        async for e in graph.stream_turn(
            FakeCompiled(), "session-1", "how many customers?", DEFAULT_CONNECTION
        )
    ]

    assert [e["type"] for e in events] == ["answer", "done"]
    # Off, so there is no trace to point at — and "" rather than a missing key,
    # because `answer` turns it back into NULL.
    assert seen["state"]["trace_id"] == ""
    assert seen["kwargs"]["config"] == {"configurable": {"thread_id": "session-1"}}

    with keys_set(monkeypatch):
        [
            e
            async for e in graph.stream_turn(
                FakeCompiled(), "session-2", "q", DEFAULT_CONNECTION
            )
        ]
    # A 32-character hex trace id, the same one the turn row will hold.
    assert len(seen["state"]["trace_id"]) == 32
    int(seen["state"]["trace_id"], 16)


async def test_stream_turn_still_never_raises_with_tracing_on(monkeypatch):
    """The span sits outside the try, so opening it must not change the promise
    the rest of the demo rests on."""
    from app import graph

    class Exploding:
        async def astream(self, state, **kwargs):
            raise TimeoutError("model took too long")
            yield  # pragma: no cover — makes this an async generator

    with keys_set(monkeypatch):
        events = [
            e
            async for e in graph.stream_turn(
                Exploding(), "session-3", "q", DEFAULT_CONNECTION
            )
        ]

    assert [e["type"] for e in events] == ["error", "done"]
    assert events[0]["fatal"] is True


async def test_a_cold_turn_traces_its_nodes_its_tools_and_its_query(pool, monkeypatch):
    """The shape a T1 trace has, without spending a T1's worth of tokens.

    The interesting one: `explore` makes up to 24 tool calls inside a single node
    visit, and the turn table records that as the integer 24. A span each is the
    only place the individual calls exist.
    """
    opened: list[tuple[str, Any]] = []

    @contextmanager
    def recorder(*, name, input=None, metadata=None, as_type="span"):
        opened.append((name, input))

        class _H:
            def update(self, **kw):
                pass

        yield _H()

    monkeypatch.setattr(tracing, "span", recorder)
    monkeypatch.setattr(
        llm,
        "complete",
        ScriptedModel(
            tool_result("list_tables", {}),
            tool_result("sample_column", {"table": "customer", "column": "deleted_at"}),
            text_result("customer holds 2000 rows; deleted_at marks soft deletes."),
            json_result(
                {
                    "sql": "SELECT count(*) AS n FROM customer WHERE deleted_at IS NULL",
                    "assumptions": ["excludes soft-deleted rows"],
                }
            ),
            no_entries(),
            text_result("1,840 customers, excluding soft-deleted."),
        ),
    )

    await run()

    names = [n for n, _ in opened]
    # Every node it went through, in the order the graph runs them.
    assert names == [
        "load_cache",
        "plan",
        "explore",
        "tool.list_tables",
        "tool.sample_column",
        "generate_sql",
        "execute",
        "sql.execute",
        "extract",
        "answer",
    ]
    # And the query itself is on the span, not just the fact that one ran.
    sql = dict(opened)["sql.execute"]
    assert "deleted_at IS NULL" in sql


# ------------------------------------------------------------- the turn table


async def test_a_turns_trace_id_round_trips(agent_conn):
    """`make turns` shows a turn that cost 11,500 tokens; this is the column
    that gets you from that row to the calls that spent them."""
    turn_id = await store.start_turn(
        agent_conn,
        connection_id=DEFAULT_CONNECTION,
        session_id=uuid4(),
        question="how many customers?",
    )
    await store.finish_turn(
        agent_conn, turn_id, answer="1,840", trace_id="0123456789abcdef" * 2
    )

    rows = await store.read_turns(agent_conn, connection_id=DEFAULT_CONNECTION)
    row = next(r for r in rows if r["id"] == turn_id)
    assert row["trace_id"] == "0123456789abcdef" * 2


async def test_a_turn_taken_without_tracing_has_no_trace_id(agent_conn):
    """NULL means "not recorded", which is what is true — and is the common
    case, since tracing is off unless both keys are set."""
    turn_id = await store.start_turn(
        agent_conn,
        connection_id=DEFAULT_CONNECTION,
        session_id=uuid4(),
        question="how many customers?",
    )
    await store.finish_turn(agent_conn, turn_id, answer="1,840")

    rows = await store.read_turns(agent_conn, connection_id=DEFAULT_CONNECTION)
    row = next(r for r in rows if r["id"] == turn_id)
    assert row["trace_id"] is None
