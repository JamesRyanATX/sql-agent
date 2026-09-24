"""The one module that knows Langfuse exists.

`import langfuse` appears here and nowhere else. The rest of `app/` sees three
context managers — `turn`, `generation`, `span` — plus `score` and `score_text`
to write a verdict, `observations()` and `scores()` to read everything back,
and a boolean. A tool observation is a `span` with `as_type="tool"`.

**Off is the default, and off is free.** With no keys set no client is
constructed, `langfuse` is never imported, and every helper yields the same
do-nothing handle, so no call site branches on it.

**On captures the prompts, the SQL and the result rows.** The store is
self-hosted, so nothing leaves the machine — but it holds whatever the
registered warehouse holds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app import prompts
from app.settings import settings

log = logging.getLogger(__name__)

# `_broken` is separate from `_client is None` so a client that failed to build
# is not retried on every model call for the rest of the process's life.
_client: Any = None
_broken = False


def enabled() -> bool:
    """Both keys, or nothing. No third flag that could disagree with them."""
    s = settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


def client() -> Any:
    """The Langfuse client, or None when tracing is off."""
    global _client, _broken
    if not enabled() or _broken:
        return None
    if _client is None:
        from langfuse import Langfuse

        s = settings()
        try:
            # Explicit rather than the SDK's own env lookup: `.env` is read by
            # pydantic-settings and never reaches os.environ on the host, so
            # implicit resolution works in the container and not outside it.
            _client = Langfuse(
                public_key=s.langfuse_public_key,
                secret_key=s.langfuse_secret_key,
                host=s.langfuse_host,
            )
        except Exception as e:  # pragma: no cover — construction barely fails
            _broken = True
            log.warning("tracing disabled: could not build the client (%s)", type(e).__name__)
    return _client


def shutdown() -> None:
    """Flush what is buffered and stop the exporter thread.

    Called from the lifespan's `finally`, which the suite runs on every test —
    so it has to be cheap and idempotent when tracing was never on.
    """
    global _client, _broken
    _broken = False
    if _client is not None:
        try:
            _client.shutdown()
        except Exception as e:  # pragma: no cover
            log.warning("could not flush traces (%s)", type(e).__name__)
        _client = None


def score(
    *,
    trace_id: str,
    name: str,
    value: float,
    comment: str | None = None,
) -> bool:
    """Attach a verdict to a turn that has already finished.

    False when tracing is off, which is how a caller tells "recorded" from
    "nowhere to record it" without importing anything from here.

    `create_score` enqueues rather than sends, and Langfuse's own docstring
    warns that ingestion is asynchronous — a score is delivered at flush and
    readable some seconds after that. So True means accepted, not stored.

    No try/except: `create_score` wraps its whole body in one and logs, so a
    second here would be unreachable. It is worth saying, because the absence
    reads as an oversight beside `client()`, which does catch.
    """
    lf = client()
    if lf is None:
        return False
    lf.create_score(
        name=name,
        value=value,
        trace_id=trace_id,
        # 1 or 0 and nothing else; the API rejects any other value at this type.
        data_type="BOOLEAN",
        comment=comment,
    )
    return True


def score_text(
    *,
    trace_id: str,
    name: str,
    value: str,
    comment: str | None = None,
) -> bool:
    """Attach a piece of text to a turn that has already finished.

    The reference query is the case: `make corpus` files the golden query on
    every turn it asks, and a person files a corrected one from the Langfuse
    UI, in the same field. A TEXT score carries a string where `score()`
    carries a 1 or a 0; everything else about it is the same, including that
    True means accepted, not stored.
    """
    lf = client()
    if lf is None:
        return False
    lf.create_score(
        name=name,
        value=value,
        trace_id=trace_id,
        data_type="TEXT",
        comment=comment,
    )
    return True


class _Null:
    """What every helper yields when tracing is off, so callers never branch."""

    __slots__ = ()

    trace_id: str | None = None

    def update(self, **_: Any) -> None:
        pass


_NULL = _Null()


def _serialisable(obj: Any) -> Any:
    """Anything, as plain JSON types.

    Required: the Anthropic path echoes raw SDK content blocks back through
    `llm.assistant_turn()`, so a message list is not made of dicts.
    """
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return str(obj)


@contextmanager
def turn(*, session_id: str, question: str) -> Iterator[Any]:
    """One turn: the root span every generation and tool call hangs off.

    `.trace_id` is minted here rather than read out of the ambient context inside
    a node, so writing it to the turn row does not depend on OpenTelemetry
    context reaching LangGraph's tasks.

    `session_id` is the agent's session, LangGraph's `thread_id` and Langfuse's
    session at once — one identifier throughout.
    """
    lf = client()
    if lf is None:
        yield _NULL
        return

    from langfuse import propagate_attributes

    trace_id = lf.create_trace_id()
    with lf.start_as_current_observation(
        name="turn",
        as_type="span",
        trace_context={"trace_id": trace_id},
        input=_serialisable({"question": question}),
        # Which prose produced this turn, so a harvest can tell a run under the
        # seed from a run under a candidate. Eight hex characters per node is
        # 124 for the whole dict, under the 200 a metadata value truncates at —
        # a longer hash would lose its tail and compare equal every time.
        metadata=_serialisable({"prompts": prompts.fingerprint()}),
    ) as span:
        with propagate_attributes(
            session_id=str(session_id),
            trace_name="turn",
        ):
            yield span


@contextmanager
def generation(
    *,
    name: str,
    model: str,
    input: Any = None,
    metadata: Any = None,
) -> Iterator[Any]:
    """One model call. Opened in exactly one place — `llm.complete()`."""
    lf = client()
    if lf is None:
        yield _NULL
        return

    with lf.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        input=_serialisable(input),
        metadata=_serialisable(metadata),
    ) as gen:
        try:
            yield gen
        except Exception as e:
            # A refusal and a 400 from a local server are successful-looking
            # control flow elsewhere; on the trace they must read as failures.
            gen.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
            raise


# --------------------------------------------------------------- the read half
#
# Everything above writes; this reads. Here rather than in `tools/gepa/`, because
# `import langfuse` belongs to one module. tests/test_cli_isolation.py enforces it.


def observations(
    *,
    name: str,
    kind: str = "GENERATION",
    since: Any = None,
    until: Any = None,
    page: int = 100,
) -> Iterator[dict[str, Any]]:
    """Everything recorded under one observation name, oldest page first.

    Two shapes are used. `kind="GENERATION"` with a node name is `tools/gepa/`'s
    per-node dataset of exact inputs and outputs. `name="turn", kind="SPAN"` is
    what says which prose produced a recorded call — its metadata carries the
    prompt fingerprints, which cannot come from a join to `turn.trace_id`
    because `make reset` empties that table by design.

    Yields nothing when tracing is off, which is the common case.
    """
    lf = client()
    if lf is None:
        return

    cursor = None
    while True:
        # `fields` defaults to core,basic, where input and output are absent and
        # the call still succeeds — the symptom is a corpus of empty prompts.
        # `parse_io_as_json` returns 400 if set at all, so input arrives as a raw
        # string and `_as_json` below is required rather than defensive.
        response = lf.api.observations.get_many(
            name=name,
            type=kind,
            from_start_time=since,
            to_start_time=until,
            fields="core,basic,io,metadata,usage",
            limit=page,
            cursor=cursor,
        )
        for observation in response.data:
            yield {
                "id": observation.id,
                "trace_id": observation.trace_id,
                "name": observation.name,
                "start_time": observation.start_time,
                "level": observation.level,
                "input": _as_json(observation.input),
                "output": _as_json(observation.output),
                "metadata": _as_json(observation.metadata),
                "usage": observation.usage_details or {},
            }
        cursor = response.meta.cursor
        if not cursor:
            return


def scores(
    *,
    name: str,
    since: Any = None,
    page: int = 100,
) -> Iterator[dict[str, Any]]:
    """Every score filed under one name, oldest page first.

    The other half of `score()` and `score_text()`. A verdict or a reference
    query lands on a trace, and this is how a harvest reads it back: `value`
    for a BOOLEAN or NUMERIC score (a 1.0 or a 0.0), `string_value` for a TEXT
    or CATEGORICAL one, `comment` for either. Through the v3 endpoint: the v2
    one is gone on a Langfuse v4 server, which is what `make langfuse-up`
    runs, and it answers 404 with a sentence saying so. A score names what it
    is on as a `subject`; only one on a trace, or on an observation inside a
    trace, has a trace id to join on.

    Yields nothing when tracing is off.
    """
    lf = client()
    if lf is None:
        return

    cursor = None
    while True:
        # Like `observations()`: the core fields come alone unless the groups
        # are named. `subject` is the trace the score is on, `details` its
        # comment; without them a verdict reads back as a bare 1 or 0 on
        # nothing. An unknown group name is a 400.
        response = lf.api.scores_v3.get_many_v3(
            name=name,
            from_timestamp=since,
            fields="details,subject",
            limit=page,
            cursor=cursor,
        )
        for row in response.data:
            value = getattr(row, "value", None)
            yield {
                "trace_id": _trace_of(row),
                "name": row.name,
                "value": float(value) if isinstance(value, (bool, int, float)) else None,
                "string_value": value if isinstance(value, str) else None,
                "comment": row.comment,
                "timestamp": row.timestamp,
                "source": getattr(row, "source", None),
            }
        cursor = response.meta.cursor
        if not cursor:
            return


def _trace_of(row: Any) -> str | None:
    """The trace a score is on, from its subject. A session or an experiment
    score has none, and a harvest keyed on turns cannot use it."""
    subject = getattr(row, "subject", None)
    if subject is None:
        return None
    kind = getattr(subject, "kind", None)
    if kind == "trace":
        return subject.id
    if kind == "observation":
        return getattr(subject, "trace_id", None)
    return None


def _as_json(value: Any) -> Any:
    """Input and output come back as raw strings. Decode, or hand back as-is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


@contextmanager
def span(
    *,
    name: str,
    input: Any = None,
    metadata: Any = None,
    as_type: str = "span",
) -> Iterator[Any]:
    """Anything that is not a model call: a graph node, a tool, a query."""
    lf = client()
    if lf is None:
        yield _NULL
        return

    with lf.start_as_current_observation(
        name=name,
        as_type=as_type,
        input=_serialisable(input),
        metadata=_serialisable(metadata),
    ) as sp:
        try:
            yield sp
        except Exception as e:
            sp.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
            raise
