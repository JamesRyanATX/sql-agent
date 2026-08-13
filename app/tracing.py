"""The one module that knows Langfuse exists.

Same rule [app/dialects.py](app/dialects.py) has for dialect names and
[app/secrets.py](app/secrets.py) has for encryption: `import langfuse` appears
here and nowhere else, so the rest of `app/` sees three context managers, a
reader and a boolean. That is what makes "tracing off" cost nothing to prove —
there is one `enabled()` to read rather than a scattering of `if` at every call
site. (`turn`, `generation` and `span`; a tool observation is `span` with
`as_type="tool"`, which is why the trace diagram in CLAUDE.md shows four kinds.)

The reader is newer than the rest and inverts the module's old one-way
character: `observations()` pulls recorded calls back out, which is what
lets `optim/` build a prompt corpus from what the agent has already done. It
lives here rather than where it is wanted for exactly the reason above.

**Off is the default, and off must be free.** With no keys set no client is ever
constructed, `langfuse` is never even imported, and every helper yields the same
do-nothing handle. `make up`, `make test` and the demo behave exactly as they did
before this file existed.

**On means the prompts, the SQL and the result rows are captured**, because that
is the entire point of looking at a trace and there is no half-measure that stays
useful. The trace store is self-hosted (`make langfuse-up`), so "captured" means
written to a container on this machine — but it holds whatever the registered
warehouse holds, and that is a fact worth knowing before pointing a connection at
a real one.
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

# Lazily constructed on first use, like llm._anthropic(). `_broken` is separate
# from `_client is None` so a client that failed to build is not retried on
# every model call for the rest of the process's life.
_client: Any = None
_broken = False


def enabled() -> bool:
    """Both keys, or nothing.

    Deliberately not a separate `LANGFUSE_TRACING_ENABLED` flag: a flag is a
    third state that can disagree with the keys, and "enabled but unconfigured"
    is a startup warning nobody reads.
    """
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
            # Credentials passed explicitly rather than left to the SDK's own
            # environment lookup. `.env` is read by pydantic-settings and never
            # reaches os.environ on the host, so implicit resolution would work
            # inside the container and silently not work outside it.
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

    Called from the lifespan's `finally`. The suite's `client` fixture runs that
    lifespan on every test, so this also has to be cheap and idempotent when
    tracing was never on.
    """
    global _client, _broken
    _broken = False
    if _client is not None:
        try:
            _client.shutdown()
        except Exception as e:  # pragma: no cover
            log.warning("could not flush traces (%s)", type(e).__name__)
        _client = None


class _Null:
    """What every helper yields when tracing is off.

    A real handle's `update()` takes only keywords, so ignoring them all is the
    whole implementation. Callers never branch on whether tracing is on.
    """

    __slots__ = ()

    trace_id: str | None = None

    def update(self, **_: Any) -> None:
        pass


_NULL = _Null()


def _serialisable(obj: Any) -> Any:
    """Anything, as plain JSON types.

    Not decoration: the Anthropic path echoes raw SDK content blocks back
    through `llm.assistant_turn()`, so a message list handed to `complete()` is
    not made of dicts. `default=str` is the same fallback app/events.py already
    uses to get graph output onto the wire.
    """
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return str(obj)


@contextmanager
def turn(*, session_id: str, question: str, connection_id: str) -> Iterator[Any]:
    """One turn: the root span every generation and tool call hangs off.

    The handle's `.trace_id` is minted here rather than read back out of the
    ambient context inside a node, so `turn.trace_id` can be written to the turn
    row without depending on OpenTelemetry context reaching LangGraph's tasks.

    `session_id` is the agent's session, LangGraph's `thread_id` and Langfuse's
    session all at once — they were always the same identifier, and a trace store
    that grouped them differently would be describing a different program.
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
        input=_serialisable({"question": question, "connection_id": connection_id}),
        # Which prose produced this turn. The one thing Langfuse's own prompt
        # management would have given for free, and the reason it is worth
        # having: a harvest that cannot tell a run under the seed from a run
        # under a candidate will train round two on round one's output. Read
        # after the `lf is None` return, so off still costs nothing.
        #
        # Eight hex characters per node, which is 124 for the whole dict —
        # under the 200 a metadata value is truncated at, so reading it back
        # needs no `expand_metadata`. A longer hash would silently lose its tail
        # and every fingerprint would start comparing equal.
        metadata=_serialisable({"prompts": prompts.fingerprint()}),
    ) as span:
        # The connection as a tag, not just metadata: "everything this warehouse
        # was ever asked" is the question you actually want to filter on, and it
        # is the same scoping rule the cache and the API routes already use.
        with propagate_attributes(
            session_id=str(session_id),
            trace_name="turn",
            tags=[f"connection:{connection_id}"],
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
            # A refusal and a 400 from a local server are both successful-looking
            # control flow elsewhere; on the trace they have to read as failures.
            gen.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
            raise


# --------------------------------------------------------------- the read half
#
# Everything above writes. This reads, and it lives here for the same reason the
# writes do: `import langfuse` appears in this module and nowhere else, so a
# second module wanting its own client is the failure the rule exists to
# prevent. It is one function returning plain dicts — if it grows past that,
# the signal is to split this file into write and read halves, not to relax the
# rule. tests/test_cli_isolation.py now enforces it.


def observations(
    *,
    name: str,
    kind: str = "GENERATION",
    since: Any = None,
    until: Any = None,
    page: int = 100,
) -> Iterator[dict[str, Any]]:
    """Everything recorded under one observation name, oldest page first.

    This is what makes an offline prompt corpus possible at all. Because
    `llm.complete()` is the only place a generation is opened, and it records
    `input={"system", "messages"}` under a `name=` that is the graph node,
    Langfuse already holds a per-node dataset of exact inputs and outputs. The
    architecture built the harvester by accident; this reads it back.

    Two shapes are used. `kind="GENERATION"` with a node name is the dataset.
    `name="turn", kind="SPAN"` is the *scope*: the turn span records
    `input={"question", "connection_id"}` and the prompt fingerprint in its
    metadata, which is the only way to say which warehouse a recorded call was
    about. It has to come from here rather than from a join to `turn.trace_id`,
    because `make reset` empties that table by design and Langfuse keeps the
    trace — and v4 removed the trace-list endpoint, so an observation cannot be
    filtered by the `connection:` tag either.

    Yields nothing at all when tracing is off, which is the common case.
    """
    lf = client()
    if lf is None:
        return

    cursor = None
    while True:
        # `fields` defaults to core,basic — input and output are *absent*
        # without `io`, and the call succeeds, so the symptom is a corpus of
        # empty prompts rather than an error. `parse_io_as_json` is deprecated
        # and returns 400 if set at all: input arrives as a raw string, which
        # is why `_as_json` below is required rather than defensive.
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
