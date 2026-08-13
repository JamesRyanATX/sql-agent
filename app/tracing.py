"""The one module that knows Langfuse exists.

Same rule [app/dialects.py](app/dialects.py) has for dialect names and
[app/secrets.py](app/secrets.py) has for encryption: `import langfuse` appears
here and nowhere else, so the rest of `app/` sees four context managers and a
boolean. That is what makes "tracing off" cost nothing to prove — there is one
`enabled()` to read rather than a scattering of `if` at every call site.

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
