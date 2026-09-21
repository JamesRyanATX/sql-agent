"""Different text for one turn, put back afterwards.

An optimisation run has to ask "what would this turn have cost under a different
prompt?" — and then ask it again, a hundred times, in one process. Three things
it needs to vary are fixed for the life of that process: prompts are read from
`config/prompts/*.md` once and memoised, tool descriptions are a list literal in
`app/tools.py`, and per-node effort comes from a memoised `config()`. Clearing
those caches would change the text for every turn in the process, including any
the API is serving.

So the values live in a `ContextVar` instead, empty by default, and `prompts`,
`config` and `tools` look here first. Nothing is written to disk, and a turn
running beside this one is unaffected.

**Set it inside the coroutine that drives the turn, never from the caller.**
`asyncio` copies the current context when it creates a task, so a value set
before `astream` reaches every node. It does *not* cross a thread boundary:
`tools/gepa/adapter.py` drives everything through one event loop in a daemon
thread via `asyncio.run_coroutine_threadsafe`, which starts from an empty
context. Setting it inside is also what makes parallel trials safe, because
each gathered coroutine gets its own copy to modify.

This module imports nothing from the rest of `app`, so the three readers can use
it without an import cycle.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Overrides:
    """What to use instead, for one turn. Empty means "whatever is on disk"."""

    # node name -> the whole instruction block, as `config/prompts/<node>.md`
    # would have held it.
    prompts: dict[str, str] = field(default_factory=dict)
    # tool name -> its description. Only the description: the name and the input
    # schema are what `tools.run_tool` dispatches on, so a candidate that renamed
    # a tool would break the call rather than score badly for it.
    tools: dict[str, str] = field(default_factory=dict)
    # node name -> effort. Same labels `Config.effort_for` takes, so
    # `explore.summary` still falls back to `explore`.
    efforts: dict[str, str] = field(default_factory=dict)


NONE = Overrides()

_current: contextvars.ContextVar[Overrides] = contextvars.ContextVar(
    "sql_agent_overrides", default=NONE
)


def current() -> Overrides:
    """What is in force right now. `NONE` everywhere except inside `using`."""
    return _current.get()


@contextmanager
def using(overrides: Overrides) -> Iterator[Overrides]:
    """Run a block with this text in force, and put the old value back after.

    The token, rather than setting it back to `NONE`: nested use is not expected
    but silently flattening it would be a bug nobody could see from a trace.
    """
    token = _current.set(overrides)
    try:
        yield overrides
    finally:
        _current.reset(token)
