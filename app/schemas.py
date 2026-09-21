"""What the API accepts and returns.

`store.CacheEntry` is not the wire format: `schema_fp` and the turn pointers are
the graph's bookkeeping, not part of a versioned contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.config import Config

Kind = Literal["schema_fact", "recipe"]


class AskBody(BaseModel):
    question: str = Field(..., min_length=1)
    # False asks the turn to ignore what the agent has learned and to keep
    # nothing it learns. What an optimisation run sends: it is measuring a
    # prompt, and a turn answering from memory measures the memory.
    memory: bool = True
    # A caller that supplies none gets a distinct thread rather than a shared
    # default. The value used comes back on the X-Session-Id response header.
    session_id: UUID = Field(default_factory=uuid4)


class FeedbackBody(BaseModel):
    """What a person thought of one answer.

    `extra="forbid"`, so a client sending `verdict` or `correct_sql` learns that
    here rather than by watching a corpus never fill up.
    """

    model_config = ConfigDict(extra="forbid")

    correct: bool
    # Free text, and only worth asking for when the answer was wrong. It is the
    # side information a later optimisation reads, so it is prose rather than an
    # enum of reasons somebody guessed in advance.
    comment: str | None = None


class FeedbackOut(BaseModel):
    """Where the verdict went. The trace is the record, not a row here."""

    trace_id: str
    name: str
    value: float


class CacheEntryOut(BaseModel):
    """One entry, as the model sees it plus what a human needs to judge it."""

    id: int
    kind: Kind
    name: str | None
    claim: str
    sql_fragment: str | None
    tables: list[str]
    origin: str
    pinned: bool
    tombstone: bool
    verified: bool
    hits: int
    # Computed per request, not stored: the entry was learned against a schema
    # that has since changed shape.
    stale: bool


class CacheSummary(BaseModel):
    total: int
    verified: int
    stale: int
    # Entries `load_cache` filtered out, so they aren't invisible.
    disabled: int


class CacheListOut(BaseModel):
    summary: CacheSummary
    entries: list[CacheEntryOut]


class ResetOut(BaseModel):
    """Rows removed, per table. Empty for a table the migrations haven't made."""

    wiped: dict[str, int]


class TargetTestOut(BaseModel):
    """What a probe of the target database found.

    `ok: false` is a successful diagnostic, not an error — the reason is in the
    body. `read_only` and `warnings` are advisory; the enforcement is
    `readonly_tier`. See app/dialects.py.
    """

    ok: bool
    driver: str | None = None
    readonly_tier: Literal["enforced", "partial"] | None = None
    latency_ms: int | None = None
    server_version: str | None = None
    username: str | None = None
    # Which schema the agent will explore. Reflected, not assumed.
    default_schema: str | None = None
    tables: int | None = None
    # None means "we could not find out", which on SQLite is the truth: there
    # are no credentials to judge.
    read_only: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    # Sanitised. Never echoes the DSN, which carries the password.
    error: str | None = None


class TurnOut(BaseModel):
    """One row of the demo chart."""

    id: int
    question: str
    explored: bool
    tool_calls: int
    cache_entries: int
    tokens_in: int
    tokens_out: int
    # in + out, precomputed: the chart's y value.
    tokens: int
    latency_ms: int | None
    sql: str | None
    answer: str | None
    created_at: datetime
    # None means the turn was taken with tracing off, which is the common case.
    trace_id: str | None = None
    # Dollars, where the backend itemised the charge. None means nobody said,
    # which is not the same as free.
    cost: float | None = None


class TurnListOut(BaseModel):
    turns: list[TurnOut]


class ConfigOut(BaseModel):
    """What the server is running, and which of it the local overlay decided.

    `config` is the validated merge, nulls included: an unset node effort is
    `null` on the wire so the shape never changes with the file. The CLI is
    what omits them.
    """

    overlay: str | None  # config.local.yaml's path, or null when there is none
    overridden: list[str]  # dotted keys the overlay set, sorted
    config: Config
    # Not part of the merge — it comes from the environment, not the file — but
    # it is the other half of "what is this process doing", and a client that
    # needs a trace to write to can ask before spending a turn finding out.
    tracing: bool
