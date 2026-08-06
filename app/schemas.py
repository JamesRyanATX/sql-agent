"""What the API accepts and returns.

`store.CacheEntry` is not the wire format. It carries fields that exist for the
graph's benefit — `schema_fp`, `created_turn`, `last_used_turn` — and putting
them on the wire would make internal bookkeeping part of a versioned contract.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

Kind = Literal["schema_fact", "recipe"]


class AskBody(BaseModel):
    question: str = Field(..., min_length=1)
    # Optional: the CLI has no conversation to name, and a caller that doesn't
    # supply one still gets a distinct thread rather than sharing a default.
    # The value used comes back on the X-Session-Id response header.
    session_id: UUID = Field(default_factory=uuid4)


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
