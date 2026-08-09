"""What the API accepts and returns.

`store.CacheEntry` is not the wire format. It carries fields that exist for the
graph's benefit — `schema_fp`, `created_turn`, `last_used_turn` — and putting
them on the wire would make internal bookkeeping part of a versioned contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Kind = Literal["schema_fact", "recipe"]

# Matches the CHECK in migrations/002_connections.sql. The id is a path segment
# and the name a user types on every command, so it is a slug, not free text.
ConnectionId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$")]
SslMode = Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]


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


# ------------------------------------------------------------------ connections


class ConnectionCreate(BaseModel):
    """Register a database the agent can be pointed at.

    `extra="forbid"` so a misspelled field is a 422 rather than a silent no-op
    that looks like it worked.
    """

    model_config = ConfigDict(extra="forbid")

    id: ConnectionId
    label: str | None = None
    host: str = Field(..., min_length=1)
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(..., min_length=1)
    username: str = Field(..., min_length=1)
    password: str | None = None
    sslmode: SslMode = "prefer"


class ConnectionPatch(BaseModel):
    """Change some fields. An absent field is left alone."""

    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    host: str | None = Field(default=None, min_length=1)
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = Field(default=None, min_length=1)
    username: str | None = Field(default=None, min_length=1)
    password: str | None = None
    sslmode: SslMode | None = None


class ConnectionOut(BaseModel):
    """A registered connection, as anyone outside this process may see it.

    **There is no `password` field** — not `None`, not `"***"`. A field that does
    not exist on this model cannot be leaked by a future `**row` splat or by
    dumping the wrong object, which is a stronger guarantee than remembering to
    blank one out. `dsn` is the address without it.
    """

    id: str
    label: str | None
    # 'env' is the built-in connection, whose address is TARGET_DATABASE_URL.
    # It cannot be changed or deleted over HTTP — the environment owns it.
    origin: Literal["api", "env"]
    host: str | None
    port: int | None
    database: str | None
    username: str | None
    sslmode: str
    has_password: bool
    dsn: str
    cache_entries: int
    turns: int
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ConnectionListOut(BaseModel):
    connections: list[ConnectionOut]


class ConnectionTestOut(BaseModel):
    """What a probe found. `ok: false` is a successful diagnostic, not an error.

    `read_only` and `warnings` are advisory and never gate: a legitimate
    warehouse role holding INSERT on one unrelated staging table is common, and
    refusing it would mean the feature does not work. The guarantee is the
    read-only session every target pool opens with (app/db.py), not this.
    """

    ok: bool
    latency_ms: int | None = None
    server_version: str | None = None
    username: str | None = None
    tables: int | None = None
    read_only: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    # Sanitised. Never echoes the DSN, which carries the password.
    error: str | None = None


class ConnectionCreatedOut(BaseModel):
    connection: ConnectionOut
    # None when ?probe=false. A failed probe does not block the create — a typo
    # should be visible now, but refusing to save because the warehouse is down
    # for maintenance is worse than saving it with a warning.
    test: ConnectionTestOut | None = None


class TurnOut(BaseModel):
    """One row of the demo chart."""

    id: int
    question: str
    explored: bool
    tool_calls: int
    cache_entries: int
    tokens_in: int
    tokens_out: int
    # in + out, precomputed: it is the chart's y value, and every client would
    # otherwise add the two together itself.
    tokens: int
    latency_ms: int | None
    sql: str | None
    answer: str | None
    created_at: datetime


class TurnListOut(BaseModel):
    turns: list[TurnOut]
