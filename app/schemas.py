"""What the API accepts and returns.

`store.CacheEntry` is not the wire format. It carries fields that exist for the
graph's benefit — `schema_fp`, `created_turn`, `last_used_turn` — and putting
them on the wire would make internal bookkeeping part of a versioned contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Kind = Literal["schema_fact", "recipe"]

# Matches the CHECK in migrations/002_connections.sql. The id is a path segment
# and the name a user types on every command, so it is a slug, not free text.
ConnectionId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$")]
SslMode = Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]

# Async-capable drivers only — see app/store.py's DRIVERS, and the CHECK in
# migrations/003 that says the same thing where a hand-edited row can hear it.
Driver = Literal["postgresql+psycopg", "mysql+asyncmy", "sqlite+aiosqlite"]


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
    driver: Driver = "postgresql+psycopg"
    label: str | None = None
    # host, port and username lost their `min_length` and their defaults
    # because SQLite has none of them — `database` is the whole address there.
    # The requirement did not go away; it moved into the validator below, which
    # is where a rule that depends on another field has to live.
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str = Field(..., min_length=1)  # the file path, for sqlite
    username: str | None = None
    password: str | None = None
    sslmode: SslMode = "prefer"
    # Driver kwargs the address columns cannot express. Filtered against a
    # per-driver allowlist in app/db.py before anything is dialled.
    options: dict[str, str | int | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _address_is_complete(self) -> ConnectionCreate:
        if self.driver.startswith("sqlite"):
            for f in ("host", "port", "username", "password"):
                if getattr(self, f) is not None:
                    raise ValueError(
                        f"sqlite has no {f} — `database` is the file path"
                    )
        else:
            for f in ("host", "username"):
                if not (getattr(self, f) or "").strip():
                    raise ValueError(f"{f} is required for {self.driver}")
            if self.port is None:
                self.port = 5432 if self.driver.startswith("postgresql") else 3306
        return self


class ConnectionPatch(BaseModel):
    """Change some fields. An absent field is left alone.

    **No `driver`.** A cached recipe is SQL in a dialect, so repointing a
    connection at another engine silently invalidates everything learned about
    it — and `schema_fp` would not catch it, because it hashes types and
    nullability and both survive the move. `PATCH` refuses one with a 409, the
    same way it refuses editing the env-owned row: the wrong state is made
    unrepresentable rather than detected.
    """

    model_config = ConfigDict(extra="forbid")

    # Accepted only so the refusal can explain itself. `extra="forbid"` would
    # reject it as an unknown field with a 422 that says nothing about why, and
    # "why" is the whole content of this rule.
    driver: Driver | None = None
    label: str | None = None
    host: str | None = Field(default=None, min_length=1)
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = Field(default=None, min_length=1)
    username: str | None = Field(default=None, min_length=1)
    password: str | None = None
    sslmode: SslMode | None = None
    options: dict[str, str | int | bool] | None = None


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
    driver: str
    host: str | None
    port: int | None
    database: str | None
    username: str | None
    sslmode: str
    has_password: bool
    dsn: str
    # How far read-only can be pushed on this engine. Derived from `driver`,
    # so it costs no I/O. See app/dialects.py.
    readonly_tier: Literal["enforced", "partial"]
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
    refusing it would mean the feature does not work. The enforcement is
    `readonly_tier`, and how strong that is depends on the dialect — see
    app/dialects.py.
    """

    ok: bool
    driver: str | None = None
    readonly_tier: Literal["enforced", "partial"] | None = None
    latency_ms: int | None = None
    server_version: str | None = None
    username: str | None = None
    # Which schema the agent will explore. Reflected, not assumed.
    # `schema` shadows a BaseModel attribute, hence the alias.
    default_schema: str | None = None
    tables: int | None = None
    # None means "we could not find out", which on SQLite is the truth: there
    # are no credentials to judge.
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
    # The Langfuse trace, when the turn was taken with tracing on. None is the
    # common case and means "not recorded", not "lost".
    trace_id: str | None = None


class TurnListOut(BaseModel):
    turns: list[TurnOut]
