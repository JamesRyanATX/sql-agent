"""How far read-only can actually be pushed, per dialect.

Enforce what the dialect allows, name the tier on the wire, warn where it is
weaker than Postgres's — and never refuse a registration over it. Read-only is a
capability rather than a boolean because the three engines genuinely differ:
SQLite has no statement timeout, and MySQL's bounds SELECTs only.

**Nothing outside this module branches on a dialect name.** If you find yourself
writing `if dialect == "mysql"` elsewhere, the fact belongs here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal["enforced", "partial"]


@dataclass(frozen=True, slots=True)
class Capability:
    """What one dialect can be made to promise."""

    name: str  # the SQLAlchemy dialect name

    #: Statements run on every new connection. The whole write-protection story
    #: for `db.target()`, which has no transaction of its own.
    session: Callable[[Any, int], tuple[str, ...]]
    #: Statements run inside `db.target_readonly()`'s transaction. Empty where
    #: the dialect has no transaction-scoped equivalent.
    transaction: Callable[[Any, int], tuple[str, ...]] = lambda d, ms: ()

    #: (sql, expected) proving the session statements took. What fails if the
    #: connect event is registered on the async engine instead of `sync_engine`,
    #: which otherwise misses silently.
    probe: tuple[str, str] | None = None

    blocks_dml: bool = True
    blocks_ddl: bool = True
    has_timeout: bool = True
    #: How the engine treats unquoted identifiers. `fold` lower-cases them
    #: (Postgres), `preserve` keeps them (MySQL on a case-sensitive filesystem,
    #: SQLite), `insensitive` matches without regard to case.
    identifier_case: Literal["fold", "preserve", "insensitive"] = "fold"
    has_auth: bool = True
    #: What `_probe` must warn about. tests/test_capabilities.py asserts each of
    #: these reaches ConnectionTestOut.warnings.
    gaps: tuple[str, ...] = ()

    @property
    def tier(self) -> Tier:
        """`enforced` only when writes are blocked *and* a runaway query dies."""
        return (
            "enforced"
            if self.blocks_dml and self.blocks_ddl and self.has_timeout
            else "partial"
        )


# The timeout is interpolated because `SET` takes no bind parameters on any of
# the three dialects. `int()` is the guarantee, and the value comes from
# `settings()`, never from a request.


def _postgres_session(dialect: Any, timeout_ms: int) -> tuple[str, ...]:
    return (
        # Not `options=-c ...` in the URL, which pgbouncer rejects in transaction
        # mode. This binds any role including a superuser, because
        # transaction_read_only is a property of the transaction, not a privilege.
        "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
        # Session-level, so it also covers `db.target()` — `describe_table`'s
        # count(*) on a billion-row fact table is the case that needs it.
        f"SET SESSION statement_timeout = {int(timeout_ms)}",
    )


def _postgres_transaction(dialect: Any, timeout_ms: int) -> tuple[str, ...]:
    # Redundant with the session statements: defence that exists in one place is
    # one edit from not existing.
    return (
        "SELECT set_config('transaction_read_only', 'on', true)",
        f"SELECT set_config('statement_timeout', '{int(timeout_ms)}', true)",
    )


def _mysql_session(dialect: Any, timeout_ms: int) -> tuple[str, ...]:
    read_only = "SET SESSION TRANSACTION READ ONLY"
    if getattr(dialect, "_is_mariadb", False):
        # MariaDB's knob is max_statement_time, in **seconds as a float**. The
        # two spellings differ by 1000x and crossing them means no timeout.
        return (read_only, f"SET SESSION max_statement_time = {timeout_ms / 1000:.3f}")
    return (read_only, f"SET SESSION max_execution_time = {int(timeout_ms)}")


def _sqlite_session(dialect: Any, timeout_ms: int) -> tuple[str, ...]:
    # Per connection, and it covers attached databases. Not `file:...?mode=ro`,
    # which is stronger but breaks WAL mode — that needs write access to -shm.
    return ("PRAGMA query_only = 1",)


CAPABILITIES: dict[str, Capability] = {
    "postgresql": Capability(
        name="postgresql",
        session=_postgres_session,
        transaction=_postgres_transaction,
        probe=("SHOW default_transaction_read_only", "on"),
        identifier_case="fold",
    ),
    "mysql": Capability(
        name="mysql",
        session=_mysql_session,
        probe=("SELECT @@SESSION.transaction_read_only", "1"),
        # DDL is blocked despite the received wisdom that it autocommits past a
        # read-only transaction: MySQL 8.4 returns ERROR 1792 for CREATE and
        # DROP. tests/test_capabilities.py asserts this in both directions.
        blocks_ddl=True,
        # Hence no timeout gap either: max_execution_time bounds SELECTs only,
        # and nothing else can run.
        identifier_case="preserve",
    ),
    "sqlite": Capability(
        name="sqlite",
        session=_sqlite_session,
        probe=("PRAGMA query_only", "1"),
        has_timeout=False,
        identifier_case="preserve",
        # A file has no users, so `query_only` is the whole guarantee.
        has_auth=False,
        gaps=(
            "SQLite has no statement timeout: a runaway query runs to "
            "completion and the turn waits for it. Point this at a database "
            "you can afford to have scanned",
            "SQLite has no users, so there are no credentials to restrict — "
            "read-only rests entirely on PRAGMA query_only",
        ),
    ),
}


def for_dialect(name: str) -> Capability:
    cap = CAPABILITIES.get(name)
    if cap is None:
        raise KeyError(
            f"no capability record for dialect {name!r} — "
            f"this agent speaks {', '.join(sorted(CAPABILITIES))}"
        )
    return cap


def install(engine: Any, dialect_name: str, timeout_ms: int) -> None:
    """Run the read-only session statements on every connection of `engine`.

    The agent reaches a registered warehouse with credentials it did not choose,
    and `db.target()` has no transaction of its own to guard.

    **`engine.sync_engine`, not `engine`.** Registering the event on an
    `AsyncEngine` binds nothing and raises nothing; the callback never fires and
    every target connection is quietly writable.
    """
    from sqlalchemy import event

    cap = for_dialect(dialect_name)

    @event.listens_for(engine.sync_engine, "connect")
    def _configure(dbapi_connection: Any, _record: Any) -> None:
        # Sync SQL inside an async engine's event works because the connect
        # event runs in the greenlet SQLAlchemy drives the driver from. This is
        # the documented pattern for aiosqlite pragmas.
        cursor = dbapi_connection.cursor()
        try:
            for statement in cap.session(engine.sync_engine.dialect, timeout_ms):
                cursor.execute(statement)
        finally:
            cursor.close()


def is_read_only_error(exc: BaseException) -> bool:
    """Did this failure come from the read-only guard rather than the SQL?

    Matches on the message, because the dialects share no exception type through
    SQLAlchemy's wrapper. Used only by tests, where a false negative is visible.
    """
    text = str(getattr(exc, "orig", None) or exc).lower()
    return any(
        phrase in text
        for phrase in (
            "read-only transaction",  # postgres
            "read only transaction",  # mysql
            "readonly database",  # sqlite
            "attempt to write",  # sqlite, older wording
        )
    )
