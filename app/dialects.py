"""How far read-only can actually be pushed, per dialect.

This used to be one line in a pool's `configure=` callback and one sentence in
CLAUDE.md promising a structural guarantee. Both were true while every target
was Postgres. They are not true of SQLite, which has a per-connection pragma and
no statement timeout at all, and they are only half true of MySQL, whose
transaction read-only mode does not touch DDL and whose timeout bounds SELECTs
only.

So the guarantee became a **capability**: enforce what the dialect allows, name
the tier on the wire, warn where it is weaker than Postgres's, and never refuse a
registration over it. A registration refused because a dialect is honest about
its limits is a feature that does not work; one accepted while claiming a
guarantee it does not have is worse than either.

**`read_only` is deliberately not a boolean.** MySQL blocks DML and not DDL,
SQLite blocks both, Postgres blocks both and adds a timeout. A single flag would
have to round MySQL up — a lie — or down, a false alarm on every registration.
Three engines, three genuinely different tiers.

`app/db.py` and `app/api.py` both read this module. **Nothing outside it branches
on a dialect name**; if you find yourself writing `if dialect == "mysql"`
elsewhere, the fact belongs here.
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

    #: Statements run on every new connection, before any query. This is the
    #: whole write-protection story for `db.target()`, which has no transaction
    #: of its own — it is what `explore`, `infer_tables` and `extract` use.
    session: Callable[[Any, int], tuple[str, ...]]
    #: Statements run inside `db.target_readonly()`'s explicit transaction.
    #: Empty where the dialect has no transaction-scoped equivalent; that
    #: emptiness is the model working, not a gap.
    transaction: Callable[[Any, int], tuple[str, ...]] = lambda d, ms: ()

    #: (sql, expected) proving the session statements actually took. The direct
    #: successor to `SHOW default_transaction_read_only`, and what fails if the
    #: connect event were registered on the wrong engine — an easy and
    #: completely silent miss, since `event.listens_for` must target
    #: `engine.sync_engine` rather than the async engine.
    probe: tuple[str, str] | None = None

    blocks_dml: bool = True
    blocks_ddl: bool = True
    has_timeout: bool = True
    #: How the engine treats unquoted identifiers. `fold` lower-cases them
    #: (Postgres), `preserve` keeps them (MySQL on a case-sensitive filesystem,
    #: SQLite), `insensitive` matches without regard to case.
    identifier_case: Literal["fold", "preserve", "insensitive"] = "fold"
    has_auth: bool = True
    #: What `_probe` must warn about. An untested warning is a feature that gets
    #: deleted as noise six months from now, so tests/test_capabilities.py
    #: asserts each of these reaches ConnectionTestOut.warnings.
    gaps: tuple[str, ...] = ()

    @property
    def tier(self) -> Tier:
        """`enforced` only when writes are blocked *and* a runaway query dies."""
        return (
            "enforced"
            if self.blocks_dml and self.blocks_ddl and self.has_timeout
            else "partial"
        )


# Interpolating the timeout into the statement is deliberate and is the same
# reason `db.target_readonly` reaches for `set_config`: `SET` takes no bind
# parameters on any of the three dialects. `int()` is the structural guarantee,
# and the value comes from `settings()`, never from a request.


def _postgres_session(dialect: Any, timeout_ms: int) -> tuple[str, ...]:
    return (
        # SET SESSION CHARACTERISTICS rather than `options=-c ...` in the URL,
        # because pgbouncer in transaction mode rejects the latter. It binds any
        # role including a superuser: transaction_read_only is a property of the
        # transaction, not a privilege.
        "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
        # New with the port, and worth having: the timeout is session-level now,
        # so it also covers `db.target()`, which never had one.
        # `describe_table`'s SELECT count(*) on somebody's billion-row fact
        # table is the case that made that obvious.
        f"SET SESSION statement_timeout = {int(timeout_ms)}",
    )


def _postgres_transaction(dialect: Any, timeout_ms: int) -> tuple[str, ...]:
    # Redundant with the session statements above, and kept: defence that exists
    # in one place is one edit from not existing.
    return (
        "SELECT set_config('transaction_read_only', 'on', true)",
        f"SELECT set_config('statement_timeout', '{int(timeout_ms)}', true)",
    )


def _mysql_session(dialect: Any, timeout_ms: int) -> tuple[str, ...]:
    read_only = "SET SESSION TRANSACTION READ ONLY"
    if getattr(dialect, "_is_mariadb", False):
        # MariaDB has no max_execution_time. Its knob is max_statement_time and
        # it is **seconds as a float**, not milliseconds — the two spellings
        # differ by 1000x, and crossing them fails silently in the direction of
        # "no timeout at all".
        return (read_only, f"SET SESSION max_statement_time = {timeout_ms / 1000:.3f}")
    return (read_only, f"SET SESSION max_execution_time = {int(timeout_ms)}")


def _sqlite_session(dialect: Any, timeout_ms: int) -> tuple[str, ...]:
    # Per connection, and it covers attached databases.
    #
    # `file:...?mode=ro` at the OS level would be stronger and is deliberately
    # not used: a database in WAL mode needs write access to its -shm file, so
    # mode=ro turns a working warehouse into "attempt to write a readonly
    # database" at connect time. Enforcement that breaks correct configurations
    # is not stronger enforcement.
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
        # DDL *is* blocked, and this was written the other way round first.
        # The received wisdom is that DDL autocommits and so escapes a read-only
        # transaction; MySQL 8.4 disagrees — CREATE and DROP both come back
        # ERROR 1792 (25006) "Cannot execute statement in a READ ONLY
        # transaction". tests/test_capabilities.py asserts the claim in *both*
        # directions, which is what caught the over-pessimistic version: a
        # dialect warned about a hole it does not have is a dialect nobody
        # trusts the warnings of.
        blocks_ddl=True,
        # And that is also why there is no timeout gap. max_execution_time
        # bounds SELECTs only, which would matter if anything else could run —
        # nothing else can.
        identifier_case="preserve",
    ),
    "sqlite": Capability(
        name="sqlite",
        session=_sqlite_session,
        probe=("PRAGMA query_only", "1"),
        has_timeout=False,
        identifier_case="preserve",
        # A file has no users, so there is nothing to authenticate and nothing
        # to hold SELECT-and-nothing-else. `query_only` is the whole guarantee.
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

    The `configure=` callback of the psycopg pool this replaced, in SQLAlchemy's
    spelling — and the reason is unchanged: the agent reaches a registered
    warehouse with credentials it did not choose, and `db.target()` has no
    transaction of its own to guard.

    **`engine.sync_engine`, not `engine`.** Registering the event on an
    `AsyncEngine` binds nothing and raises nothing; the callback simply never
    fires and every target connection is quietly writable.
    """
    from sqlalchemy import event

    cap = for_dialect(dialect_name)

    @event.listens_for(engine.sync_engine, "connect")
    def _configure(dbapi_connection: Any, _record: Any) -> None:
        # Sync, and executing SQL from inside an async engine's event — which
        # works because the connect event runs inside the greenlet SQLAlchemy's
        # asyncio layer drives the driver from. This is the documented pattern
        # for aiosqlite pragmas, not a shortcut.
        cursor = dbapi_connection.cursor()
        try:
            for statement in cap.session(engine.sync_engine.dialect, timeout_ms):
                cursor.execute(statement)
        finally:
            cursor.close()


def is_read_only_error(exc: BaseException) -> bool:
    """Did this failure come from the read-only guard rather than the SQL?

    Every dialect words it differently and none of them share an exception type
    through SQLAlchemy's wrapper, so this matches on the message — which is
    exactly the kind of thing that rots. It is used only by tests, where a false
    negative is a visible failure rather than a silent one.
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
