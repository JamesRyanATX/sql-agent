"""The capability table says what each engine enforces. This checks it is true.

`app/dialects.py` is a set of claims — this dialect blocks DML, that one does
not block DDL, the third has no statement timeout at all. Claims drift. A
dialect that claims enforcement it does not have is a security hole; one that
claims none while quietly enforcing means the table is over-pessimistic and
users are warned for nothing. **Both directions fail here**, deliberately.

Two rules for this file, because they are what stop it quietly testing nothing:

1. Expectations come from the record, evidence comes from the database. Never
   `assert observed == observed`.
2. **Never `pytest.skip` keyed on the claim.** A skip conditioned on what the
   table says is how a wrong table starts agreeing with itself. `blocks_ddl` is
   asserted for MySQL as *False*, so a future MySQL that starts blocking DDL
   fails this file and prompts an update, rather than passing in silence.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app import db, dialects

WRITES = {
    # `TRUNCATE` is deliberately absent: SQLite has none (it optimises
    # `DELETE FROM t`), so including it would test the parser, not the guard.
    # It stays in the Postgres-only matrix in test_isolation.py, where it means
    # something.
    "insert": ("dml", "INSERT INTO ledger (ledger_id, sku_id, label) VALUES (99, 99, 'x')"),
    "update": ("dml", "UPDATE ledger SET label = 'x'"),
    "delete": ("dml", "DELETE FROM ledger"),
    "create": ("ddl", "CREATE TABLE cap_probe (x int)"),
    "drop": ("ddl", "DROP TABLE ledger_item"),
}


def test_every_supported_driver_has_a_capability_record():
    """A missing record must fail at registration, not on the first turn inside
    an SSE stream where it reads as the agent breaking."""
    from app import store

    assert {d.split("+")[0] for d in store.DRIVERS} == set(dialects.CAPABILITIES)
    with pytest.raises(KeyError, match="postgresql"):
        dialects.for_dialect("oracle")


async def test_the_claimed_read_only_state_is_actually_set(portable):
    """The direct successor to `SHOW default_transaction_read_only`, and what
    fails if the connect event were registered on the async engine instead of
    `engine.sync_engine` — an easy and completely silent miss, because binding
    it there raises nothing and simply never fires."""
    cap = dialects.for_dialect(portable.dialect)
    statement, expected = cap.probe
    async with db.target(portable.cid) as conn:
        got = (await conn.exec_driver_sql(statement)).scalar_one()
    assert str(got) == expected, (
        f"{portable.dialect} claims read-only sessions but reports {got!r} — "
        "either dialects.install did not run or the probe is wrong"
    )


@pytest.mark.parametrize("name", sorted(WRITES))
async def test_the_claimed_write_block_is_real(portable, name):
    """Fails in *both* directions, which is the whole point of the file."""
    kind, statement = WRITES[name]
    cap = dialects.for_dialect(portable.dialect)
    expected = cap.blocks_dml if kind == "dml" else cap.blocks_ddl

    async with db.target(portable.cid) as conn:
        try:
            await conn.exec_driver_sql(statement)
            refused = False
        except Exception:
            refused = True

    assert refused is expected, (
        f"{portable.dialect} claims blocks_{kind}={expected}, but {name} was "
        f"{'refused' if refused else 'ACCEPTED'} — the capability table is wrong"
    )


async def test_the_data_survives_whatever_the_tier(portable):
    """The one assertion identical on every dialect, and the one that matters.

    Where a tier is weaker the *disclosure* is the feature; where it is not, the
    row count is.
    """
    async with db.target(portable.cid) as conn:
        for _, statement in WRITES.values():
            try:
                await conn.exec_driver_sql(statement)
            except Exception:
                pass
    async with portable.engine.connect() as conn:
        n = (await conn.execute(text("SELECT count(*) FROM ledger"))).scalar_one()
    cap = dialects.for_dialect(portable.dialect)
    if cap.blocks_dml:
        assert n == 5, "DML was claimed blocked and the rows moved anyway"


async def test_generated_sql_runs_under_the_transaction_guard_too(portable):
    """`target_readonly` is where the model's SQL runs. On Postgres it re-applies
    both settings; on MySQL and SQLite the transaction list is empty and the
    session guard is doing the work — that emptiness is the model working, not
    a gap, and this is what says so."""
    cap = dialects.for_dialect(portable.dialect)
    async with db.target_readonly(portable.cid) as conn:
        assert (await conn.exec_driver_sql("SELECT count(*) FROM ledger")).scalar_one() == 5
        if cap.blocks_dml:
            with pytest.raises(Exception) as e:
                await conn.exec_driver_sql("DELETE FROM ledger")
            assert dialects.is_read_only_error(e.value), (
                f"refused, but not by the read-only guard: {e.value}"
            )


async def test_the_probe_reports_the_capability_truthfully(client, portable):
    body = (await client.post(f"/v1/connections/{portable.cid}/test")).json()
    cap = dialects.for_dialect(portable.dialect)
    assert body["ok"] is True, body
    assert body["driver"] == f"{portable.dialect}+{ {'postgresql':'psycopg','sqlite':'aiosqlite','mysql':'asyncmy'}[portable.dialect] }"
    assert body["readonly_tier"] == cap.tier
    assert body["tables"] == 3
    assert body["default_schema"]


async def test_a_dialect_that_cannot_enforce_says_so(client, portable):
    """If enforcement is best-effort, the warning *is* the feature — and an
    untested warning is a feature that gets deleted as noise six months from
    now. An undisclosed gap and an undisclosed hole are the same bug."""
    cap = dialects.for_dialect(portable.dialect)
    body = (await client.post(f"/v1/connections/{portable.cid}/test")).json()
    for gap in cap.gaps:
        assert any(gap in w for w in body["warnings"]), (
            f"{portable.dialect} declares the gap {gap!r} and never mentions it: "
            f"{body['warnings']}"
        )
    if cap.tier == "enforced":
        assert cap.gaps == (), "an enforced tier has nothing to warn about"


async def test_sqlite_reports_no_credentials_rather_than_read_only(client, portable):
    """`read_only: true` on SQLite would be a lie a user acts on — there are no
    credentials to judge, so `None` is the honest answer and it finally means
    something."""
    if portable.dialect != "sqlite":
        pytest.skip("about SQLite's absence of users, not about the others")
    body = (await client.post(f"/v1/connections/{portable.cid}/test")).json()
    assert body["username"] is None
    assert not dialects.for_dialect("sqlite").has_auth


def test_sqlite_does_not_claim_a_timeout_it_cannot_deliver():
    """Flagged in advance rather than discovered: do NOT fake this with
    `asyncio.wait_for`. Cancelling a coroutine does not stop a query already
    running on aiosqlite's thread, and claiming a timeout on that basis is
    precisely the drift this file exists to prevent."""
    cap = dialects.for_dialect("sqlite")
    assert cap.has_timeout is False
    assert cap.tier == "partial"
    assert any("no statement timeout" in g for g in cap.gaps)
