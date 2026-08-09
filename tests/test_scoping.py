"""What the agent learns about one database stays there.

The cache used to be one global table, and `cache_entry.name` was globally
unique — so learning `revenue` against a second warehouse would not merely leak,
it would *overwrite* the first one's recipe through the upsert. These tests are
the reason the unique index moved to `(connection_id, name)`.

The failure this guards against is quiet. A crossed entry produces an answer
that looks right, SQL that looks right, and numbers from the wrong database.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from psycopg import AsyncConnection

from app import store
from tests.conftest import DEFAULT_CONNECTION as A
from tests.conftest import OTHER_CONNECTION as B

PROBE = "fp_probe"


def entry(name: str, claim: str = "a claim", **kw) -> store.CacheEntry:
    return store.CacheEntry(kind="recipe", name=name, claim=claim, **kw)


@pytest.fixture(autouse=True)
async def clean(agent_conn: AsyncConnection):
    """Both connections' entries, before and after — the assertions here count."""
    for cid in (A, B):
        await agent_conn.execute("DELETE FROM cache_entry WHERE connection_id = %s", (cid,))
    yield
    for cid in (A, B):
        await agent_conn.execute("DELETE FROM cache_entry WHERE connection_id = %s", (cid,))


async def test_two_connections_do_not_share_a_cache(agent_conn):
    await store.write_entries(agent_conn, [entry("spec:only-a")], connection_id=A)
    await store.write_entries(agent_conn, [entry("spec:only-b")], connection_id=B)

    assert [e.name for e in await store.load_cache(agent_conn, connection_id=A)] == ["spec:only-a"]
    assert [e.name for e in await store.load_cache(agent_conn, connection_id=B)] == ["spec:only-b"]


async def test_the_same_entry_name_can_exist_once_per_connection(agent_conn):
    """The one that would have failed before the index moved.

    `revenue` on two warehouses is two facts. With the old global unique index
    the second write went down `DO UPDATE` and destroyed the first — silently,
    because an upsert reports success.
    """
    await store.write_entries(
        agent_conn, [entry("spec:revenue", "revenue excludes cancelled")], connection_id=A
    )
    await store.write_entries(
        agent_conn, [entry("spec:revenue", "revenue is gross, cancellations included")],
        connection_id=B,
    )

    a = await store.load_cache(agent_conn, connection_id=A)
    b = await store.load_cache(agent_conn, connection_id=B)
    assert [e.claim for e in a] == ["revenue excludes cancelled"]
    assert [e.claim for e in b] == ["revenue is gross, cancellations included"]


async def test_re_learning_a_name_still_upserts_within_one_connection(agent_conn):
    """Scoping the index must not have turned the upsert into an append."""
    await store.write_entries(agent_conn, [entry("spec:revenue", "first")], connection_id=A)
    await store.write_entries(agent_conn, [entry("spec:revenue", "refined")], connection_id=A)

    loaded = await store.load_cache(agent_conn, connection_id=A)
    assert [e.claim for e in loaded] == ["refined"]


async def test_count_disabled_is_scoped_too(agent_conn):
    await store.write_entries(
        agent_conn, [entry("spec:off", disabled=True)], connection_id=B
    )
    assert await store.count_disabled(agent_conn, connection_id=A) == 0
    assert await store.count_disabled(agent_conn, connection_id=B) == 1


async def test_bump_hits_will_not_credit_another_connections_entry(agent_conn):
    """A resumed checkpoint carries entry ids, and a thread replayed against
    another connection would otherwise credit that connection's entries here."""
    (written,) = await store.write_entries(agent_conn, [entry("spec:b")], connection_id=B)

    await store.bump_hits(agent_conn, [written], connection_id=A, turn_id=1)

    (survivor,) = await store.load_cache(agent_conn, connection_id=B)
    assert survivor.hits == 0

    await store.bump_hits(agent_conn, [written], connection_id=B, turn_id=1)
    (survivor,) = await store.load_cache(agent_conn, connection_id=B)
    assert survivor.hits == 1


async def test_stale_is_computed_against_the_entrys_own_target(
    agent_conn, target_conn, target_conn_b
):
    """The quiet one.

    `fp_probe` exists on both business databases with the same name and a
    different shape. An entry learned about A's must read fresh against A and
    stale against B — otherwise resolving the target by anything other than the
    entry's own connection_id is undetectable.
    """
    await target_conn.execute(f"DROP TABLE IF EXISTS {PROBE}")
    await target_conn.execute(f"CREATE TABLE {PROBE} (id bigint, created timestamptz)")
    await target_conn.execute(f"GRANT SELECT ON {PROBE} TO reader")

    learned = entry("spec:probe shape", "the probe table", tables=[PROBE])
    await store.fingerprint_entries(target_conn, [learned])
    await store.write_entries(agent_conn, [learned], connection_id=A)
    assert learned.schema_fp  # it named a table, so it got stamped

    entries = await store.load_cache(agent_conn, connection_id=A)
    assert await store.stale_ids(target_conn, entries) == set()
    # B's fp_probe is (id, created_at) — same name, different columns.
    assert await store.stale_ids(target_conn_b, entries) == {learned.id}

    await target_conn.execute(f"DROP TABLE IF EXISTS {PROBE}")


async def test_the_cache_endpoint_asks_the_right_target(
    client: AsyncClient, agent_conn, target_conn
):
    """The wiring half of the test above: /connections/{id}/cache must open
    *that* connection's target, not whichever one is handy."""
    await target_conn.execute(f"DROP TABLE IF EXISTS {PROBE}")
    await target_conn.execute(f"CREATE TABLE {PROBE} (id bigint, created timestamptz)")
    await target_conn.execute(f"GRANT SELECT ON {PROBE} TO reader")

    learned = entry("spec:probe shape", "the probe table", tables=[PROBE])
    await store.fingerprint_entries(target_conn, [learned])
    await store.write_entries(agent_conn, [learned], connection_id=A)

    body = (await client.get(f"/v1/connections/{A}/cache")).json()
    assert body["summary"]["stale"] == 0
    assert [e["name"] for e in body["entries"]] == ["spec:probe shape"]

    # B never learned it, so it is not merely fresh — it is absent.
    other = (await client.get(f"/v1/connections/{B}/cache")).json()
    assert other["entries"] == []

    await target_conn.execute(f"DROP TABLE IF EXISTS {PROBE}")


async def test_a_turn_records_the_connection_it_ran_against(agent_conn):
    turn_id = await store.start_turn(
        agent_conn, connection_id=B, session_id="44444444-4444-4444-4444-444444444444",
        question="how many widgets?",
    )
    cur = await agent_conn.execute(
        "SELECT connection_id FROM turn WHERE id = %s", (turn_id,)
    )
    assert (await cur.fetchone())["connection_id"] == B

    assert await store.session_connection(
        agent_conn, "44444444-4444-4444-4444-444444444444"
    ) == B
    await agent_conn.execute("DELETE FROM turn WHERE id = %s", (turn_id,))


async def test_a_session_cannot_change_connection_mid_thread(client: AsyncClient, agent_conn):
    """A thread's history is checkpointed. Reusing it against a second warehouse
    would hand the model the first one's conversation while it answers about the
    second — so the API refuses rather than producing a plausible answer."""
    session = "55555555-5555-5555-5555-555555555555"
    turn_id = await store.start_turn(
        agent_conn, connection_id=A, session_id=session, question="how many customers?"
    )

    resp = await client.post(
        f"/v1/connections/{B}/ask", json={"question": "how many widgets?", "session_id": session}
    )
    assert resp.status_code == 409
    assert A in resp.json()["detail"]

    await agent_conn.execute("DELETE FROM turn WHERE id = %s", (turn_id,))
