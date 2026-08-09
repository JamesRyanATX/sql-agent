"""The /v1 surface: auth, what the cache listing says, and what DELETE takes.

These run in-process over ASGI — no server, no container. The point is the
contract, since `scripts/*` are now clients of it and nothing else checks the
shape they parse.

Requires Postgres up (`make up`).
"""

from collections.abc import AsyncIterator

import pytest
from httpx import AsyncClient
from psycopg import AsyncConnection

from app import store
from app.settings import settings
from tests.conftest import DEFAULT_CONNECTION as CID

# Every route about learned state hangs off the connection it is about, so the
# unscoped path does not exist to be reached by accident.
CACHE = f"/v1/connections/{CID}/cache"


PROBE = "api_probe"


@pytest.fixture
def conn(agent_conn):
    """The cache lives on the agent's server."""
    return agent_conn


@pytest.fixture(autouse=True)
async def clean(
    agent_conn: AsyncConnection, target_conn: AsyncConnection
) -> AsyncIterator[None]:
    """An empty cache each side. The graph reads the cache in full, so a leftover
    entry from one test is an input to the next."""
    await agent_conn.execute("TRUNCATE cache_entry RESTART IDENTITY CASCADE")
    yield
    await agent_conn.execute("TRUNCATE cache_entry RESTART IDENTITY CASCADE")
    # The probe table is on the demo server — that is where the schemas cache
    # entries describe actually live.
    await target_conn.execute(f"DROP TABLE IF EXISTS {PROBE}")


async def seed_entries(conn: AsyncConnection) -> None:
    await store.write_entries(
        conn,
        [
            store.CacheEntry(
                kind="recipe",
                name="active customer",
                claim="an active customer is one whose deleted_at is null",
                sql_fragment="customer WHERE deleted_at IS NULL",
                tables=["customer"],
                verified=True,
            ),
            store.CacheEntry(
                kind="schema_fact",
                name="orders.created",
                claim="the timestamp column on orders is `created`, not `created_at`",
                tables=["orders"],
            ),
        ],
        connection_id=CID,
    )


# ----------------------------------------------------------------------- auth


async def test_health_needs_no_token(client: AsyncClient, monkeypatch):
    """A load balancer has no credentials, and shouldn't need any."""
    monkeypatch.setenv("API_TOKEN", "s3cret")
    settings.cache_clear()
    try:
        assert (await client.get("/health")).status_code == 200
    finally:
        settings.cache_clear()


@pytest.mark.parametrize(
    "header",
    [
        pytest.param({}, id="missing"),
        pytest.param({"authorization": "Bearer wrong"}, id="wrong token"),
        pytest.param({"authorization": "s3cret"}, id="no bearer scheme"),
        pytest.param({"authorization": "Basic s3cret"}, id="wrong scheme"),
    ],
)
async def test_v1_rejects_anything_but_the_token(client: AsyncClient, monkeypatch, header):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    settings.cache_clear()
    try:
        resp = await client.get(CACHE, headers=header)
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"
    finally:
        settings.cache_clear()


async def test_the_right_token_gets_in(client: AsyncClient, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    settings.cache_clear()
    try:
        resp = await client.get(CACHE, headers={"authorization": "Bearer s3cret"})
        assert resp.status_code == 200
    finally:
        settings.cache_clear()


async def test_an_unset_token_leaves_v1_open(client: AsyncClient):
    """What a first `make up` and the test suite run on. The server warns at
    startup rather than leaving it silent."""
    assert settings().api_token == ""
    assert (await client.get(CACHE)).status_code == 200


# ---------------------------------------------------------------- GET /v1/cache


async def test_cache_is_empty_before_anything_is_learned(client: AsyncClient):
    body = (await client.get(CACHE)).json()
    assert body["entries"] == []
    assert body["summary"] == {"total": 0, "verified": 0, "stale": 0, "disabled": 0}


async def test_cache_lists_entries_as_the_model_sees_them(client: AsyncClient, conn):
    await seed_entries(conn)
    body = (await client.get(CACHE)).json()

    assert body["summary"]["total"] == 2
    assert body["summary"]["verified"] == 1

    entry = next(e for e in body["entries"] if e["name"] == "active customer")
    assert entry["kind"] == "recipe"
    assert entry["sql_fragment"] == "customer WHERE deleted_at IS NULL"
    assert entry["tables"] == ["customer"]
    assert entry["verified"] is True
    assert entry["stale"] is False
    assert entry["hits"] == 0


async def test_the_listing_is_ordered_by_hits_like_load_cache(client: AsyncClient, conn):
    """Same order the model reads them in — that equivalence is what `make cache`
    is for."""
    await seed_entries(conn)
    cur = await conn.execute("SELECT id FROM cache_entry WHERE name = 'orders.created'")
    await store.bump_hits(conn, [(await cur.fetchone())["id"]], connection_id=CID)

    body = (await client.get(CACHE)).json()
    assert [e["name"] for e in body["entries"]] == ["orders.created", "active customer"]


async def test_tombstones_are_listed_and_disabled_entries_are_only_counted(
    client: AsyncClient, conn
):
    """A tombstone is content — the model has to see it. A disabled entry is not,
    but it shouldn't go quietly missing either."""
    await store.write_entries(
        conn,
        [
            store.CacheEntry(
                kind="recipe",
                name="revenue",
                claim="revenue does not include cancelled orders",
                tables=["orders"],
                tombstone=True,
            ),
            store.CacheEntry(
                kind="schema_fact",
                name="switched off",
                claim="not in play",
                tables=["orders"],
                disabled=True,
            ),
        ],
        connection_id=CID,
    )
    body = (await client.get(CACHE)).json()

    assert [e["name"] for e in body["entries"]] == ["revenue"]
    assert body["entries"][0]["tombstone"] is True
    assert body["summary"] == {"total": 1, "verified": 0, "stale": 0, "disabled": 1}


async def test_kind_filters_the_entries_but_not_the_summary(client: AsyncClient, conn):
    """Filtering the view must not misreport how big the cache is."""
    await seed_entries(conn)
    body = (await client.get(CACHE, params={"kind": "recipe"})).json()

    assert [e["name"] for e in body["entries"]] == ["active customer"]
    assert body["summary"]["total"] == 2

    assert (await client.get(CACHE, params={"kind": "nonsense"})).status_code == 422


async def test_a_renamed_column_marks_its_entry_stale(
    client: AsyncClient, conn, target_conn
):
    """The §5 drift check, as the listing shows it, and the clearest case of the
    endpoint spanning both servers: the entry is on one, the schema it describes
    is on the other. Phase 6 acts on it; here it is reported."""
    await target_conn.execute(f"CREATE TABLE {PROBE} (id bigint, created timestamptz)")
    entry = store.CacheEntry(
        kind="schema_fact",
        name="probe shape",
        claim="the probe table has a created column",
        tables=[PROBE],
    )
    await store.fingerprint_entries(target_conn, [entry])
    await store.write_entries(conn, [entry], connection_id=CID)

    body = (await client.get(CACHE)).json()
    assert body["entries"][0]["stale"] is False
    assert body["summary"]["stale"] == 0

    await target_conn.execute(f"ALTER TABLE {PROBE} RENAME COLUMN created TO created_at")

    body = (await client.get(CACHE)).json()
    assert body["entries"][0]["stale"] is True
    assert body["summary"]["stale"] == 1


async def test_internal_bookkeeping_stays_off_the_wire(client: AsyncClient, conn):
    """`schema_fp` and the turn pointers exist for the graph. Putting them on the
    wire would make them part of a versioned contract."""
    await seed_entries(conn)
    entry = (await client.get(CACHE)).json()["entries"][0]
    for field in ("schema_fp", "created_turn", "last_used_turn", "created_at"):
        assert field not in entry


# ------------------------------------------------------------- DELETE /v1/cache


async def test_delete_wipes_learned_state_and_reports_what_it_took(
    client: AsyncClient, conn
):
    await seed_entries(conn)
    await store.start_turn(conn, connection_id=CID,
                          session_id="11111111-1111-1111-1111-111111111111",
                           question="how many customers do we have?")

    wiped = (await client.delete(CACHE)).json()["wiped"]
    assert wiped["cache_entry"] == 2
    assert wiped["turn"] >= 1
    # The checkpoints go too: an empty cache beside a turn log saying the
    # questions were already asked is a state nothing knows how to read.
    assert "checkpoints" in wiped

    assert (await client.get(CACHE)).json()["entries"] == []


async def test_delete_leaves_the_business_data_alone(client: AsyncClient, target_conn):
    """The stage recovery button forgets what the agent learned. It does not
    touch the database the agent is answering questions about.

    That used to rest on `reset_learned` naming the right six tables. It now
    rests on the wiring: the connection it runs on cannot reach this server.
    """
    cur = await target_conn.execute("SELECT count(*) AS n FROM customer")
    before = (await cur.fetchone())["n"]
    assert before > 0

    await client.delete(CACHE)

    cur = await target_conn.execute("SELECT count(*) AS n FROM customer")
    assert (await cur.fetchone())["n"] == before
