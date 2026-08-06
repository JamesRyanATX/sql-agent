"""The /v1 surface: auth, what the cache listing says, and what DELETE takes.

These run in-process over ASGI — no server, no container. The point is the
contract, since `scripts/*` are now clients of it and nothing else checks the
shape they parse.

Requires Postgres up (`make up`).
"""

from collections.abc import AsyncIterator

import psycopg
import pytest
from httpx import AsyncClient
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from app import store
from app.settings import settings

PROBE = "api_probe"


@pytest.fixture
async def conn() -> AsyncIterator[AsyncConnection]:
    """Committing, unlike test_store's: the endpoint reads on its own connection
    and would not see an open transaction's writes. Cleaned up by `clean`."""
    async with await psycopg.AsyncConnection.connect(
        settings().database_url, row_factory=dict_row, autocommit=True
    ) as c:
        yield c


@pytest.fixture(autouse=True)
async def clean(conn: AsyncConnection) -> AsyncIterator[None]:
    """An empty cache each side. The graph reads the cache in full, so a leftover
    entry from one test is an input to the next."""
    await conn.execute("TRUNCATE cache_entry RESTART IDENTITY CASCADE")
    yield
    await conn.execute("TRUNCATE cache_entry RESTART IDENTITY CASCADE")
    await conn.execute(f"DROP TABLE IF EXISTS {PROBE}")


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
        resp = await client.get("/v1/cache", headers=header)
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"
    finally:
        settings.cache_clear()


async def test_the_right_token_gets_in(client: AsyncClient, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    settings.cache_clear()
    try:
        resp = await client.get("/v1/cache", headers={"authorization": "Bearer s3cret"})
        assert resp.status_code == 200
    finally:
        settings.cache_clear()


async def test_an_unset_token_leaves_v1_open(client: AsyncClient):
    """What a first `make up` and the test suite run on. The server warns at
    startup rather than leaving it silent."""
    assert settings().api_token == ""
    assert (await client.get("/v1/cache")).status_code == 200


# ---------------------------------------------------------------- GET /v1/cache


async def test_cache_is_empty_before_anything_is_learned(client: AsyncClient):
    body = (await client.get("/v1/cache")).json()
    assert body["entries"] == []
    assert body["summary"] == {"total": 0, "verified": 0, "stale": 0, "disabled": 0}


async def test_cache_lists_entries_as_the_model_sees_them(client: AsyncClient, conn):
    await seed_entries(conn)
    body = (await client.get("/v1/cache")).json()

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
    await store.bump_hits(conn, [(await cur.fetchone())["id"]])

    body = (await client.get("/v1/cache")).json()
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
    )
    body = (await client.get("/v1/cache")).json()

    assert [e["name"] for e in body["entries"]] == ["revenue"]
    assert body["entries"][0]["tombstone"] is True
    assert body["summary"] == {"total": 1, "verified": 0, "stale": 0, "disabled": 1}


async def test_kind_filters_the_entries_but_not_the_summary(client: AsyncClient, conn):
    """Filtering the view must not misreport how big the cache is."""
    await seed_entries(conn)
    body = (await client.get("/v1/cache", params={"kind": "recipe"})).json()

    assert [e["name"] for e in body["entries"]] == ["active customer"]
    assert body["summary"]["total"] == 2

    assert (await client.get("/v1/cache", params={"kind": "nonsense"})).status_code == 422


async def test_a_renamed_column_marks_its_entry_stale(client: AsyncClient, conn):
    """The §5 drift check, as the listing shows it. Phase 6 acts on it; here it
    is reported."""
    await conn.execute(f"CREATE TABLE {PROBE} (id bigint, created timestamptz)")
    await store.write_entries(
        conn,
        [
            store.CacheEntry(
                kind="schema_fact",
                name="probe shape",
                claim="the probe table has a created column",
                tables=[PROBE],
            )
        ],
    )
    body = (await client.get("/v1/cache")).json()
    assert body["entries"][0]["stale"] is False
    assert body["summary"]["stale"] == 0

    await conn.execute(f"ALTER TABLE {PROBE} RENAME COLUMN created TO created_at")

    body = (await client.get("/v1/cache")).json()
    assert body["entries"][0]["stale"] is True
    assert body["summary"]["stale"] == 1


async def test_internal_bookkeeping_stays_off_the_wire(client: AsyncClient, conn):
    """`schema_fp` and the turn pointers exist for the graph. Putting them on the
    wire would make them part of a versioned contract."""
    await seed_entries(conn)
    entry = (await client.get("/v1/cache")).json()["entries"][0]
    for field in ("schema_fp", "created_turn", "last_used_turn", "created_at"):
        assert field not in entry


# ------------------------------------------------------------- DELETE /v1/cache


async def test_delete_wipes_learned_state_and_reports_what_it_took(
    client: AsyncClient, conn
):
    await seed_entries(conn)
    await store.start_turn(conn, session_id="11111111-1111-1111-1111-111111111111",
                           question="how many customers do we have?")

    wiped = (await client.delete("/v1/cache")).json()["wiped"]
    assert wiped["cache_entry"] == 2
    assert wiped["turn"] >= 1
    # The checkpoints go too: an empty cache beside a turn log saying the
    # questions were already asked is a state nothing knows how to read.
    assert "checkpoints" in wiped

    assert (await client.get("/v1/cache")).json()["entries"] == []


async def test_delete_leaves_the_business_schema_alone(client: AsyncClient, conn):
    """The stage recovery button forgets what the agent learned. It does not
    touch the database the agent is answering questions about."""
    cur = await conn.execute("SELECT count(*) AS n FROM customer")
    before = (await cur.fetchone())["n"]
    assert before > 0

    await client.delete("/v1/cache")

    cur = await conn.execute("SELECT count(*) AS n FROM customer")
    assert (await cur.fetchone())["n"] == before
