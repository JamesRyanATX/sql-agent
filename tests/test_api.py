"""The /v1 surface: auth, what the cache listing says, and what DELETE takes.

These run in-process over ASGI — no server, no container. The point is the
contract, since `scripts/*` are now clients of it and nothing else checks the
shape they parse.

Requires Postgres up (`make up`).
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from httpx import AsyncClient
from psycopg import AsyncConnection

from app import api, store
from app.settings import settings
from tests.conftest import DEFAULT_CONNECTION as CID
from tests.conftest import DEMO
from tests.conftest import OTHER_CONNECTION as OTHER

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
    entry from one test is an input to the next.

    The turn log too, since the feedback tests below write turns: the chart is
    "every turn this connection took", so a leftover row is an extra line in
    somebody else's assertion about what the demo prints.
    """
    await agent_conn.execute("TRUNCATE cache_entry RESTART IDENTITY CASCADE")
    yield
    await agent_conn.execute("TRUNCATE cache_entry RESTART IDENTITY CASCADE")
    await agent_conn.execute("DELETE FROM turn WHERE connection_id = %s", (CID,))
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
        # Scoped and unscoped alike: the dependency is on the router.
        for path in (CACHE, "/v1/config"):
            resp = await client.get(path, headers=header)
            assert resp.status_code == 401, path
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


# --------------------------------------------------------------- GET /v1/config


@pytest.fixture
async def config_client(config_dir, client):
    """`config_dir` first, so the lifespan's `config()` reads the scratch dir
    and not whatever overlay the developer happens to have."""
    return config_dir, client


async def test_config_is_the_tracked_file_when_there_is_no_overlay(config_client):
    _, client = config_client
    body = (await client.get("/v1/config")).json()
    assert body["overlay"] is None
    assert body["overridden"] == []
    assert body["config"]["model"]["provider"] == "anthropic"
    assert body["config"]["explore"]["effort"] == "high"
    # Nulls stay on the wire, so the shape never changes with the file.
    assert body["config"]["explore"]["model"] is None
    assert body["config"]["plan"] == {"effort": None, "model": None}


async def test_config_reports_what_the_overlay_decided(config_client):
    write, client = config_client
    write(DEMO, local={
        "model": {"provider": "openai_compat", "model": "qwen3:32b",
                  "url": "http://10.0.0.1:11434/v1"},
        "explore": {"effort": "low"},
    })
    body = (await client.get("/v1/config")).json()
    assert body["overlay"].endswith("config.local.yaml")
    assert body["overridden"] == [
        "explore.effort", "model.model", "model.provider", "model.url",
    ]
    assert body["config"]["model"]["model"] == "qwen3:32b"
    assert body["config"]["explore"]["effort"] == "low"
    # The layer underneath survives, key by key.
    assert body["config"]["extract"]["effort"] == "low"
    assert body["config"]["model"]["max_tokens"] == 16_000


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
    client: AsyncClient, conn, target_conn, reader_conn
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
    # Fingerprinted as the reader, on SQLAlchemy — which is what `extract`
    # does. `target_conn` is the owner, and only ever runs DDL here.
    await store.fingerprint_entries(reader_conn, [entry])
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


# ------------------------------------------------- POST .../turns/{id}/feedback


@pytest.fixture
def scores(monkeypatch):
    """`tracing.score` as a recorder. Nothing here reaches Langfuse — the suite
    runs with tracing off, so the real one would refuse and the route would 409
    on every case below."""
    recorded: list[dict] = []
    monkeypatch.setattr(
        api.tracing, "score", lambda **kw: bool(recorded.append(kw) or True)
    )
    return recorded


async def a_turn(conn: AsyncConnection, *, connection_id=CID, trace_id=None) -> int:
    turn_id = await store.start_turn(
        conn,
        connection_id=connection_id,
        session_id=uuid4(),
        question="how many customers do we have?",
    )
    await store.finish_turn(conn, turn_id, answer="1,840", trace_id=trace_id)
    return turn_id


def feedback(turn_id: int, cid: str = CID) -> str:
    return f"/v1/connections/{cid}/turns/{turn_id}/feedback"


async def test_a_wrong_answer_files_the_prose_with_the_verdict(
    client: AsyncClient, conn, scores
):
    """The comment is the point of a 0: it is what a later optimisation reads,
    and a bare 0 says only that something was wrong."""
    turn_id = await a_turn(conn, trace_id="0123456789abcdef" * 2)

    resp = await client.post(
        feedback(turn_id),
        json={"correct": False, "comment": "counted the cancelled orders"},
    )

    assert resp.status_code == 202, resp.text
    assert resp.json() == {
        "trace_id": "0123456789abcdef" * 2,
        "name": "correct",
        "value": 0.0,
    }
    assert scores == [
        {
            "trace_id": "0123456789abcdef" * 2,
            "name": "correct",
            "value": 0.0,
            "comment": "counted the cancelled orders",
        }
    ]


async def test_an_approved_answer_needs_no_prose(client: AsyncClient, conn, scores):
    turn_id = await a_turn(conn, trace_id="f" * 32)

    resp = await client.post(feedback(turn_id), json={"correct": True})

    assert resp.status_code == 202, resp.text
    assert scores[0]["value"] == 1.0
    assert scores[0]["comment"] is None


async def test_a_turn_with_no_trace_has_nowhere_to_put_a_verdict(
    client: AsyncClient, conn, scores
):
    """Tracing was off when it ran. 409 rather than a quiet 202: the verdict is
    lost either way, and only one of those says so."""
    turn_id = await a_turn(conn, trace_id=None)

    resp = await client.post(feedback(turn_id), json={"correct": True})

    assert resp.status_code == 409
    assert "no trace" in resp.json()["detail"]
    assert scores == []


async def test_tracing_off_now_is_the_same_answer(client: AsyncClient, conn, monkeypatch):
    """The row has a trace id, but this process cannot reach Langfuse. The real
    `tracing.score` returns False, and the route must not report success."""
    turn_id = await a_turn(conn, trace_id="a" * 32)

    resp = await client.post(feedback(turn_id), json={"correct": True})

    assert resp.status_code == 409


async def test_a_verdict_cannot_reach_another_connections_turn(
    client: AsyncClient, conn, scores
):
    """Turn ids are global and warehouses are not. Routed through `other`, a
    turn belonging to `default` does not exist."""
    turn_id = await a_turn(conn, trace_id="b" * 32)

    resp = await client.post(feedback(turn_id, OTHER), json={"correct": True})

    assert resp.status_code == 404
    assert str(turn_id) in resp.json()["detail"]
    assert scores == []


async def test_an_unknown_turn_is_a_404(client: AsyncClient, scores):
    assert (await client.post(feedback(10**9), json={"correct": True})).status_code == 404
    assert scores == []


async def test_a_misspelled_field_is_refused(client: AsyncClient, conn, scores):
    """`extra="forbid"`, so a client sending `verdict` learns it here rather
    than by watching the corpus never fill up."""
    turn_id = await a_turn(conn, trace_id="c" * 32)

    resp = await client.post(feedback(turn_id), json={"verdict": "ok"})

    assert resp.status_code == 422
    assert scores == []
