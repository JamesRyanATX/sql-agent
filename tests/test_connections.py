"""The registry, and the one rule it must never break.

Registering a connection means handing the agent somebody else's database
credentials. The password goes in, and it never comes back out — not masked, not
truncated, not in an error message. `ConnectionOut` has no such field at all,
which is a stronger guarantee than remembering to blank one, and these tests
check the whole response body rather than the field, because the leak that
matters is the one nobody thought to blank.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from psycopg.conninfo import conninfo_to_dict

from app import db, store
from app.settings import settings
from tests.conftest import DEFAULT_CONNECTION, OTHER_CONNECTION

# A password that appears nowhere else in a response. The reader role's real
# password is the string "reader", which is also its username — so a leak test
# written against it would pass on a response that echoed the username and
# nothing else.
SENTINEL = "correct-horse-battery-staple"


def payload(cid: str = "warehouse", **kw) -> dict:
    """A registration pointing at the suite's own business database.

    Derived from TARGET_DATABASE_URL rather than written out: the tests run on
    the host and the app runs in a container, so `demo-db` and `localhost` are
    the same server under two names, and hardcoding either breaks in one of the
    two places. The password is the real one, so the probe tests can connect.
    """
    d = conninfo_to_dict(settings().target_database_url)
    return {
        "id": cid,
        "host": d["host"],
        "port": int(d.get("port") or 5432),
        "database": d["dbname"],
        "username": d["user"],
        "password": d["password"],
        **kw,
    }


@pytest.fixture
async def registered(client: AsyncClient, agent_conn):
    """A connection that exists for the length of one test."""
    created = []

    async def make(cid: str = "warehouse", **kw):
        resp = await client.post(
            "/v1/connections", params={"probe": "false"}, json=payload(cid, **kw)
        )
        assert resp.status_code == 201, resp.text
        created.append(cid)
        return resp

    yield make
    for cid in created:
        await agent_conn.execute("DELETE FROM connection WHERE id = %s", (cid,))
        await db.evict(cid)


# ------------------------------------------------------------------- the CRUD


async def test_create_returns_201_and_a_location_header(client, registered):
    resp = await registered()
    p = payload()
    assert resp.headers["location"] == "/v1/connections/warehouse"
    body = resp.json()["connection"]
    assert body["id"] == "warehouse"
    assert body["origin"] == "api"
    assert body["has_password"] is True
    assert body["dsn"] == f"postgresql://{p['username']}@{p['host']}:{p['port']}/{p['database']}"
    assert resp.json()["test"] is None  # probe=false


async def test_a_new_connection_starts_with_nothing_learned(client, registered):
    await registered()
    body = (await client.get("/v1/connections/warehouse")).json()
    assert body["cache_entries"] == 0 and body["turns"] == 0
    assert (await client.get("/v1/connections/warehouse/cache")).json()["entries"] == []
    assert (await client.get("/v1/connections/warehouse/turns")).json()["turns"] == []


async def test_the_password_never_comes_back(client, registered):
    """Checked against the whole response text, not against the field.

    A `password` key is easy to remember to omit. The leak that gets shipped is
    a `dsn` built by f-string, or an error message quoting the conninfo.
    """
    create = await registered(password=SENTINEL)
    get = await client.get("/v1/connections/warehouse")
    listing = await client.get("/v1/connections")
    patch = await client.patch("/v1/connections/warehouse", json={"port": 5433})

    for resp in (create, get, listing, patch):
        assert SENTINEL not in resp.text, resp.request.url
        assert "password" not in resp.json().get("connection", resp.json())


async def test_the_password_is_not_stored_in_plaintext(client, agent_conn, monkeypatch):
    """With CONNECTION_SECRET set, the column holds a token — and the connection
    still works, which is the half that catches a seal without an unseal."""
    from cryptography.fernet import Fernet

    monkeypatch.setenv("CONNECTION_SECRET", Fernet.generate_key().decode())
    settings.cache_clear()
    try:
        resp = await client.post(
            "/v1/connections",
            params={"probe": "false"},
            json=payload("sealed", password=SENTINEL),
        )
        assert resp.status_code == 201

        cur = await agent_conn.execute(
            "SELECT password FROM connection WHERE id = 'sealed'"
        )
        stored = (await cur.fetchone())["password"]
        assert SENTINEL not in stored
        assert stored.startswith("fernet:")

        # And it round-trips: the address resolves back to something connectable.
        assert (await db.resolve("sealed")).password == SENTINEL
    finally:
        await agent_conn.execute("DELETE FROM connection WHERE id = 'sealed'")
        settings.cache_clear()


async def test_a_missing_secret_stores_plaintext_and_says_so(client, agent_conn, registered):
    """The same bargain API_TOKEN ships: an insecure default is fine when it is
    loud. The startup warning is in app/main.py; this is the behaviour."""
    assert settings().connection_secret == ""
    await registered(password=SENTINEL)
    cur = await agent_conn.execute(
        "SELECT password FROM connection WHERE id = 'warehouse'"
    )
    assert (await cur.fetchone())["password"] == f"plain:{SENTINEL}"


async def test_duplicate_id_is_409(client, registered):
    await registered()
    resp = await client.post("/v1/connections", json=payload("warehouse"))
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


@pytest.mark.parametrize("cid", ["Warehouse", "-nope", "with space", "", "a" * 64])
async def test_a_connection_id_must_be_a_slug(client, cid):
    """It is a path segment and something a user types on every command."""
    assert (await client.post("/v1/connections", json=payload(cid))).status_code == 422


async def test_an_unknown_patch_field_is_422(client, registered):
    """extra='forbid', so a typo is an error rather than a silent no-op that
    looks like the PATCH worked."""
    await registered()
    resp = await client.patch("/v1/connections/warehouse", json={"hostname": "elsewhere"})
    assert resp.status_code == 422


async def test_patch_changes_only_what_it_was_sent(client, registered, agent_conn):
    await registered()
    body = (await client.patch("/v1/connections/warehouse", json={"port": 5433})).json()
    assert body["port"] == 5433
    assert body["host"] == payload()["host"] and body["username"] == "reader"
    # An omitted password keeps the stored one, so changing a port does not mean
    # re-sending a secret.
    assert body["has_password"] is True


async def test_patch_evicts_the_pool_so_the_next_turn_uses_the_new_address(
    client, registered
):
    await registered()
    await db.target_pool("warehouse")
    assert "warehouse" in db._target_pools

    await client.patch("/v1/connections/warehouse", json={"port": 5433})
    assert "warehouse" not in db._target_pools


async def test_the_env_owned_default_cannot_be_patched_or_deleted(client):
    """Its address is TARGET_DATABASE_URL. Two ways to configure one connection
    is exactly the fallback app/settings.py argues against."""
    for resp in (
        await client.patch(f"/v1/connections/{DEFAULT_CONNECTION}", json={"port": 1}),
        await client.delete(f"/v1/connections/{DEFAULT_CONNECTION}"),
    ):
        assert resp.status_code == 409
        assert "TARGET_DATABASE_URL" in resp.json()["detail"]


async def test_delete_reports_what_it_wiped_and_leaves_others_alone(
    client, registered, agent_conn
):
    await registered()
    await store.write_entries(
        agent_conn,
        [store.CacheEntry(kind="recipe", name="spec:doomed", claim="x")],
        connection_id="warehouse",
    )
    await store.write_entries(
        agent_conn,
        [store.CacheEntry(kind="recipe", name="spec:spared", claim="y")],
        connection_id=OTHER_CONNECTION,
    )

    wiped = (await client.delete("/v1/connections/warehouse")).json()["wiped"]
    assert wiped["cache_entry"] == 1

    assert (await client.get("/v1/connections/warehouse")).status_code == 404
    survivors = await store.load_cache(agent_conn, connection_id=OTHER_CONNECTION)
    assert [e.name for e in survivors] == ["spec:spared"]

    await agent_conn.execute("DELETE FROM cache_entry WHERE name = 'spec:spared'")


# ------------------------------------------------------------------ the routes


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/v1/connections/nope/ask"),
        ("get", "/v1/connections/nope/cache"),
        ("delete", "/v1/connections/nope/cache"),
        ("get", "/v1/connections/nope/turns"),
        ("get", "/v1/connections/nope"),
        ("patch", "/v1/connections/nope"),
        ("delete", "/v1/connections/nope"),
        ("post", "/v1/connections/nope/test"),
    ],
)
async def test_unknown_connection_is_404_on_every_scoped_route(client, method, path):
    """Parametrized over the whole sub-router deliberately: a route added later
    without `connection_dep` fails here rather than 500ing in production."""
    resp = await getattr(client, method)(path, **({"json": {"question": "hi"}} if method in ("post", "patch") else {}))
    assert resp.status_code == 404, path
    assert "no connection named 'nope'" in resp.json()["detail"]


async def test_the_connections_routes_need_the_token_too(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    settings.cache_clear()
    try:
        assert (await client.get("/v1/connections")).status_code == 401
        assert (await client.post("/v1/connections", json=payload())).status_code == 401
        ok = await client.get(
            "/v1/connections", headers={"authorization": "Bearer s3cret"}
        )
        assert ok.status_code == 200
    finally:
        settings.cache_clear()


async def test_an_unreachable_warehouse_is_502_before_the_stream_opens(client, registered):
    """Once the generator is handed to Starlette a 200 is already on the wire.
    A dead warehouse has to be a status code, not an error event inside a
    successful response."""
    await registered("dead", port=1, password=SENTINEL)
    resp = await client.post("/v1/connections/dead/ask", json={"question": "anything?"})
    assert resp.status_code == 502
    assert "dead" in resp.json()["detail"]
    # The exception psycopg raises quotes the conninfo it failed on.
    assert SENTINEL not in resp.text


# ------------------------------------------------------------------- the probe


async def test_test_reports_a_reachable_connection(client, registered):
    await registered()
    body = (await client.post("/v1/connections/warehouse/test")).json()
    assert body["ok"] is True
    assert body["username"] == "reader"
    assert body["tables"] > 0
    assert body["read_only"] is True and body["warnings"] == []


async def test_test_reports_a_bad_password_as_200_and_never_echoes_it(client, registered):
    """A failed diagnostic is a successful diagnostic. A non-2xx would make a
    client print "502 from /v1/..." where the useful sentence is what the
    server said."""
    await registered("wrongpw", password=SENTINEL)
    resp = await client.post("/v1/connections/wrongpw/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False and body["error"]
    assert SENTINEL not in resp.text


async def test_test_warns_when_the_role_can_write(client, registered):
    """Advisory, never a gate — a warehouse role with INSERT on one unrelated
    table is common, and refusing it would mean the feature does not work. The
    guarantee is the read-only session in app/db.py."""
    owner = conninfo_to_dict(settings().target_admin_url)
    await registered("owner", username=owner["user"], password=owner["password"])
    body = (await client.post("/v1/connections/owner/test")).json()
    assert body["ok"] is True
    assert body["read_only"] is False
    assert any("INSERT" in w for w in body["warnings"])


async def test_probe_on_create_does_not_block_a_bad_address(client, agent_conn):
    """A typo should be visible now, while the user is looking at it. Refusing
    to save because the warehouse is down for maintenance is worse."""
    resp = await client.post("/v1/connections", json=payload("offline", port=1))
    try:
        assert resp.status_code == 201
        assert resp.json()["test"]["ok"] is False
        assert (await client.get("/v1/connections/offline")).status_code == 200
    finally:
        await agent_conn.execute("DELETE FROM connection WHERE id = 'offline'")
