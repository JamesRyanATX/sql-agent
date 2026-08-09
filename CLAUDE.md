# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A LangGraph agent that answers business questions against a Postgres database it
has never seen, and gets cheaper each turn by writing down what it learned as
plain-English cache entries. The demo is a token counter going down: T1 explores
(~11.5k tokens), T2 answers the same question from cache (~371), T3 answers a
*new* question by composing cached recipes (~475).

[PLAN.md](PLAN.md) is the design document and build plan — its section numbers
(§4 the graph, §5 cache hygiene, §7.1 model request shape) are referenced from
docstrings throughout the code. Read the relevant section before changing a node.

## Commands

```bash
make up && make migrate && make seed   # two databases + API, then the demo data
make test                              # pytest, excludes live model calls
make test-live                         # includes tests that spend real tokens
make customer-count                    # ask the cold-path question
make connections                       # every database the agent can be pointed at
make cache                             # print the cache as the model sees it
make turns                             # tokens per turn, from the turn table
make reset                             # wipe what `default` learned + reseed (stage button)
make psql-agent / psql-demo            # a shell on either database
make logs-api                          # the reload log
make build                             # rebuild the API image (dependency changes)
```

Everything past `make up` is the CLI wearing a Makefile: `make cache` is
`sql-agent cache -c default`. `CONN=other make cache` retargets the lot.

`make migrate` applies `migrations/*.sql` to **agent-db only**. `make seed`
applies `demo/demo.sql` to **demo-db** — that one file is role, schema and data
together, and there is no Python seeder any more.

`app/` is bind-mounted into the `api` container and uvicorn runs `--reload`, so
an edit restarts the server in place. There is deliberately no second way to run
it — `make dev` was removed when the clients became HTTP clients.

One test: `uv run pytest tests/test_store.py::test_named_entries_upsert_rather_than_duplicate -q`.
`addopts = -m 'not live'` in [pyproject.toml](pyproject.toml) means live tests are
opt-in; run them with `-m live`.

Ad-hoc questions: `uv run sql-agent "how many customers do we have?"` — needs
the API up, since the CLI is an HTTP client, and needs `SQL_AGENT_URL` set and a
connection selected (`sql-agent connect default`).

The tests do *not* need the API container: they mount the app over ASGI in-process
via the `client` fixture in [tests/conftest.py](tests/conftest.py). Both databases
must be up — conftest builds `agent_test` and `business_test` per session (schema,
checkpoint tables, and `demo.sql`) and repoints all three URLs at them. Never
point tests at the dev databases: the cache is global state the graph reads in
full, so per-test scoping cannot make a shared database safe.

Connection fixtures mirror the split: `agent_conn`, `target_conn` (owner, for
DDL) and `reader_conn` (the SELECT-only role the agent actually uses). Reach for
`reader_conn` when the test is about what the agent can see.

## One memory, N targets

**The agent's memory and the data it queries are on separate Postgres servers.**
This is the single most important thing to know before changing anything in
`app/`.

| | agent-db :5433 | a target, e.g. demo-db :5432 |
|---|---|---|
| holds | `connection`, `cache_entry`, `turn`, checkpoints | `customer`, `orders`, 38 decoys |
| built by | `migrations/*.sql` | `demo/demo.sql`, locally |
| reached via | `db.agent()` | `db.target(cid)` / `db.target_readonly(cid)` |
| how many | one | however many are registered |

There is a registry now. `TARGET_DATABASE_URL` is the address of exactly one row
in it — `default`, marked `origin='env'`, immutable over HTTP because the
environment owns it. Everything else is registered at runtime through
`POST /v1/connections` and lives in the `connection` table with its password
sealed by [app/secrets.py](app/secrets.py).

**Two rules, and both fail quietly when broken.**

*Every `db.` call names its server, and a target call names which target.*
`db.target(cid)` has no default argument and never will: a default is how a
mis-scoped node queries the demo warehouse, looks correct on stage, and answers
a customer's question against somebody else's data.

*Every function in [app/store.py](app/store.py) belongs to one server or the
other.* `schema_fingerprint`, `fingerprint_entries` and `stale_ids` take a
*target* connection; everything else takes an *agent* one, and everything that
touches learned state also takes a required keyword-only `connection_id`. The
only operation spanning both is `extract` ([app/graph.py](app/graph.py)), which
fingerprints on the target and then writes on the agent — in that order, because
no single connection reaches both.

The fingerprint functions kept their signatures but gained an obligation: the
target connection must be *the entry's own* connection's target. Crossing them
reports every entry stale, or worse, coincidentally not stale.

One naming trap. `app/db.py` uses the word "connection" ~20 times to mean a
psycopg connection, and it now also means a registry row. **Never bind a bare
`connection` variable to a registry row** — it is `connection_id: str` or
`registered: store.Connection`, always.

## Architecture

**The API is the only way in.** `sql_agent_cli/` is an HTTP client that renders
event streams and JSON — it holds no business logic, and the graph, the pools and
the checkpointer exist in exactly one process. It **imports nothing from `app`**,
which [tests/test_cli_isolation.py](tests/test_cli_isolation.py) asserts by
walking the AST; if you find yourself wanting one convenient function from
`app.store`, the logic belongs behind an endpoint instead. It also touches no
database at all — the demo data is [demo/demo.sql](demo/demo.sql), applied with
psql.

```
GET    /v1/connections                   the registry (never a password)
POST   /v1/connections                   register one; probes it, doesn't gate on it
GET    /v1/connections/{id}              one, with cache and turn counts
PATCH  /v1/connections/{id}              partial; evicts the pool
DELETE /v1/connections/{id}              the row and everything learned about it
POST   /v1/connections/{id}/test         reachability; 200 even when ok:false
POST   /v1/connections/{id}/ask          one turn, streamed as SSE
GET    /v1/connections/{id}/cache        exactly as store.load_cache hands it to the model
DELETE /v1/connections/{id}/cache        wipe cache + turns + this connection's checkpoints
GET    /v1/connections/{id}/turns        the demo chart, as rows
GET    /health                           unversioned, unauthenticated
```

**Everything about learned state hangs off the connection it is about**, as a
path segment rather than a query parameter. That makes the unscoped call
*unrepresentable* — you cannot forget the selector, because the route does not
exist without it. Same argument as the auth dependency below, and as the deleted
table filter in `tools.py`.

Everything under `/v1` requires `Authorization: Bearer $API_TOKEN`, enforced by a
router-level dependency in [app/api.py](app/api.py) so a new route is
authenticated by default; `connection_dep` on the nested router does the same for
scoping. An empty `API_TOKEN` disables enforcement and the server logs a warning
at startup, as does an empty `CONNECTION_SECRET`. The dependency reads
`settings()` at request time, not import time, because the suite clears that
`lru_cache`.

One turn = one pass through the graph in [app/graph.py](app/graph.py):

```
load_cache → plan ─(sufficient)→ execute → extract → answer
                 └(insufficient)→ explore → generate_sql → execute
                                             execute ⇄ fix (≤3 attempts)
```

- **[app/graph.py](app/graph.py)** — nodes, prompts, JSON schemas, edges. All
  system prompts and structured-output schemas live here as module constants.
- **[app/llm.py](app/llm.py)** — the *only* module that talks to a model. Two
  backends behind one `complete()`: `anthropic` (demo) and `openai_compat`
  (Ollama/vLLM/LM Studio for local dev). Nodes never see the difference; build
  messages with `assistant_turn()` / `tool_results()` because the two wire
  formats disagree about tool results.
- **[app/store.py](app/store.py)** — the connection registry, cache and turn-log
  reads/writes, plus `stale_ids`, `count_disabled`, `read_turns` and the two
  resets. See the note above for which server and which target each takes.
- **[app/secrets.py](app/secrets.py)** — `seal`/`unseal` for a registered
  warehouse password. Tagged values (`plain:` / `fernet:`), so turning
  encryption on is config rather than a migration.
- **[app/tools.py](app/tools.py)** — the four read-only introspection tools the
  explore loop calls, on a target connection.
- **[app/db.py](app/db.py)** — the agent pool, and a registry of target pools
  keyed by connection id: `agent()`, `target(cid)`, `target_readonly(cid)`,
  `resolve(cid)`, `evict(cid)`.
- **[app/api.py](app/api.py)** / **[app/schemas.py](app/schemas.py)** — the v1
  router, the auth dependency, and the wire models. `CacheEntry` is deliberately
  not the wire format: `schema_fp` and the turn pointers stay off it.
- **[app/events.py](app/events.py)** — graph output → SSE (`plan, explore, sql,
  error, fix, learned, answer, usage, done`).
- **[app/main.py](app/main.py)** — app assembly only: lifespan (pool →
  checkpointer → graph), `/health`, and the router.
- **[sql_agent_cli/](sql_agent_cli/)** — the `sql-agent` command. `http.py` is
  the client (SSE framing lives here, and the ask stream runs with **no read
  timeout** — a T1 turn takes minutes and httpx's 5s default would kill every
  one); `config.py` holds the two `SQL_AGENT_*` variables and the connection
  precedence; `main.py` holds `AskOrCommand`, which decides whether argv is a
  question or a subcommand; the rest render.

### Invariants that are load-bearing

These each exist because of an observed failure; changing them silently breaks
the demo rather than failing a test.

- **`plan` writes the SQL on the cached path**, in the same call as the
  sufficiency decision. Splitting them costs a second round trip that re-sends
  the whole cache. Sufficiency without SQL is downgraded to insufficient.
- **A cold cache skips the `plan` model call entirely** — it can only produce one
  answer, so paying for it would make T1 more expensive for nothing.
- **`extract` is skipped on a fully cached turn** (`sufficient` and no fixes).
  Re-deriving what the cache already held wrote near-duplicates and cost a call
  per turn forever. A turn that needed a fix *did* learn something, so it runs.
- **`grounded_in()` is the verification gate**: a recipe is marked `verified`
  only if its `sql_fragment` is an order-preserving *token subsequence* of the
  SQL that actually ran. Substring matching was tried and rejected — aliases
  break it. Unverified entries are still written; they just carry less authority.
- **A human's `pinned` entry is never overwritten** by extraction
  (`store.write_entries` skips it and omits its id from the return value).
- **Named entries upsert, *per connection*.** Reusing a name replaces the entry,
  so `extract` is shown the already-filed names to stop the model paraphrasing
  its own keys. The unique index is `(connection_id, name)` — it was global once,
  and `revenue` learned about a second warehouse would have overwritten the
  first's recipe through the upsert, silently, because an upsert reports success.
- **`load_cache` loads everything, every turn**, ordered by hits — no retrieval
  step. "Everything" means one connection's: what the agent knows about one
  warehouse is not evidence about another. Tombstones are included deliberately:
  a visible negative constraint is what stops exploration rediscovering the same
  wrong thing.
- **A turn's connection lives in `TurnState`**, not in
  `config["configurable"]` beside `thread_id`. State is checkpointed; a value
  outside it lets a resumed thread silently switch warehouses, replaying a cache
  loaded from one while `execute` runs against another. For the same reason a
  session that has asked about one connection is refused (409) against another.
- **Effort, never thinking-off.** Cost is controlled per node via
  `EFFORT_*` settings. Disabling thinking on Opus 5 can turn a tool call into
  visible text that never executes, which silently breaks the explore loop.
- **The agent cannot see its own tables** — they are on another server. This
  used to be a name filter in `tools.py`; it was deleted, because a filter that
  matches on names also hides a *business* table that happens to be called
  `cache_entry`, and it only works while every call site remembers it.
- **Identifiers are allowlisted against `information_schema`, then quoted with
  `psycopg.sql.Identifier`** — table/column names can't be bound as parameters.
- **`describe_table` reads constraints from `pg_catalog`, not
  `information_schema`.** `information_schema.table_constraints` only shows
  constraints to a caller with a **non-SELECT** privilege, so the read-only role
  sees none — every table would look keyless and the agent would guess joins
  from column names.
- **Read-only is enforced by the session, not by the credentials.** This flipped
  when connections became registrable: `demo/demo.sql` builds a SELECT-only role,
  but a *registered* connection's credentials are whatever the user handed us, so
  "the role can't write either" stopped being true. Every target pool now carries
  a `configure=` callback running `SET SESSION CHARACTERISTICS AS TRANSACTION
  READ ONLY`, which binds any role including a superuser — and covers
  `db.target()`, which `explore`, `infer_tables` and `extract` use and which has
  no transaction of its own. `db.target_readonly()` keeps its
  `transaction_read_only` + `statement_timeout` on top; the timeout has no
  session-level equivalent, and defence that exists in one place is one edit from
  not existing. `tests/test_isolation.py::test_a_user_supplied_dsn_still_cannot_write`
  registers credentials that genuinely can write and is what fails if the
  callback goes.
- **A warehouse's password never leaves the server.** `ConnectionOut` has no
  `password` field at all — not `None`, not masked — because a field that does
  not exist cannot be leaked by a future `**row`. DSNs are built with
  `psycopg.conninfo.make_conninfo`, never an f-string, and connection errors are
  reported by exception type rather than message, which quotes the conninfo.
- **`stream_turn` never raises.** A model timeout closes the open turn row via
  `store.fail_open_turn` and yields a fatal error event; the next question works.
- Tool errors come back as `is_error` tool results, not exceptions — the model
  corrects itself instead of the turn dying on a typo'd column name.

### The five traps

[demo/demo.sql](demo/demo.sql) is deterministic (modular arithmetic over
`generate_series`, never `random()`), so **1,840 active customers is a fact, not
a probability**. Five deliberate traps make naive SQL wrong, and finding them is
what T1's cost buys:

1. `customer.deleted_at` — soft deletes, 160 of 2,000 rows
2. `customer.region` — casing varies (`west` / `West` / `WEST`)
3. `orders.status` — 12.5% `cancelled`, inflating revenue
4. `order_item.price` (historical) vs `product.unit_price` (current)
5. `orders.created`, never `created_at`

[tests/test_traps.py](tests/test_traps.py) asserts each one still bites, and the
report `SELECT` at the bottom of `demo.sql` prints the counts on every `make
seed`. If you change the demo data, those two are the contract.

Note `demo.sql` must stay free of psql meta-commands (`\echo`, `\i`) —
[tests/conftest.py](tests/conftest.py) applies it through psycopg, which can't
parse them.

## Build status

Phases 0–5 of PLAN.md §8 are done: schema, seed, cache tables, cold path,
extraction, and the `plan` node. Not yet built: §5 compaction and schema-drift
invalidation on load (`schema_fp` is written but not checked), the `/admin/cache`
API of §6.2, and the browser UI of Phase 8.

## Conventions

- `uv` for everything (`uv run …`); dependencies in [pyproject.toml](pyproject.toml).
- Migrations are `migrations/*.sql`, applied in filename order on every `make
  migrate`, so every statement must be `IF NOT EXISTS`-idempotent. They are a
  **desired-state script, not an append-only ledger** — editing an earlier file
  is correct when a later one supersedes it, and sometimes required: leaving a
  superseded `CREATE UNIQUE INDEX` in place means a later run recreates what the
  next file dropped, and fails mid-file under `ON_ERROR_STOP=1`.
- Comments explain *why*, usually citing the failure that motivated the code or
  the PLAN.md section it implements. Match that register — don't add comments
  that restate the line below them.
- The demo GIF is re-recorded with `make demo` (needs VHS) and checked with
  `make demo-verify`, which reads the turn table and fails a bad take.
