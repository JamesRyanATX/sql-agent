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
make up && make migrate && make seed   # postgres + API, schema, deterministic seed
make test                              # pytest, excludes live model calls
make test-live                         # includes tests that spend real tokens
make t1                                # ask one question from the CLI
make cache                             # print the cache as the model sees it
make turns                             # tokens per turn, from the turn table
make reset                             # wipe learned state + reseed (stage button)
make logs-api                          # the reload log; make psql for a DB shell
make build                             # rebuild the API image (dependency changes)
```

`app/` is bind-mounted into the `api` container and uvicorn runs `--reload`, so
an edit restarts the server in place. There is deliberately no second way to run
it — `make dev` was removed when the scripts became API clients.

One test: `uv run pytest tests/test_store.py::test_named_entries_upsert_rather_than_duplicate -q`.
`addopts = -m 'not live'` in [pyproject.toml](pyproject.toml) means live tests are
opt-in; run them with `-m live`.

Ad-hoc questions: `uv run python -m scripts.ask "how many customers do we have?"`
— needs the API up, since it is an HTTP client.

The tests do *not* need the API container: they mount the app over ASGI in-process
via the `client` fixture in [tests/conftest.py](tests/conftest.py). The DB must be
up for most of the suite — `tests/conftest.py` creates and drops a
separate `sql_agent_test` database per session and repoints `DATABASE_URL` at it.
Never point tests at the dev database: the cache is global state the graph reads
in full, so per-test scoping cannot make a shared database safe.

## Architecture

**The API is the only way in.** `scripts/*` are HTTP clients that render event
streams and JSON — they hold no business logic, and the graph, the pool and the
checkpointer exist in exactly one process. If you find yourself importing
`app.graph` or `app.store` from a script, the logic belongs behind an endpoint
instead. The one exception is [scripts/seed.py](scripts/seed.py), which loads
*business* fixture data and talks to Postgres directly.

```
POST   /v1/ask      one turn, streamed as SSE
GET    /v1/cache    the cache, exactly as store.load_cache hands it to the model
DELETE /v1/cache    wipe cache + turn log + checkpoints (business schema untouched)
GET    /health      unversioned, unauthenticated
```

Everything under `/v1` requires `Authorization: Bearer $API_TOKEN`, enforced by a
router-level dependency in [app/api.py](app/api.py) so a new route is
authenticated by default. An empty `API_TOKEN` disables enforcement and the
server logs a warning at startup. The dependency reads `settings()` at request
time, not import time, because the suite clears that `lru_cache`.

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
- **[app/store.py](app/store.py)** — cache and turn-log reads/writes, plus
  `stale_ids`, `count_disabled` and `reset_learned`, which back `/v1/cache`.
  Owns `AGENT_TABLES`; `tools.py` imports it from here so the set that is hidden
  from introspection and the set that gets wiped can't drift apart.
- **[app/tools.py](app/tools.py)** — the four read-only introspection tools the
  explore loop calls.
- **[app/db.py](app/db.py)** — pool, plus `readonly()` for agent-generated SQL.
- **[app/api.py](app/api.py)** / **[app/schemas.py](app/schemas.py)** — the v1
  router, the auth dependency, and the wire models. `CacheEntry` is deliberately
  not the wire format: `schema_fp` and the turn pointers stay off it.
- **[app/events.py](app/events.py)** — graph output → SSE (`plan, explore, sql,
  error, fix, learned, answer, usage, done`).
- **[app/main.py](app/main.py)** — app assembly only: lifespan (pool →
  checkpointer → graph), `/health`, and the router.
- **[scripts/_client.py](scripts/_client.py)** — the scripts' side of the API.
  SSE framing lives here, and the ask stream runs with **no read timeout** — a
  T1 turn takes minutes and httpx's 5s default would kill every one.

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
- **Named entries upsert.** Reusing a name replaces the entry, so `extract` is
  shown the already-filed names to stop the model paraphrasing its own keys.
- **`load_cache` loads everything, every turn**, ordered by hits — no retrieval
  step. Tombstones are included deliberately: a visible negative constraint is
  what stops exploration rediscovering the same wrong thing.
- **Effort, never thinking-off.** Cost is controlled per node via
  `EFFORT_*` settings. Disabling thinking on Opus 5 can turn a tool call into
  visible text that never executes, which silently breaks the explore loop.
- **The agent cannot see its own tables.** `tools.AGENT_TABLES` hides
  `cache_entry`, `turn` and the LangGraph checkpoint tables, or the agent caches
  facts about the cache.
- **Identifiers are allowlisted against `information_schema`, then quoted with
  `psycopg.sql.Identifier`** — table/column names can't be bound as parameters.
- **Generated SQL runs inside `db.readonly()`**: `transaction_read_only` plus
  `statement_timeout`, both via `set_config(..., is_local => true)` inside an
  explicit transaction.
- **`stream_turn` never raises.** A model timeout closes the open turn row via
  `store.fail_open_turn` and yields a fatal error event; the next question works.
- Tool errors come back as `is_error` tool results, not exceptions — the model
  corrects itself instead of the turn dying on a typo'd column name.

### The five traps

The seed ([scripts/seed.py](scripts/seed.py)) is deterministic (modular
arithmetic, never `random()`), so **1,840 active customers is a fact, not a
probability**. Five deliberate traps make naive SQL wrong, and finding them is
what T1's cost buys:

1. `customer.deleted_at` — soft deletes, 160 of 2,000 rows
2. `customer.region` — casing varies (`west` / `West` / `WEST`)
3. `orders.status` — 12.5% `cancelled`, inflating revenue
4. `order_item.price` (historical) vs `product.unit_price` (current)
5. `orders.created`, never `created_at`

[tests/test_traps.py](tests/test_traps.py) asserts each one still bites. If you
change the seed, those tests are the contract.

## Build status

Phases 0–5 of PLAN.md §8 are done: schema, seed, cache tables, cold path,
extraction, and the `plan` node. Not yet built: §5 compaction and schema-drift
invalidation on load (`schema_fp` is written but not checked), the `/admin/cache`
API of §6.2, and the browser UI of Phase 8.

## Conventions

- `uv` for everything (`uv run …`); dependencies in [pyproject.toml](pyproject.toml).
- Migrations are `migrations/*.sql`, applied in filename order on every `make
  migrate`, so every statement must be `IF NOT EXISTS`-idempotent.
- Comments explain *why*, usually citing the failure that motivated the code or
  the PLAN.md section it implements. Match that register — don't add comments
  that restate the line below them.
- The demo GIF is re-recorded with `make demo` (needs VHS) and checked with
  `make demo-verify`, which reads the turn table and fails a bad take.
