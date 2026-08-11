# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A LangGraph agent that answers business questions against a Postgres database it
has never seen, and gets cheaper each turn by writing down what it learned as
plain-English cache entries. The demo is a token counter going down: T1 explores
(~11.5k tokens), T2 answers the same question from cache (~371), T3 answers a
*new* question by composing cached recipes (~475). Then it goes back up and down
again on purpose — T4 asks about `orders`, a part of the schema T1–T3 never
touched, and pays ~14.3k for it; T5 turns what T4 learned into a `regr_slope`
projection for ~1.4k without exploring. The counter is not monotonic, and a demo
that implied it was would be selling something else.

[PLAN.md](PLAN.md) is the design document and build plan — its section numbers
(§4 the graph, §5 cache hygiene, §7.1 model request shape) are referenced from
docstrings throughout the code. Read the relevant section before changing a node.

## Commands

```bash
make up && make migrate && make seed   # two databases + API, then the demo data
make test                              # pytest, excludes live model calls
make test-live                         # includes tests that spend real tokens
docker compose --profile mysql up -d   # opt-in; the dialect tests skip without it
make langfuse-up                       # opt-in; the trace stack, UI on :3000
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

**The agent's memory is on its own Postgres server, and it is psycopg. Targets
are SQLAlchemy and may be any of three engines.** This is the single most
important thing to know before changing anything in `app/`.

| | agent-db :5433 | a target |
|---|---|---|
| holds | `connection`, `cache_entry`, `turn`, checkpoints | the business data |
| built by | `migrations/*.sql` | not by us |
| engine | Postgres, always | PostgreSQL, MySQL/MariaDB or SQLite |
| reached via | `db.agent()` | `db.target(cid)` / `db.target_readonly(cid)` |
| hands back | `psycopg.AsyncConnection` | `sqlalchemy.ext.asyncio.AsyncConnection` |
| how many | one | however many are registered |

**The two are physically incompatible, and that is the design.** They share no
method that matters: `execute()` returns an `AsyncCursor` on one and a
`CursorResult` on the other, `fetchone()` is awaitable on one and not the other,
and only the target has `run_sync`. Handing an agent connection to
`store.schema_fingerprint` used to be a quiet bug — a fingerprint of the wrong
schema, stamped onto an entry that was then permanently stale. It is now an
`AttributeError` on the first line. The rule below used to be a thing to
remember; it is now a thing that raises.

Agent-db stays psycopg and stays Postgres because LangGraph's
`AsyncPostgresSaver` uses psycopg3 pipeline mode and has no SQLAlchemy seam, and
because `migrations/` leans on `text[]`, a partial unique index and a conditional
`ON CONFLICT` that implements the pinned-entry rule. There is also no user value
in porting it: you run one agent-db and nobody chooses it.

`TARGET_DATABASE_URL` is the address of exactly one registry row — `default`,
marked `origin='env'`, immutable over HTTP because the environment owns it.
Everything else is registered at runtime through `POST /v1/connections` and lives
in the `connection` table with its password sealed by
[app/secrets.py](app/secrets.py) and its engine named by `driver`.

**Two rules, and both fail quietly when broken.**

*Every `db.` call names its server, and a target call names which target.*
`db.target(cid)` has no default argument and never will: a default is how a
mis-scoped node queries the demo warehouse, looks correct on stage, and answers
a customer's question against somebody else's data.

*Every function in [app/store.py](app/store.py) belongs to one server or the
other.* `reflect_columns`, `schema_fingerprint`, `fingerprint_entries` and
`stale_ids` take a *target* connection; everything else takes an *agent* one, and
everything touching learned state also takes a required keyword-only
`connection_id`. The only operation spanning both is `extract`
([app/graph.py](app/graph.py)), which fingerprints on the target and then writes
on the agent — in that order, because no single connection reaches both.

The fingerprint functions gained an obligation: the target connection must be
*the entry's own* connection's target. Crossing them reports every entry stale,
or worse, coincidentally not stale. The type strings come from the dialect's own
reflection, so a fingerprint is comparable only within one connection — which is
why `migrations/003` NULLs every one written before the port.

One naming trap. `app/db.py` uses the word "connection" ~20 times to mean a
driver connection, and it now also means a registry row. **Never bind a bare
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
  resets. See the note above for which server and which target each takes, and
  note the four that take a *SQLAlchemy* connection.
- **[app/dialects.py](app/dialects.py)** — the capability table: how far
  read-only can be pushed on each engine, what statements do it, and what a user
  must be warned about. **Nothing outside it branches on a dialect name** — if
  you are writing `if dialect == "mysql"` elsewhere, the fact belongs here.
- **[app/secrets.py](app/secrets.py)** — `seal`/`unseal` for a registered
  warehouse password. Tagged values (`plain:` / `fernet:`), so turning
  encryption on is config rather than a migration.
- **[app/tracing.py](app/tracing.py)** — the *only* module that imports
  `langfuse`, on the same rule `dialects.py` has for dialect names. Four context
  managers and a boolean; see Observability below.
- **[app/tools.py](app/tools.py)** — the four read-only introspection tools the
  explore loop calls, on a target connection, through SQLAlchemy's `Inspector`.
- **[app/db.py](app/db.py)** — the agent's psycopg pool, and a registry of
  SQLAlchemy engines keyed by connection id: `agent()`, `target(cid)`,
  `target_readonly(cid)`, `target_engine(cid)`, `resolve(cid)`, `evict(cid)`.
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
  question or a subcommand; the rest render. `render.table` is the one aligned
  formatter — `turns`, `connections ls` and query results all go through it, so
  a psql-shaped `(N rows)` footer is a thing the tape can wait on. Query results
  come through `render.result`, which differs in three ways that are each a
  decision: alignment is decided **by what is in a column, not by its name**
  (`table`'s `right=` allowlist cannot work for SQL the model just wrote, and a
  Postgres numeric arrives as the string `"1234.50"`); cells truncate at
  `render.CELL` so one long value cannot unalign every row or overrun the tape's
  screenful, with `--json` as the untruncated view; and a result that filled
  `max_rows` prints `— more matched` rather than a total, because `fetchmany`
  never counted one.

### Observability

Off unless **both** `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, and
off has to stay free: no client is constructed, `langfuse` is never imported, and
every helper yields the same do-nothing handle. There is deliberately no third
`enabled` flag — a flag is a state that can disagree with the keys, and "on but
unconfigured" is a startup warning nobody reads. Exactly one key set is a
`log.warning`, because `extra="ignore"` makes a typo'd name indistinguishable
from an absent one.

The stack is six containers behind `profiles: ["langfuse"]` — `make langfuse-up`,
same opt-in bargain as `mysql-db`, and the `api` service deliberately does **not**
`depends_on` any of it. Only `langfuse-web` publishes a port; upstream's compose
binds its Postgres to 5432, which is demo-db.

One turn is one trace:

```
turn (span)                          stream_turn — session_id, connection_id
├── load_cache (span)                one per graph node, from `traced()`
├── plan (span) → plan (generation)  the model call, with usage and cache reads
├── explore (span)
│   ├── explore (generation)         one per loop iteration, not one per node
│   └── tool.describe_table (tool)   the 24 the turn table shows as "24"
├── execute (span) → sql.execute     the statement, the row count, the error
└── answer (span) → answer (generation)
```

- **`llm.complete()` is the only place a generation is opened**, which follows
  from it being the only place that talks to a model — one seam, both backends,
  and the `explore` loop's per-call cost visible instead of summed away inside
  the node. The wrapper is on the dispatcher and not in `_complete_anthropic` /
  `_complete_openai`, so there is one thing to keep in step rather than two. The
  `node=` keyword at the seven call sites is the only thing in `llm.py` that
  knows the graph exists.
- **Node spans are hand-rolled** (`traced()` in `graph.py`), not
  `langfuse.langchain.CallbackHandler`: that needs the whole `langchain`
  meta-package, and this project has langgraph and langchain-core only.
- **The trace id is minted in `stream_turn` and carried in `TurnState`**, not
  read back out of the ambient OpenTelemetry context inside `answer`. Same
  argument as `connection_id`: a turn's identity belongs in the checkpointed
  state, and `turn.trace_id` should not depend on context propagation into
  LangGraph's tasks. `tracing.turn()`'s `with` sits **outside** `stream_turn`'s
  `try`, because api.py breaks out of that generator on client disconnect.
- **On means the prompts, the SQL and the rows are captured.** That is what makes
  a trace worth opening and there is no useful half-measure; the store is
  self-hosted, so nothing leaves the machine, but it holds whatever the
  registered warehouse holds.

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
- **Identifiers are allowlisted against the dialect's own reflection, then
  carried as `quoted_name(..., quote=True)`** — table/column names can't be bound
  as parameters, so only a name the database just told us exists may reach a
  statement. `quoted_name` travels with the expression tree, so quoting happens
  at compile time against whichever dialect runs it. "Exists" is no longer
  case-blind: exact match wins, a unique case-insensitive match is accepted, and
  two is an error naming both — guessing there is how the agent silently reads
  the wrong table.
- **Constraints come from SQLAlchemy's reflection, and the reason the old
  hand-written `pg_catalog` query existed still holds.**
  `information_schema.table_constraints` only shows constraints to a caller with
  a **non-SELECT** privilege, so the read-only role sees none — every table would
  look keyless and the agent would guess joins from column names. SQLAlchemy's
  PostgreSQL dialect reads `pg_catalog` for that exact reason, and every other
  dialect has its own answer to its own version of the problem. The invariant did
  not go away; it stopped being ours to maintain.
- **Read-only is enforced as far as the dialect allows, and reported.** This has
  flipped twice now. It rested on the credentials — `demo/demo.sql` builds a
  SELECT-only role — until connections became registrable and the credentials
  became whatever the user handed us. It then rested on a session-level setting,
  until a second and third engine made *that* only partly true. So it is a
  capability, and [app/dialects.py](app/dialects.py) is the only place that knows
  the difference:

  | | blocks DML | blocks DDL | statement timeout | tier |
  |---|---|---|---|---|
  | postgresql | ✓ | ✓ | `statement_timeout` | `enforced` |
  | mysql | ✓ | ✓ | `max_execution_time` | `enforced` |
  | sqlite | ✓ | ✓ | **none** | `partial` |

  Those statements run on **every** connection from every target engine, which is
  the point: `db.target()` — `explore`, `infer_tables`, `extract` — has no
  transaction of its own to guard. `db.target_readonly()` layers Postgres's
  transaction-scoped versions on top, redundantly, because defence that exists in
  one place is one edit from not existing; MySQL and SQLite have no
  transaction-scoped equivalent and their list is empty, which is the model
  working rather than a gap.

  **Registration is never refused over this.** A dialect that cannot promise
  everything says so in `ConnectionTestOut.warnings` and carries its tier on
  `ConnectionOut`. An undisclosed gap and an undisclosed hole are the same bug.

  [tests/test_capabilities.py](tests/test_capabilities.py) asserts every claim
  against the database **in both directions** — a dialect claiming enforcement it
  lacks is a hole; one claiming none while quietly enforcing means users are
  warned for nothing. Not hypothetical: `blocks_ddl=False` for MySQL was written
  from received wisdom and the test caught it, because MySQL 8.4 *does* refuse
  DDL in a read-only transaction. Never `pytest.skip` on the claim — a skip keyed
  on the table is how a wrong table starts agreeing with itself.
  `tests/test_isolation.py::test_a_user_supplied_dsn_still_cannot_write`
  registers credentials that genuinely can write, and is what fails if
  `dialects.install` goes.
- **Generated SQL reaches the driver through `exec_driver_sql`, never `text()`.**
  `text()` reads `:name` as a bind parameter, and this string is whatever the
  model wrote — `WHERE status = ':pending'` becomes a missing-parameter error
  instead of the SQL error the fix node knows how to react to. Postgres `::`
  casts happen to survive `text()`'s regex, which makes the failure rare enough
  to ship and confusing enough to lose a day to.
- **Every message shown to the model or the user is unwrapped through `.orig`.**
  SQLAlchemy wraps driver errors, so `str()` prepends `(psycopg.errors.X)`,
  appends `[SQL: <the whole query>]` and a docs link. `fix` already re-sends the
  SQL, and a `sqlalche.me` URL is the last thing a user should read when their
  question failed.
- **Target engines run `isolation_level="AUTOCOMMIT"`.** Not a style choice:
  `explore` holds one connection across up to 24 tool calls with model round
  trips between them, and a SQLAlchemy connection begins a transaction implicitly
  on its first statement. Without it a T1 turn holds a snapshot open on a
  customer's production database for minutes. Nothing fails; their DBA notices.
  `target_readonly` opts back in, to `dialect.default_isolation_level` rather
  than a constant — SQLite rejects `READ COMMITTED` outright.
- **Never GROUP BY or DISTINCT on a `CAST`.** On MySQL the cast drops the
  column's collation for the connection's, which is case-insensitive by default:
  `GROUP BY CAST(region AS CHAR)` returns `west: 12` where `GROUP BY region`
  returns `west: 4, West: 4, WEST: 4`. That is trap 2 silently disappearing on
  another engine. Group on the column and stringify in Python.
- **A warehouse's password never leaves the server.** `ConnectionOut` has no
  `password` field at all — not `None`, not masked — because a field that does
  not exist cannot be leaked by a future `**row`. URLs are built with
  `sqlalchemy.URL.create`, never an f-string, and `safe_dsn()` renders one built
  *without* a password rather than relying on `hide_password=True`: masking is
  the weaker promise, one flipped keyword from leaking. Connection errors are
  reported by exception type rather than message — and unwrapped through `.orig`
  first, because SQLAlchemy's own type name is `OperationalError` for everything
  and says nothing.
- **A connection's `driver` cannot be changed.** `PATCH` refuses one with a 409.
  A cached recipe is SQL in a dialect, and `schema_fp` would not catch a
  repointing — it hashes types and nullability, and both survive the move. To
  move a connection to another engine, delete and re-register: that cascades the
  cache, which is the correct outcome, because none of it transfers.
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

**`demo/demo.sql` stays PostgreSQL, and that is a decision rather than a
backlog item.** It is 573 lines of `generate_series`, `::interval`,
`ARRAY[...][i]`, `DO $$` role creation and `ALTER DEFAULT PRIVILEGES`, producing
eight numbers that six test files assert on; SQLite has no role system at all, so
parts of it are unrepresentable rather than merely different, and a translation
that quietly shifted a count would be found on stage. The dialect tests use a
much smaller portable fixture instead —
[tests/fixtures/portable.py](tests/fixtures/portable.py), built from a
SQLAlchemy `MetaData` so it is portable by construction.

Note `demo.sql` must stay free of psql meta-commands (`\echo`, `\i`) —
[tests/conftest.py](tests/conftest.py) applies it through psycopg, which can't
parse them.

## Build status

Phases 0–5 of PLAN.md §8 are done: schema, seed, cache tables, cold path,
extraction, and the `plan` node. Not yet built: §5 compaction and schema-drift
invalidation on load (`schema_fp` is written but not checked), the `/admin/cache`
API of §6.2, and the browser UI of Phase 8.

Tracing (Observability above) is built and off by default. Not wired up: cost in
currency — Langfuse prices a generation from its model name, and `claude-opus-5`
is not in its table, so a custom model definition is what would turn the token
counts into money. Nothing reads a trace back: scores, evals and datasets are all
available and all unused.

## Conventions

- `uv` for everything (`uv run …`); dependencies in [pyproject.toml](pyproject.toml).
- `langfuse` is import-legal in [app/tracing.py](app/tracing.py) and nowhere
  else, `sqlalchemy`-style. Everything else takes a context manager from there.
- `sqlalchemy` is import-legal in `app/db.py`, `app/tools.py`, `app/dialects.py`,
  `app/api.py` and `app/store.py`'s four target functions, and nowhere else — in
  particular not in `app/graph.py` beyond what `db.` hands back, and never in
  `sql_agent_cli/`, which [tests/test_cli_isolation.py](tests/test_cli_isolation.py)
  asserts.
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
