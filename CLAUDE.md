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
make optim-probe                       # do the prompts still honour their invariants?
make optim-apply                       # write a gated winner into config/prompts/
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

**On NixOS, a `uv run` that bypasses `make` needs `LD_LIBRARY_PATH` to include
`$NIX_LD_LIBRARY_PATH`** — the two lines above do it for every recipe, so `make
test` is always fine, and it is the direct invocations that are not: the
single-test command above, and `uv run sql-agent`. The symptom is `ValueError:
the greenlet library is required`, which is a lie — greenlet is installed. Its
compiled extension links `libstdc++.so.6`, and the real error is an `ImportError`
one frame down that nothing prints. nix-ld *does* provide that library, but it
works by supplying an `ld.so` at the FHS interpreter path, which fixes foreign
**executables**; greenlet's `.so` is `dlopen`'d by an already-running
interpreter, and that lookup reads the process's own `LD_LIBRARY_PATH`. A
gitignored `.envrc` covers the gap for anyone with direnv. The Makefile's version
stays regardless: it is committed and needs nothing installed, which is the
weaker dependency of the two.

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

## Two config sources, and which is which

**`settings()` is the environment. `config()` is a tracked file.** Two objects
rather than one, so a call site says which it reads.

| | `app/settings.py` | `app/config.py` |
|---|---|---|
| from | env + `.env` | `config/config.yaml` + `config.local.yaml` |
| holds | the two database URLs, `API_TOKEN`, `CONNECTION_SECRET`, the two model API keys, the Langfuse keys, `CONFIG_DIR` | which model, per-node effort, `max_tool_calls`, `max_fix_attempts`, `statement_timeout_ms`, `max_rows` |
| tracked | no (`.env` is gitignored) | yes — the overlay is not |
| unknown key | `extra="ignore"` — dropped in silence | `extra="forbid"` — an error |

The split is "secret or address" against "behaviour". A tracked file must never
hold a key, which is why `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` stayed in the
environment while the model *name* moved. And `CONFIG_DIR` is an env var because
where the config lives cannot itself be config.

**The environment variables that moved are gone, not demoted.** `PROVIDER`,
`MODEL`, `OPENAI_*`, `EFFORT_*`, `MAX_TOOL_CALLS`, `MAX_FIX_ATTEMPTS`,
`STATEMENT_TIMEOUT_MS`, `MAX_ROWS` and `PROMPT_DIR` are not read. Folding the
yaml in as a low-priority pydantic-settings source was the alternative, and it
would have left a stale `EFFORT_PLAN=low` in somebody's `.env` silently beating
the file they were editing. `RETIRED` in [app/main.py](app/main.py) warns about
every one of them at startup, for the same reason the old `STATEMENT_TIMEOUT`
check existed: `extra="ignore"` makes a retired name and a typo identical.

`config/config.yaml` is tracked and holds the **demo** defaults — Anthropic,
`claude-opus-5`, the effort values PLAN.md §7.1 names — so a fresh clone runs.
`config/config.local.yaml` is gitignored and deep-merged over it, which is where
a machine-specific endpoint goes. Deep, not shallow: a local file naming only
`model.model` keeps the tracked file's `provider`.

Both are memoised per process, like the prompts and for a stronger reason — a
model that changed between two turns would invalidate `plan`'s cache prefix with
no symptom but the token counter going up. The lifespan reads both **first**, so
a bad config fails at boot rather than on the first model call of a demo.

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

- **[app/graph.py](app/graph.py)** — nodes, JSON schemas, edges. The
  structured-output schemas live here as module constants; the *prose* is
  `config/prompts/`, because a schema is the node's wire contract and a prompt
  is the thing an optimiser rewrites.
- **[config/prompts/](config/prompts/)** — six markdown files, and **the whole
  file is the prompt**: what is sent is the text, stripped, with no frontmatter
  and no separator. Filenames are the `node=` labels `llm.complete` uses as
  Langfuse generation names, so a harvested trace, the graph node, the config
  block and the file all name the same thing. `README.md` is the one non-prompt
  file the loader allows, and is where the comments that used to sit above the
  constants went. See [config/prompts/README.md](config/prompts/README.md).
- **[app/prompts.py](app/prompts.py)** — the loader. `get(name)` reads
  `$CONFIG_DIR/prompts/<node>.md`, resolved **once per process** — memoised
  because `plan`'s system block sits behind an Anthropic cache breakpoint on the
  promise it varies with `connection_id` alone, and a prompt re-read per turn
  could change between two turns of one server's life with no symptom but T2
  quietly costing more. A missing file, an empty file and a stray `extrct.md`
  are all errors, because a prompt that silently is not what you think it is is
  how an optimisation run measures the seed and reports it as an improvement.
  Imports stdlib and `app.settings` only — deliberately not `app.config`, since
  *where* the prompts are is an environment question and only that is needed to
  read a file. `fingerprint()` goes on the turn span so a harvest can tell which
  prose produced which trace.
- **[app/config.py](app/config.py)** — `config/config.yaml`: which model, at
  what effort, within what bounds. A global `model:` block, an optional override
  on any node, and a `gepa:` block for the teacher. `config/config.local.yaml`
  is deep-merged over it and gitignored. `extra="forbid"` at every level, which
  is what makes it safe to put node names at the top level beside settings — a
  typo is an error rather than a key that configures nothing. See below for why
  it is a second object rather than a source inside `Settings`.
- **[app/llm.py](app/llm.py)** — the *only* module that talks to a model. Two
  backends behind one `complete()`: `anthropic` (demo) and `openai_compat`
  (Ollama/vLLM/LM Studio for local dev). Which one a call uses is resolved from
  its `node=`, so a node may override the model. Nodes never see the difference;
  build messages with `assistant_turn(result, node=…)` / `tool_results(…,
  node=…)` because the two wire formats disagree about tool results — and they
  take a node for that reason: an override can change the *backend*, and reading
  a global provider there would build the wrong shape for the explore loop.
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
- **Effort, never thinking-off.** Cost is controlled per node via each node's
  `effort:` in `config/config.yaml`. Disabling thinking on Opus 5 can turn a
  tool call into visible text that never executes, which silently breaks the
  explore loop — so `effort` is a `Literal` with no value that does it, and
  there is no longer a global `OPENAI_REASONING_EFFORT` that overrode every node
  at once. A global override of a per-node setting can only lose the
  distinction.
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
counts into money. Langfuse's own scores, evals and datasets are all still
unused; the one thing that reads a trace back is `tracing.observations()`, below.

### Prompt evaluation and search (`optim/`)

The second flywheel, and the one PLAN.md §10 named as the road not taken: *"the
prompt template never changes — only what's in it. DSPy optimizers like GEPA are
the industrial version."* Built for `extract` only, manual, human-committed.

The idea it is **not** built on is optimising in realtime from telemetry. GEPA
scores by *running* candidates; a trace records what happened under prompt P and
carries no information about P′. Traces feed the mutation step, never the
scoring step. So: telemetry supplies the corpus, scoring is offline single-node
replay, and an authored probe suite is a hard gate outside the objective.

What makes it cheap is an accident of the architecture. Because `llm.complete()`
is the only place a generation is opened, and it records
`input={"system","messages"}` under a `name=` that is the graph node, Langfuse
already holds a per-node dataset of exact inputs and outputs — `extract` is a
function of three recorded strings, so one metric call is one `low`-effort model
call and needs no database at all.

**The harvest reads Langfuse and nothing else**, and that is a correction. It
first asked Postgres which warehouse each recorded call was about, by joining
`turn.trace_id` — and `make reset` empties the `turn` table by design, while the
trace store keeps everything. So a reset turned every earlier recording into
debris: intact, and unattributable. Thirteen of fifteen, the first time. A reset
should reset; what was wrong was depending on a table whose job is to be
emptied. Scope now comes from the turn span's own `input.connection_id`, and
`prompts.fingerprint()` on the same span says which prose produced it. Note the
asymmetry for later: **Langfuse has the inputs, Postgres has the outcomes, and
only Postgres gets reset** — `extract` scores entirely off the recorded call, but
a `plan` metric would need outcomes and would meet this again.

```
make optim-probe     do the current prompts still honour their invariants?
make optim-harvest   recorded extract calls -> optim/out/extract.jsonl
make optim-run       GEPA over one node's prompt, gated on the probes
make optim-diff      what the winner changed, beside the invariant checklist
make optim-apply     write it into config/prompts/<node>.md (nothing committed)
```

- **`optim/` reads the app; the app never reads `optim/`.** It is a development
  tool in the category of pytest, so it imports `app` directly rather than going
  through the API. The boundary that does hold: **node replay is in-process,
  turn replay is HTTP** — re-driving `stream_turn` by hand is how a harness and
  a product drift. `gepa` is a dependency group, so `uv sync --no-dev` excludes
  it by the same mechanism that excludes pytest, and the two test modules that
  import it `importorskip`.
- **No DSPy**, and not on weight grounds. `app/llm.py` carries Anthropic's
  `output_config` with effort and a JSON-schema format, adaptive thinking, the
  server-side-fallback beta, `cache_control: ephemeral`. Optimising a prompt
  under a different request shape than production uses tunes it for a
  configuration you do not run — and DSPy's own field markers would make harness
  tokens stop being production tokens, in a repo whose claim is a token count.
  GEPA's `reflection_lm` is `llm.complete` at the `gepa:` block's effort (max
  by default), so there is still exactly one module that talks to a model.
- **The probes are the valset and they are not GEPA's valset.** GEPA sees a
  train/val split of harvested cases; the probes run *after* it returns, against
  every candidate in the pool, and any that regresses one the seed passed is
  discarded whatever it scored. Weights cannot express "never" — a
  mean-maximising search trades a rare catastrophic failure for a broad small
  gain whenever the arithmetic allows. Feeding probes in as a valset would also
  score them 0..1 with the metric, and a probe is a predicate.
- **Every cheap metric here is self-defeating if taken alone**, which is why
  `optim/metric_extract.py` has five weighted terms and two gates rather than
  one number. `grounded_in` accepts a token subsequence, so `count(*)` verifies
  against any query that counts — optimising the verified rate destroys the gate
  it is derived from. `tests/test_optim_metric.py` is a list of the degenerate
  prompts someone reasoned their way to; if one starts passing, the metric has a
  hole. An empty extraction scored 0.6 before that file existed, because census,
  names and cost are all vacuously perfect when nothing was recorded.
- **`optim-apply` writes the file and stops.** It stages nothing and commits
  nothing; what it leaves is a working-tree diff. This reversed a policy — there
  used to be no `apply` at all, on the argument that a machine-applied prompt
  arrives without the comment explaining why it says what it says. That argument
  was about `app/prompts.py`, where the prose sat inside a Python string literal
  and the only way to promote a candidate was a copy-paste with no diff: the one
  step in the loop nothing could review. Now the prose is a tracked file, so the
  review gate is `git diff config/prompts/`, which is strictly stronger than a
  human retyping it. The comment did not stop being required — it moved to the
  commit message, and `config/prompts/README.md` carries the per-prompt notes
  the file body no longer can. `apply` refuses a target with uncommitted
  changes, so the diff you read is the one it wrote, and refuses a winner with
  no recorded passing gate, so a file hand-dropped into `optim/out/` cannot be
  promoted as though a run had cleared it. And it refuses a winner that scored
  *below the seed*: `_gate` ranks survivors against each other after skipping
  the seed, so GEPA reporting "best program: 0" — the search found nothing
  better — still produced a written winner and an invitation to promote it.
  Observed, not hypothetical: a completed run scored the seed 0.959 and its best
  survivor 0.928. That was survivable while promotion was a copy-paste under
  human eyes; `apply` made it one command, so `optimize` now records
  `seed_score` in the verdict and `apply` compares against it. A probe suite is
  a set of predicates about failures, not a measure of quality — clearing every
  probe does not make a candidate better than what it replaces.
- **`tokens_in` is not comparable during a run** — every candidate is a distinct
  cache prefix, so `cache_system=True` never hits in the harness. The cost term
  scores `tokens_out` only; "extract got cheaper" measured here would be a lie
  about the number the demo sells.
- Not built: `plan` (needs outcome labelling and SQL execution against a
  deterministic warehouse, and has the most destructive degenerate optimum —
  always say sufficient), and whole-turn A/B over HTTP as a final gate. Never
  worth building for `explore` (a metric call is a whole turn) or `answer` (its
  only cheap proxy is length, and the comment above `ANSWER_SYSTEM` records that
  trade already being measured and decided the other way).

## Conventions

- `uv` for everything (`uv run …`); dependencies in [pyproject.toml](pyproject.toml).
- **Prose belongs in `config/prompts/*.md`, settings in `config/config.yaml`,
  secrets and addresses in the environment.** A string literal in `app/` that a
  model reads is in the wrong file — that is what the move was — and a key in a
  tracked file is a leak waiting for a `git push`.
- `langfuse` is import-legal in [app/tracing.py](app/tracing.py) and nowhere
  else, `sqlalchemy`-style. Everything else takes a context manager from there —
  or, now that traces are read back as well as written, `observations()`. That
  rule was documentation until the optimiser gave a second module a reason to
  want its own client; [tests/test_cli_isolation.py](tests/test_cli_isolation.py)
  asserts it, along with "nothing that ships imports `optim`".
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
