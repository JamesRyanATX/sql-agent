# Self-Optimizing SQL Agent — Build Plan

An agent that answers business questions against a database it has never seen.
The first question is expensive: it explores the schema, writes bad SQL, fixes
it, and eventually lands on an answer. Every question after that is cheap,
because it kept what it learned.

The demo is a token counter going down.

---

## 1. The claim

> **T1:** how many customers do we have? → expensive
> **T2:** how many customers do we have? → cheap
> **T3:** how many customers in the west region? → also cheap

T2 is the same question and the same answer, several times cheaper. T3 is a
question never asked before, answered from what earlier turns learned rather
than from a stored result.

**Measured** end to end on `qwen3.6:27b-mlx`, cold cache, phase 5 complete:

| Turn | Explored | Tokens | Time | Answer |
|---|---|---:|---:|---|
| T1 | 3 tool calls | **10,504** | 293s | 1,840 ✓ |
| T2 | **none** | **2,778** | 119s | 1,840 ✓ |
| T3 (west region) | 2 tool calls | 8,423 | — | 460 ✓ |

**3.8× on T2**, and T3 — a question never asked — costs less than T1 because
`plan` names the gap and `explore` goes straight to it. The curve bumps rather
than resetting, which is what §9 step 5 shows.

These are local-model numbers and the shape is unusual: nearly all the remaining
cost is *output*, because a reasoning model's thinking bills as output. Opus 5's
`effort` is a finer instrument than `reasoning_effort`, so the ratio should
differ — but that is a measurement to take, not a result to assume.

Nothing here needs ground truth, a judge model, or a human labeling anything.
The metric is tokens, and the database is the only oracle required.

**On the exact numbers.** Earlier drafts asserted 8,400 → 400 tokens. The
measured ratio is 3.8×, not 20×. The cached turn's cost is almost entirely
*output*, not the cached input. Three things got
it from 1.5× **worse** than T1 to 3.8× better, none of which was adding decoys:

1. **Per-node `effort` actually applied** (it was Anthropic-only, so the local
   backend ran unbounded and spent 14,339 output tokens deciding).
2. **No extraction on a fully cached turn** — if the cache answered it, nothing
   was learned, and re-deriving it wrote near-duplicates at a call per turn.
3. **Reporting totals, not input.** Prompt caching cuts the cached path's cost
   but not its token count, and the counter on stage shows tokens.

The lever is making the cached path cheap, not making T1 expensive.

---

## 2. What's in the cache

Two kinds of thing, and the distinction matters.

**Schema facts** — discovered by introspection. Which tables exist and which
are relevant, column names and types, enum values actually present, join keys,
row counts, and the observation that `customer.deleted_at` is populated on some
rows.

**Recipes** — how to express a business concept in SQL:

| Concept | Recipe |
|---|---|
| active customer | `customer WHERE deleted_at IS NULL` |
| revenue | `SUM(oi.qty * oi.price)`, excluding `orders.status = 'cancelled'` |
| region | normalize casing before grouping |
| last quarter | `orders.created`, not `created_at` |

**Recipes compose.** "Revenue by region last quarter" is three known recipes
assembled — no exploration needed, even though the question is new. That's the
mechanism behind T3, and it's why this isn't memoization.

What accumulates is a semantic layer, derived from exploration rather than
hand-maintained. The same definitions usually live in dbt or a BI tool and are
written by people.

---

## 3. Schema

**As shipped, this is two Postgres servers, not one.** §3.1 lives on `agent-db`,
built from `migrations/*.sql`. §3.2 lives on `demo-db`, built from
`demo/demo.sql`, and the agent reaches it as a role holding `SELECT` and nothing
else. The plan below assumed a single database and a `public` schema shared
between them — the §5 tombstone and drift machinery is unaffected, but "the
agent's own tables are hidden from introspection" is now "they are on another
server", and §3.2's schema is a demo fixture rather than part of the product.

### 3.1 The cache

```sql
CREATE TABLE cache_entry (
  id           bigserial PRIMARY KEY,
  kind         text NOT NULL,        -- schema_fact | recipe
  name         text,                 -- 'revenue', 'active customer'
  claim        text NOT NULL,        -- the prose the model reads
  sql_fragment text,                 -- recipes only
  tables       text[] NOT NULL,      -- for drift invalidation
  origin       text NOT NULL,        -- learned | human
  pinned       boolean NOT NULL DEFAULT false,
  disabled     boolean NOT NULL DEFAULT false,
  tombstone    boolean NOT NULL DEFAULT false,  -- a learned-wrong thing
  verified     boolean NOT NULL DEFAULT false,
  hits         int NOT NULL DEFAULT 0,
  schema_fp    text,                 -- fingerprint of tables[] at write time
  created_turn bigint,
  last_used_turn bigint,             -- §5 compaction needs this; see below
  created_at   timestamptz NOT NULL DEFAULT now(),
  updated_at   timestamptz NOT NULL DEFAULT now()
);
```

```sql
CREATE TABLE turn (
  id            bigserial PRIMARY KEY,
  session_id    uuid NOT NULL,
  question      text NOT NULL,
  sql           text,
  answer        text,
  tool_calls    int NOT NULL DEFAULT 0,
  explored      boolean NOT NULL DEFAULT false,
  tokens_in     int NOT NULL DEFAULT 0,
  tokens_out    int NOT NULL DEFAULT 0,
  latency_ms    int,
  cache_entries int NOT NULL,        -- how many loaded for this turn
  created_at    timestamptz NOT NULL DEFAULT now()
);
```

`turn.tokens_in + tokens_out` against `created_at` is the entire demo chart. Free
to collect.

`last_used_turn` wasn't in the first draft of this table, but §5's compaction
rule — *drop unverified entries with `hits = 0` and no use in 20 turns* — is
unanswerable from `created_turn` alone. `bump_hits()` maintains it.

Two invariants the store layer enforces rather than the schema:

- **Named entries upsert.** Re-learning `revenue` refines one row instead of
  accumulating near-duplicates for compaction to clean up. Unnamed schema facts
  can duplicate; that's what compaction is for.
- **A human's `pinned` entry is never overwritten by an extraction.** §6.2's
  whole argument is that correcting a recipe fixes every future question that
  composes it, so an extraction reverting that on the next turn would undo the
  correction silently.

### 3.2 The business schema

Shipped as [demo/demo.sql](demo/demo.sql) — role, schema and deterministic data
in one file, applied by `make seed`. It is a fixture, not the product, which is
why it sits under `demo/` beside the tape that records it.

Two properties matter.

**Wide, not deep.** ~40 tables, of which 4 are relevant. Exploration has to
genuinely *search*, or T1 is too cheap for the difference to be visible. It is
also closer to a real warehouse than four tables would be.

**Booby-trapped**, so exploration produces recipes worth reading:

```
customer(id, name, region, signed_up, deleted_at)   -- soft deletes, ~8%
product(id, name, unit_price, discontinued)
orders(id, customer_id, created, status)            -- NOT created_at
order_item(order_id, product_id, qty, price)        -- historical unit price
+ ~36 plausible decoys (audit_log, sessions, feature_flags, ...)
```

Five traps: soft deletes, `region` casing, `status = 'cancelled'`, historical
`order_item.price` vs `product.unit_price`, and `created` vs `created_at`.

Trap 1 fires on the most obvious question anyone can ask, so T1 has to actually
investigate.

Seed targets: 2,000 customers, 160 soft-deleted → **1,840 active**. That number
is the answer to step 1 of the demo, so it's fixed and deterministic.

---

## 4. The graph

```
load_cache → plan → ┬─ (cache sufficient) ─────────────────────────┐
                    │                                              │
                    └─ explore ⇄ introspect (loop) ─▶ generate_sql ─┤
                                                                    ▼
                                                                execute
                                                                    │
                                       ┌── error ───────────────────┤
                                       ▼                            │ ok
                                 fix (≤3) ──▶ execute               ▼
                                                                extract
                                                                    │
                                                                    ▼
                                                                 answer
```

**`load_cache`** — everything not `disabled`, ordered by hits. All of it, every
turn. It fits in context; retrieval would only add a way to miss the entry you
needed.

**`plan`** — one cheap call. Can this question be answered from what's cached?
This is the branch that makes T2 cheap.

> **Amendment (was two nodes).** Earlier drafts routed
> `plan → generate_sql` as two separate calls on the cached path, each
> re-sending the system prompt and the entire cache. But if the cache is
> sufficient to *answer*, it is sufficient to write the SQL. So `plan` emits a
> single structured response: `{sufficient: true, sql: "..."}` and the graph
> edges straight to `execute`; or `{sufficient: false, missing: [...]}` and
> exploration runs, narrowed by `missing`. This roughly halves T2, which is
> what §9 step 3 shows.

**`explore`** — a ReAct loop over read-only introspection tools:
`list_tables`, `describe_table`, `sample_column`, `count_distinct`. Bounded —
cap tool calls, or T1 runs forever on stage.

Sampling is what discovers the traps. `sample_column('customer','deleted_at')`
returns non-nulls, and that's how the agent infers soft deletes rather than
being told.

**`execute`** — read-only, timeout:

```sql
SET LOCAL transaction_read_only = on;
SET LOCAL statement_timeout = '5s';
```

**`fix`** — error text plus SQL back to the model, ≤3 attempts.

**`extract`** — after a successful query, write schema facts and recipes. Gate
this: a recipe is only `verified = true` if exploration actually confirmed the
assumption (it sampled `status` and saw `'cancelled'`) or if the recipe has been
reused successfully. **A wrong recipe compounds** — a bad `revenue` silently
corrupts every revenue answer forever, which is far worse than a bad schema
fact.

**`answer`** — prose, the SQL, and the assumptions stated inline:

> 1,840 customers (excluding soft-deleted).

---

## 5. Cache hygiene

**Compaction.** Cap at ~40 entries. Over the cap, one LLM call merges the set
down. Rules: merge entries about the same table or concept; drop unverified
entries with `hits = 0` and no use in 20 turns; **never touch `pinned` rows.** A
compaction pass that removes a human's correction from last week would undo it
silently, with nothing in the output to show it happened.

**Schema drift.** `schema_fp` is a hash of `information_schema` for the entry's
`tables[]`. On load, recompute; on mismatch, mark the entry stale and force
re-exploration for anything touching it. Ten lines, and it's the detail that
makes this look production-grade rather than a demo toy.

**Tombstones.** Deleting a wrong learned recipe doesn't work — exploration
rediscovers the same wrong thing next session. A delete leaves a visible
negative constraint: "revenue does not include cancelled orders."

---

## 6. API

### 6.1 Chat

One endpoint the user touches. Streaming, because the point is watching T1's
exploration scroll past and then *not* happen on T2.

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

def sse(ev: dict) -> str:
    return f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"

@app.post("/ask")
async def ask(req: Request, body: AskBody):     # {session_id, question}
    async def gen():
        async for mode, chunk in graph.astream(
            {"session_id": body.session_id, "question": body.question},
            stream_mode=["updates", "custom"],
            config={"configurable": {"thread_id": body.session_id}},
        ):
            if await req.is_disconnected():
                break            # else the graph keeps burning tokens
            for ev in to_events(chunk):
                yield sse(ev)
        yield sse({"type": "done"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Events: `plan`, `explore`, `sql`, `error`, `fix`, `learned`, `answer`, `usage`,
`done`. The `usage` event carries running token count — that's what the counter
in the UI binds to.

Two things that bite: `X-Accel-Buffering: no`, or nginx holds the stream until
completion and the live demo looks frozen; and the disconnect check, or a closed
tab leaves the graph running.

**As shipped**, this is `POST /v1/ask` behind a bearer token, and it is the only
way to run a turn — `scripts/*` are HTTP clients of it rather than a second
implementation, so the code path the demo exercises is the one a user gets. The
cache surface below landed alongside it as `GET`/`DELETE /v1/cache`; the rest of
§6.2 is still outstanding.

### 6.2 Admin

The cache is the product, so it gets a real surface.

```
GET    /admin/cache            list, filter by kind, with provenance + hits
POST   /admin/cache            hand-write an entry (origin=human, pinned)
PATCH  /admin/cache/{id}       edit claim or sql_fragment → origin=human, pinned
DELETE /admin/cache/{id}       tombstone, not row removal
POST   /admin/cache/compact    run compaction on demand
```

Three things this buys beyond the demo:

- **Correcting a recipe fixes every future question** that composes it. Fixing
  an answer in chat fixes one. This is the right layer for the human.
- **`POST` is not just a test hook.** Seed the cache from existing dbt
  definitions and skip the expensive exploration entirely.
- **Blast radius.** Show `hits` and, for recipes, what composes them — editing
  `revenue` changes every revenue answer, and whoever hits save should see that
  first.

Gate it behind auth. It's a privileged surface even in a demo.

Plus `GET /stats` — tokens per turn — for the chart.

---

## 7. Stack

- **LangGraph** + `AsyncPostgresSaver`, one Postgres for cache, turns, checkpoints
- **The Anthropic SDK called directly from inside the nodes** — no
  `langchain-anthropic`. LangGraph owns the graph, streaming, and checkpointing;
  each node calls `anthropic.AsyncAnthropic` itself. The demo metric is a token
  count on screen, and reading `response.usage` directly gives exact
  `input_tokens` / `output_tokens` / `cache_read_input_tokens` attributable per
  node with no adapter in between. It also keeps us on the current model request
  surface without depending on a wrapper to forward those fields.
- **A second backend for local development.** `provider = openai_compat` points
  at any OpenAI-shaped endpoint (Ollama, vLLM, LM Studio). Nodes never see the
  difference — messages are built with `llm.assistant_turn()` and
  `llm.tool_results()` because the two wire formats disagree about how a tool
  result is represented. See §7.2.
- `psycopg` (v3, async) throughout
- `claude-opus-5`
- `uv` for dependencies, Docker for Postgres
- `make reset` — truncate `cache_entry`, `turn`, and the checkpoint tables, then
  reseed — so you can rehearse and recover on stage

### 7.1 Model request shape

Every node depends on this, and it's easy to get wrong:

- **Thinking is on by default** on Opus 5. Omitting `thinking` runs adaptive.
  `max_tokens` caps thinking *plus* response text together — size it with
  headroom, or answers truncate mid-sentence.
- **Don't disable thinking to save tokens.** Use `output_config: {"effort": "low"}`
  instead. With thinking disabled, Opus 5 can write a tool call into its visible
  text instead of emitting a `tool_use` block — the call silently never runs,
  which is fatal inside the explore loop — and can leak `<thinking>` tags into
  output.
- **Effort per node:** `low` for the cached plan path, `high` for exploration and
  SQL generation, `low` for extraction.
- **Rejected with a 400:** `temperature`, `top_p`, `top_k`, `budget_tokens`, and
  last-assistant-turn prefills.
- **Structured output** via `output_config: {"format": {"type": "json_schema", …}}`
  for the plan decision and for extraction. Replaces prefill, removes parsing.
- **Check `stop_reason == "refusal"` before reading `content`** — a refusal is an
  HTTP 200 with empty or partial content. Enable `fallbacks: "default"`.
- **Prompt caching:** `cache_control` breakpoint on the system + cache prefix.
  Minimum cacheable prefix on Opus 5 is 512 tokens, which the cache block clears
  once it holds ~10 entries.

### 7.2 The local backend, and what it can't do

For development when the Anthropic key isn't available:

```
PROVIDER=openai_compat
OPENAI_BASE_URL=http://<host>:11434/v1
OPENAI_MODEL=qwen3.6:27b-mlx
```

**Structured output goes through a forced tool call, not `response_format`.**
Measured against Ollama 0.30.8 serving `qwen3.6:27b-mlx`, three routes were
tried:

| Route | Result |
|---|---|
| OpenAI-compat `response_format: json_schema` | **ignored** — returns prose in a markdown fence |
| Native `/api/chat` with `format: {schema}` | **ignored** — same |
| Forced tool call whose parameters *are* the schema | **works**, and is faster (no prose to generate) |

Both ignored routes need grammar-constrained decoding, which the `-mlx` runner
doesn't implement — MLX is a separate inference engine from llama.cpp. A GGUF
build would restore them, but there's no reason to bother: the tool channel is
JSON by construction, and the explore loop already needs tool calling, so this
is one code path rather than two.

**The local backend is the primary development loop.** Vet behaviour here, then
spend Opus 5 tokens on the measurements that actually need the demo model.

**Local latency is about hardware contention, not model capability.** The same
first-`explore` request shape measured 600s+ (timed out) under contention and
**22s** on a free machine. Under contention it looks like the model is
incapable; it isn't. If a turn crawls, check what else is on the GPU before
changing anything in the prompt or the graph — an earlier revision of this file
concluded the model "can't reach step 5" and prescribed a prompt fix, on
evidence that was entirely a busy machine.

Generation controls, measured on the same request (free machine):

| Setting | Time | Completion tokens |
|---|---|---|
| `max_tokens: 16000` (baseline) | 22s | 154 |
| `max_tokens: 2000` | 9s | 164 |
| `reasoning_effort: "low"` | 6s | 108 |
| `enable_thinking: false` | 5s | 84 |

All four produced a valid tool call, so the caps are available if a turn ever
needs bounding — `OPENAI_MAX_TOKENS` and `OPENAI_REASONING_EFFORT` — but none
is needed by default, and **`OPENAI_MAX_TOKENS` should not be set low.**

That last point cost a phase. A 4,000-token cap was added here to bound "slow
generation" — the problem that turned out to be GPU contention — and three
phases later it silently broke extraction on any turn whose prompt had grown:
the model spent the budget thinking and never reached its tool call, so the
response came back empty with no server error. The symptom (`got ''`) named
neither the cause nor the node. Two consequences worth keeping:

- A limit imposed on a wrong diagnosis does not stay where you put it. It waits.
- **A structured call that returns nothing must report `finish_reason`.**
  `'length'` is the entire diagnosis and costs one field to surface.

**What still needs Opus 5:** the T1 gate number. "Is T1 over ~3,000 tokens?" is
a question about the demo model — a different model tokenizes and explores
differently, so a local figure doesn't transfer. Everything upstream of that
number can be vetted locally first.

---

## 8. Build phases

Each phase ends in a state you can demo or test, and can be picked up cold.
Phases 1–2 are the yak-shaving; 3 and 5 hold the risk.

### Phase 0 — Foundation

Repo skeleton and a running loop, nothing intelligent.

- `pyproject.toml` via `uv`: `langgraph`, `langgraph-checkpoint-postgres`,
  `anthropic`, `fastapi`, `uvicorn`, `psycopg[binary,pool]`, `pytest`,
  `pytest-asyncio`
- `docker-compose.yml` — `postgres:15-alpine`
- `Makefile` — `up`, `down`, `dev`, `migrate`, `seed`, `reset`, `test`
- `app/settings.py` — `DATABASE_URL`, `ANTHROPIC_API_KEY`, model + per-node effort
- `app/main.py` — FastAPI, `GET /health`, the §6.1 SSE scaffold including both
  details that bite
- Spike a two-node LangGraph with `AsyncPostgresSaver` (`await saver.setup()`)
  and stream one `custom` event out over SSE

**Exit:** `make up`, then `curl -N localhost:8000/v1/ask` streams events
from a trivial graph and terminates cleanly on client disconnect. ✅ Done.
(The endpoint was `/ask` and `make dev` ran a host-side uvicorn until the API
was containerized; `make up` now starts it with reload.)

### Phase 1 — Business schema, seed, and the five traps

Half the work, all the yak-shaving.

- the four real tables per §3.2 plus ~36 decoys
  (`audit_log`, `sessions`, `feature_flags`, `webhook_delivery`, `email_queue`, …)
  with plausible columns and non-trivial row counts, so exploration has to search
- the seed, deterministic and idempotent:
  - 2,000 customers, 160 with `deleted_at` → **1,840 active**
  - `region` deliberately mixed-case (`west` / `West` / `WEST`)
  - `orders.created` (never `created_at`), `status` including `'cancelled'`
  - `order_item.price` diverging from `product.unit_price` on older orders
  (both shipped as `demo/demo.sql`; this was a migration plus a Python script
  until the databases split)
- `tests/test_traps.py` — one test per trap, asserting the naive query returns a
  *different* answer from the correct one

**Exit:** `make seed` is idempotent; `pytest tests/test_traps.py` passes with all
five traps demonstrably biting. ✅ Done — 40 business tables, and each trap
measured: soft deletes hide 160 of 2,000; `region = 'west'` finds 350 of 500;
cancelled orders inflate revenue 10.3%; `product.unit_price` inflates it 11.0%.

**Found here, for Phase 3:** `public` also holds the agent's own six tables
(`cache_entry`, `turn`, and LangGraph's four checkpoint tables). `list_tables`
must exclude them, or the agent explores its own memory, caches facts about the
cache, and step 2 fills up with noise.

### Phase 2 — Cache and turn tables

- `migrations/002_cache.sql` — §3.1 exactly, plus indexes on
  `(disabled, hits desc)` and `turn(session_id, created_at)`
- `app/store.py` — `load_cache()`, `write_entries()`, `bump_hits()`,
  `start_turn()` / `finish_turn()`, `schema_fingerprint(tables)` (hash over
  `information_schema.columns`)

The turn write is split in two rather than the single `record_turn()` first
sketched: entries written by `extract` mid-turn need a `created_turn` to point
at, and the turn's cost isn't known until it ends.

**Exit:** round-trip unit tests; fingerprint provably changes when a column is
renamed and is stable otherwise. ✅ Done — 14 tests.

### Phase 3 — The cold path

No cache yet: `explore ⇄ introspect → generate_sql → execute → fix → answer`.

- Read-only introspection tools, parameter-bound, never string-interpolated
- ReAct loop with a hard tool-call cap
- `execute` in a transaction with `transaction_read_only` and a 5s statement timeout
- `fix` node, ≤3 attempts
- Per-node token accounting accumulated into the `turn` row
- SSE events: `explore`, `sql`, `error`, `fix`, `answer`, `usage`, `done`

**Exit:** "how many customers do we have?" → **1,840 (excluding soft-deleted)**,
after visible exploration and at least one bad query it fixes itself.
**Gate:** record T1. If it's under ~3,000 tokens, the delta won't sell.

**Making T1 expensive — the decoy lever is probably the wrong one.** §7 Step 2
assumed more decoys would do it. The local run says otherwise: the model listed
40 tables once, saw one called `customer`, and went straight there. A table
whose name matches the question isn't a search, and a 41st decoy doesn't change
that.

What actually cost tokens was *reasoning and verification* — describing the
table, sampling the column, working out what the null pattern meant. 7,761
tokens on three tool calls, a third of it output. So T1 clears the gate
comfortably, just not for the reason the plan assumed. Two implications:

- Don't pad the schema further. It's already wide enough to be realistic, and
  wider won't buy expense.
- The questions that are genuinely expensive are the ones spanning several
  tables with several traps between them — step 5's "revenue by region last
  quarter" needs `customer`, `orders`, `order_item` and `product`, plus the
  casing, cancelled and historical-price traps. If T1 ever looks too cheap on
  Opus 5, the lever is the question, not the schema.

✅ **Behaviour verified locally; only the gate number is outstanding.** All five
nodes, the tools, and the wiring are done and covered by 40 tests that script
the model rather than call it — those prove the graph, not the prompts. The
prompts were then vetted against a live model (§7.2), which is what found the
two defects below.

Measured on `qwen3.6:27b-mlx`, cold cache every time:

| Question | Calls | Tokens | Time | Correct |
|---|---|---|---|---|
| how many customers do we have? | 3 | 7,761 | 316s | 1,840 ✓ |
| how many customers do we have? (rerun) | 2 | 7,175 | 293s | 1,840 ✓ |
| revenue last quarter, by region? | 9 + 1 fix | 20,149 | 667s | ✓ (see caveat) |

Token cost is stable across runs (within 8%) even when the number of tool calls
differs — encouraging for a reproducible T1 on stage.

**Exploration depth is not deterministic.** The two T1 runs reached the same
answer by different routes: the first sampled `deleted_at`, the second inferred
soft deletes from the column being nullable and never sampled at all. Step 1 of
§9 promises the audience a visible sample. Worth revisiting at phase 5 — the
lever is the prompt, and it should be tested, not assumed.

⚠️ **The local model drifts, and step 3 is where that shows.** Asked *"how many
customers do we have?"* with a two-entry cache that mentioned nothing about
regions, `qwen3.6:27b-mlx` called `count_distinct(region)` unprompted and
answered **460** — the west-region figure — instead of 1,840. Nothing in the
prompt pointed there.

This matters more than a single wrong answer: §1's claim is that T2 reproduces
T1's answer more cheaply. A model that silently answers a different question
on the second ask breaks the demo whatever the token count says. Verify T2
reproduces T1 exactly on Opus 5 before trusting the step — and if it doesn't,
that is a prompt problem to solve in phase 5, not a number to report.

Two separate things are still open, and only the second is a blocker:

- **The local run — done, and the core premise holds.** On `qwen3.6:27b-mlx`:
  3 tool calls (`list_tables` → `describe_table(customer)` →
  `sample_column(deleted_at)`), 7,761 tokens, 316s, answer **1,840**. It
  inferred the soft-delete convention from the sample rather than being told,
  and said so unprompted: *"A newcomer might miss this convention and count all
  2,000 rows instead of the correct 1,840."* That sentence is the recipe phase 4
  has to extract, produced without asking for it.

  Caveat worth carrying forward: it reached `customer` in three calls, because
  a table called `customer` in an alphabetical list is not a search. §3.2 wants
  T1 to require real investigation — see "Making T1 expensive" below.
- **The gate itself needs Opus 5.** "Is T1 over ~3,000 tokens?" is a question
  about the demo model — a different model tokenizes differently, explores
  differently, and gives a number that doesn't transfer. The
  `ANTHROPIC_API_KEY` in the environment returns `401 authentication_error`
  against `api.anthropic.com` (confirmed with plain curl, so it's the
  credential, not the client). With a valid key, `make t1` runs the real
  measurement.

### Phase 4 — Extraction and `load_cache`

- `extract` node, structured output, writing both kinds from §2
- Verification gating per §4 — a wrong recipe compounds
- `load_cache` prepends everything not `disabled`, ordered by `hits`
- Increment `hits` on entries the turn actually used

**Gate the extraction against the SQL, not against the prose.** Observed on the
phase-3 step-5 run: the model's SQL said

```sql
WHERE o.status NOT IN ('cancelled', 'refunded')
```

while its stated assumption and its final answer both claimed pending orders
were excluded too. They weren't — that's £206,598 reported where the stated
rule gives £178,199, a **16% overstatement of its own caveat**. The number was
right for the query it ran; the sentence describing what was counted was false.

This is the §10 failure mode with a concrete instance: an `extract` node that
reads the answer prose would cache *"revenue excludes pending orders"* as a
recipe, and every future revenue question would compose a claim the SQL never
implemented. Two consequences for this phase:

- Derive recipes from the executed SQL, and treat the prose only as a label.
- A recipe is `verified` only when the fragment it stores actually appears in
  SQL that ran successfully. "The model said so" is not confirmation.

**Match on tokens, not substrings.** The first gate did exact substring
containment and was too brittle to be useful: a recipe of `COUNT(*) FROM
customer WHERE deleted_at IS NULL` failed against `SELECT COUNT(*) AS
active_customer_count FROM customer WHERE deleted_at IS NULL`, because the alias
sits between two adjacent fragment tokens. A gate that rejects correct recipes
is no more use than one that accepts wrong ones. `grounded_in()` now does an
order-preserving token subsequence — tolerant of aliases and formatting, still
rejecting a fragment that names anything the query never mentioned.

**Backfill `tables[]`.** The model returns it empty often enough to matter, and
an entry with no tables can never go stale however far the schema moves under
it — `schema_fp` would hash nothing, so phase 6's drift detection silently never
fires on it. `infer_tables()` scans the executed SQL against
`information_schema` when the model omits them.

**A failed extraction must not fail the turn.** Observed live: the model
answered *1,840* correctly, the extract call came back empty, and the turn was
recorded as `failed` — a user would have seen an error while a correct answer
sat one node away. Extraction is a bonus paid for by a question that has already
been answered. It now emits `learned {count: 0, failed: …}` and continues.

**Show `extract` the names already filed — and both failure modes around it.**
Entries upsert on `name`, so naming can fail in two opposite directions:

- *Too little reuse → duplicates.* Left to itself the model paraphrases: a live
  run filed `active customer count` on one turn and `active customers count` on
  the next, which the upsert cannot merge. By step 3 the cache reads as
  near-duplicates rather than the six clean entries §9 promises.
- *Too much reuse → clobbering, which is worse.* After adding a "reuse the
  name" instruction, a west-region turn reused `number of customers` and
  **replaced** the general recipe with a region-filtered one. Duplicates are
  noise; this silently destroys a correct general rule, and every later
  question that composes it inherits the special case.

The prompt now shows each filed name *with its claim*, states plainly that
reuse overwrites, and says to reuse only for the same concept — a narrower or
merely related finding gets its own name.

**Stop tuning the prompt here; compaction owns this.** Three iterations of the
naming instruction produced three different failures — no guidance gave
duplicates, "reuse names" gave clobbering, "reuse only for the same concept"
gave duplicates again. The model cannot reliably judge "same concept" at write
time with only names and claims in front of it, and a live cache after three
turns held six entries of which two pairs said the same thing:

```
active customer definition   ─┐ same claim
active customer              ─┘
customer.region              ─┐ same claim
region casing convention     ─┘
```

That is exactly what §5's first compaction rule is for — *merge entries about
the same table or concept*. Duplication is the expected steady state between
extraction and compaction, not a defect to prompt away. Two consequences:

- Keep the current instruction (duplicates are the safe failure; clobbering
  destroys correct rules) and let phase 6 do the merging.
- **Step 2 of §9 depends on phase 6, not just phase 4.** "Six entries appeared,
  read them aloud" does not survive two duplicate pairs. Either compaction runs
  before that step, or the cap comes down so it triggers naturally within the
  demo's handful of turns.

**Exit:** after T1 the cache holds ≥5 entries readable aloud as English; T2 still
explores (no `plan` node yet) but answers correctly.

**Status: mechanism done, exit partly met.** Three turns produced six entries,
all readable English, correctly verified, with tables backfilled — and the cache
demonstrably cut exploration (T1 took 4 tool calls, later turns took 2). Two
things are not met and both are model-judgement problems rather than plumbing:
the duplicate pairs above, and T2 failing to reproduce T1's answer. Neither is
worth further local iteration — see the drift warning under phase 3 and the
compaction note above.

**Extraction roughly doubles a cold turn** — T1 measured 13,308 tokens with it
against 7,175 without, on the same question and model. That cost is paid once
and never recurs, so it widens the phase-5 delta rather than eating into it.

**Test isolation, learned the hard way.** These tests share one database with
whatever else is running. Two rules, both from breaking them: scope
cache/turn assertions to the row's own `created_turn` rather than reading the
whole table, and never `TRUNCATE` shared tables in a test — a suite run during
a live experiment destroyed the entries the experiment was measuring. Anything
that is really a property of a pure function (`render_cache([]) == ""`) belongs
in a test that never touches the database at all.

### Phase 5 — The `plan` node

Where the order of magnitude appears. Everything before this is setup.

- Merged plan+generate structured call (§4 amendment)
- Conditional edge: sufficient → `execute`; insufficient → `explore`, narrowed
  by `missing`
- `cache_control` breakpoint on the system + cache prefix

⚠️ **The cached path can cost *more* than the exploration it replaces.** First
live measurement, `qwen3.6:27b-mlx`:

| Turn | Explored | In | Out | Total |
|---|---|---:|---:|---:|
| T1 | yes | 6,079 | 4,425 | **10,504** |
| T2 | **no** | 1,714 | **14,339** | **16,053** |

T2 did everything right — skipped exploration entirely, reproduced T1's answer
of 1,840 — and cost 50% more. Input collapsed as expected (1,714: the cache is
small). Output exploded, because a reasoning model handed *"decide whether this
is enough"* will happily deliberate for 14k tokens, and thinking is billed as
output.

**So the delta depends on the `plan` node being genuinely cheap, and the only
brake is effort.** `effort_plan = low` existed for exactly this and was being
ignored — `effort` was Anthropic-only on the provider seam, so the local
backend ran unbounded. It now maps onto `reasoning_effort` too.

Three things to carry into the measurement:

- **Report totals, not input.** Prompt caching cuts the *cost* of the cached
  path but not its token count, and the counter on stage shows tokens. A chart
  built on input tokens alone would show a drop that the total does not.
- **The risk is not local-only.** Opus 5 thinks by default and bills thinking as
  output. `effort_plan = low` is the mitigation and it is set — but whether a
  low-effort plan call is actually cheap enough is a measurement, not an
  assumption.
- **If T2 is not clearly cheaper, do not proceed to phase 6.** The gate is the
  point of this phase.

**Exit:**
- T2 (same question) → no exploration, same answer, measured token drop recorded
- T3 ("how many customers in the west region?") → never asked before, answered
  with **no new exploration** — recipes composed
- **Gate:** if the T1/T2 ratio is unconvincing, tune decoys and exploration
  budget *now*, before building anything on top of it

✅ **Done. 10,504 → 2,778 tokens (3.8×), same answer, no exploration; T3
answered a new question for 8,423 — less than T1.** See §1 for the measured
table and the three changes that got there. The gate is met on the local model;
the Opus 5 figure is still to be taken.

**Skip extraction when `plan` was sufficient.** If the cache answered the
question, nothing was learned — re-deriving it writes near-duplicates of entries
sitting right there and bills a call every turn to do it. That single change took
T2 from 5,546 to 2,778. A turn that needed a `fix` still extracts, because
correcting the query *is* something learned.

### Phase 6 — Cache hygiene

Compaction, drift detection, tombstones — §5.

**Exit:** a test renames a column and proves the dependent entry goes stale; a
compaction run over 45 entries preserves every pinned row.

### Phase 7 — Admin API and stats

§6.2, plus `GET /stats`.

**Exit:** steps 6 and 7 of §9 work — hand-edit the `revenue` recipe, re-ask the
revenue question, get the corrected answer.

### Phase 8 — Demo UI

Two panels: chat with a live token counter bound to the `usage` event, and the
cache browser with provenance and hits. Plus the tokens-per-turn chart.

**Exit:** the whole §9 script runs end to end in a browser.

### Phase 9 — Rehearsal hardening

- `make reset` covers cache, turns, *and* checkpoint tables
- Runbook: what to do when a step misfires live
- Rehearse the §9 script against a cold cache at least twice

---

## 9. Demo script

Live, nothing pre-baked. Two panels: chat with a token counter, and the cache.

Open with an empty `cache_entry` table, in front of the room.

| # | Action | What they see |
|---|---|---|
| 1 | how many customers do we have? | Exploration scrolls: 40 tables listed, columns described, a sampled column, one bad query, a fix. **1,840 customers (excluding soft-deleted).** |
| 2 | Show `/admin/cache` | Six entries appeared. Read them aloud — they're English. |
| 3 | Same question again | No exploration. Counter drops by an order of magnitude. Same answer. |
| 4 | how many in the west region? | Never asked. Still cheap, no exploration — recipes composed. |
| 5 | revenue by region last quarter | Small incremental exploration for `order_item`, then answer. Curve bumps, doesn't reset. |
| 6 | Open `/admin/cache`, edit the `revenue` recipe | Fix a subtly wrong definition by hand. |
| 7 | Ask the revenue question again | Corrected — and every future revenue question with it. |
| 8 | The chart | Tokens per turn, descending, with the step at turn 5. |

Step 3 shows the same question costing less. Step 4 shows a question that was
never asked being answered from the cache. Step 6 shows the cache is editable.

---

## 10. Caveats worth saying out loud

- This is context management, not model improvement. The prompt template never
  changes — only what's in it. DSPy optimizers like GEPA are the industrial
  version; they search over the instructions themselves.
- Cheap doesn't mean correct. The agent can be confidently wrong at 400 tokens
  just as easily as at 8,400. `/admin/cache` exists because of this.
- A wrong recipe compounds across every question that composes it. Verification
  gating and the admin surface are the defense; provenance is the audit trail.
- Schema drift silently invalidates recipes. Fingerprinting catches renames; it
  won't catch semantic changes to a column's meaning.
- Cache-in-prompt doesn't survive a thousand-table warehouse. At that size you
  need retrieval over the cache, and every question this demo dodges comes back.
