# sql-agent

A self-optimizing SQL agent that gets cheaper with every question.

![The agent answering three questions against a database it has never seen](demo/demo.gif)

## Overview

End to end on `claude-opus-5`, from a cold cache:

| Turn | Explored | Tokens | Time | Answer |
|---|---|---:|---:|---|
| T1 — how many customers do we have? | 5 tool calls | **11,505** | 34s | 1,840 ✓ |
| T2 — the same question again | **none** | **371** | 8s | 1,840 ✓ |
| T3 — how many in the west region? | **none** | 475 | 6s | 460 ✓ |

**T1** primes the agent's memory.

**T2** is the same question and the same answer, but cheaper now. 

**T3** had never been asked, but costs about the same as T2.

## Quickstart

```bash
make up && make migrate && make seed   # two databases + API, then 2,000 customers
make demo                              # generate demo video
make cache                             # what it learned, as the model sees it
make turns                             # tokens per turn
```

Copy `.env.example` to `.env` and pick a provider: `PROVIDER=anthropic` for the
demo model, or `PROVIDER=openai_compat` with `OPENAI_BASE_URL` / `OPENAI_MODEL`
for anything OpenAI-shaped (Ollama, vLLM, LM Studio).

## Architecture

### LangGraph Workflow

```mermaid
flowchart TD
    START([start]) --> load_cache
    load_cache --> plan
    plan -- cache sufficient --> execute
    plan -- cache insufficient --> explore
    explore --> generate_sql
    generate_sql --> execute
    execute -- error, attempts left --> fix
    fix --> execute
    execute -- error, attempts exhausted --> answer
    execute -- ok --> extract
    extract --> answer
    answer --> END([end])
```

### API

The agent runs behind one server, and `make t1` / `make cache` / `make reset`
are terminal renderers for these endpoints — there is no second code path.

```
POST   /v1/ask      one turn, streamed as SSE
GET    /v1/cache    what the agent has learned, exactly as the model reads it
DELETE /v1/cache    forget all of it — cache, turn log, checkpoints
GET    /health      unversioned and open, for a load balancer
```

Everything under `/v1` takes `Authorization: Bearer $API_TOKEN`. Leaving
`API_TOKEN` unset leaves it open, which the server warns about at startup.

`make up` starts both databases and the API together, with `app/` mounted and
`--reload` — an edit restarts the server in place, and only a dependency change
needs `make build`.

### Two databases

The agent's memory and the data it queries are on **separate Postgres servers**,
and it reaches the second through a role holding `SELECT` and nothing else.

| | `agent-db` :5433 | `demo-db` :5432 |
|---|---|---|
| holds | what the agent has learned | `customer`, `orders`, 38 decoys |
| built by | `migrations/*.sql` | `demo/demo.sql` |
| agent connects as | owner | `reader` — **SELECT only** |

So the agent can't explore its own cache and cache facts about caching, and
generated SQL can't write to the business data. Neither of those rests on
application code remembering to check: `make reset` wipes the agent's database
because that connection cannot see any other, and a `DELETE FROM customer` fails
on privileges before the read-only transaction guard is even consulted.

`demo/demo.sql` is not the product — it's a booby-trapped fixture so the agent
has something to explore, which is why it sits next to the tape that records it.
Point `TARGET_DATABASE_URL` at a real warehouse and nothing else changes.

To re-record the demo above:

```bash
make demo          # live take
make demo-verify   # check the take from the turn table
```

`make demo` needs [VHS](https://github.com/charmbracelet/vhs).

## Design

See [PLAN.md](PLAN.md) for architecture details and the build plan.