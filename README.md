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

## The CLI

`sql-agent` is the client — it talks to the server over HTTP and never touches a
database itself. Against a local `make up` it needs no configuration:

```bash
uv run sql-agent connections ls
```

Put the venv on your PATH and you can drop the `uv run`:

```bash
export PATH="$PWD/.venv/bin:$PATH"
```

Two environment variables point it somewhere else, both optional:

```bash
SQL_AGENT_URL=http://localhost:8000/v1   # the default; the /v1 is part of it
SQL_AGENT_API_KEY=...                    # only if the server has API_TOKEN set
```

```bash
# Point it at a database. `default` is TARGET_DATABASE_URL, already registered.
sql-agent connections create warehouse \
    --hostname db.internal --database analytics --username reader
sql-agent connections ls
sql-agent connect warehouse

# Ask. The subcommand is optional — a bare argument is a question.
sql-agent "how many customers do we have?"
sql-agent -v "how many customers are in the west region?"   # show the work

# What it learned, and what each turn cost.
sql-agent cache
sql-agent turns
sql-agent reset          # forget it all, for one connection
```

Every command takes `-c/--connection` to override the connected one. Give the
agent a role holding `SELECT` and nothing else: it only ever reads, and the
session it opens is read-only regardless, but that is a guarantee about the
agent rather than about the credentials you handed it.

**What the agent learns belongs to the connection it learned it about.** Ask
`warehouse` about revenue and `staging` knows nothing about it — `revenue` means
something different on each, and one shared cache would let them overwrite each
other.

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

The agent runs behind one server, and `sql-agent` is a terminal renderer for
these endpoints — there is no second code path, and the CLI imports nothing from
the server.

```
GET    /v1/connections                the registry — never a password
POST   /v1/connections                register a database
GET    /v1/connections/{id}           one, with cache and turn counts
PATCH  /v1/connections/{id}           partial update
DELETE /v1/connections/{id}           the row, and everything learned about it
POST   /v1/connections/{id}/test      can we reach it, and what are we?
POST   /v1/connections/{id}/ask       one turn, streamed as SSE
GET    /v1/connections/{id}/cache     what it learned, exactly as the model reads it
DELETE /v1/connections/{id}/cache     forget it — cache, turns, checkpoints
GET    /v1/connections/{id}/turns     tokens per turn
GET    /health                        unversioned and open, for a load balancer
```

Everything about learned state hangs off the connection it is about, as a path
segment: there is no unscoped route to reach by accident, because it doesn't
exist.

Everything under `/v1` takes `Authorization: Bearer $API_TOKEN`. Leaving
`API_TOKEN` unset leaves it open, which the server warns about at startup — as
does leaving `CONNECTION_SECRET` unset, which stores registered warehouse
passwords in plaintext.

`make up` starts both databases and the API together, with `app/` mounted and
`--reload` — an edit restarts the server in place, and only a dependency change
needs `make build`.

### One memory, N targets

The agent's memory is on its **own Postgres server**, separate from every
database it answers questions about.

| | `agent-db` :5433 | a target, e.g. `demo-db` :5432 |
|---|---|---|
| holds | the registry, and what the agent has learned | `customer`, `orders`, 38 decoys |
| built by | `migrations/*.sql` | `demo/demo.sql`, locally |
| how many | one | however many are registered |

So the agent can't explore its own cache and cache facts about caching. That
doesn't rest on application code remembering to check: `sql-agent reset` cannot
touch the business data because that connection cannot see it.

Generated SQL can't write either, and this one is worth being precise about.
`demo/demo.sql` builds a `reader` role holding SELECT and nothing else — but a
*registered* connection's credentials are whatever you gave us, so the guarantee
cannot live there. Every connection the agent opens to a target is put into a
read-only session, which binds any role including a superuser, and generated SQL
additionally runs in a read-only transaction with a statement timeout.

`demo/demo.sql` is not the product — it's a booby-trapped fixture so the agent
has something to explore, which is why it sits next to the tape that records it.
Point `TARGET_DATABASE_URL` at a real warehouse, or register one with
`sql-agent connections create`, and nothing else changes.

To re-record the demo above:

```bash
make demo          # live take
make demo-verify   # check the take from the turn table
```

`make demo` needs [VHS](https://github.com/charmbracelet/vhs).

## Design

See [PLAN.md](PLAN.md) for architecture details and the build plan.