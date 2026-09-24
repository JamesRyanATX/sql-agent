# sql-agent

A self-optimizing SQL agent that gets cheaper with every question.

![The agent answering five questions against a database it has never seen](demo/demo.gif)

## Example

End to end on `claude-opus-5`, from a cold cache:

| Turn | Explored | Tokens | Time | Answer |
|---|---|---:|---:|---|
| T1 — how many customers do we have? | 5 tool calls | **11,505** | 34s | 1,840 ✓ |
| T2 — the same question again | **none** | **371** | 8s | 1,840 ✓ |
| T3 — how many in the west region? | **none** | 475 | 6s | 460 ✓ |
| T4 — orders per quarter? | 3 tool calls | **14,254** | 48s | 9 quarters ✓ |
| T5 — project the next two quarters | **none** | **1,416** | 10s | a least-squares fit ✓ |

**T1** primes the agent's memory.

**T2** is the same question and the same answer, but cheaper now. 

**T3** had never been asked, but costs about the same as T2.

**T4** goes up again, and that is the honest shape of the claim: T1–T3 are all
about `customer`, and `orders` is a part of the schema the agent has never seen.
It does not get cheaper at everything — it gets cheaper at what it has seen.

**T5** is the one worth watching. Nothing in the cache is a forecast. What T4
filed away is that orders are dated by `created` rather than `created_at`, that
`status` includes cancellations, and how to bucket them into quarters — and from
those three facts the agent writes a least-squares fit with `regr_slope` and
projects two quarters forward, without opening the schema again.

## Quickstart

```bash
# Two files to copy. Keys go in .env; which model runs is config.local.yaml.
cp .env.example .env                                        # paste your key
cp config/config.local.yaml.example config/config.local.yaml

# Bring up environment, including demo database with 2k customers
make up && make migrate && make seed

# Re-generate video demo
make demo

# Display recipe cache / memory
make cache

# Display tokens-per-turn table
make turns
```

`.env` holds secrets and addresses. Everything else is
[config/config.yaml](config/config.yaml), which is tracked and ships pointed at
`claude-opus-5` through Anthropic directly:

```yaml
model:
  provider: anthropic     # or openai_compat, for anything OpenAI-shaped
  model: claude-opus-5

plan:    {effort: low}    # per node — see PLAN.md §7.1 before lowering one
explore: {effort: high}
```

To run against something else without editing a tracked file, put just the keys
you want to change in `config/config.local.yaml`. It is gitignored and merged
key by key, so this block switches the endpoint and leaves everything else —
`max_tokens`, `timeout`, every node's effort — as the tracked file has it:

```yaml
model:
  provider: openai_compat
  model: qwen3:32b
  url: http://192.168.1.10:11434/v1   # not localhost — the API is a container
```

Three providers. `anthropic` talks to the SDK. `openai_compat` is any
OpenAI-shaped endpoint — Ollama, vLLM, LM Studio, OpenAI itself. `openrouter` is
OpenAI-shaped too and is worth naming separately, because three things differ
and each costs money to get wrong: effort goes in a `reasoning` object rather
than a flat field that gateway ignores, prompt caching needs an explicit marker
or every cached turn pays full price, and the response reports what the call
charged.

```yaml
model:
  provider: openrouter
  model: google/gemini-2.5-flash       # anything in their catalogue
  url: https://openrouter.ai/api/v1

gepa:                                  # a better model for the one node that
  model:                               # proposes rewrites, while the hundreds
    provider: openrouter               # of rollouts stay cheap
    model: anthropic/claude-sonnet-5
    url: https://openrouter.ai/api/v1
```

Where the backend reports a charge, `sql-agent turns` grows a cost column and a
total. Where it does not, the column is absent rather than showing zeros, since
a turn nobody priced is not a free one. `max_spend` in `config.yaml` stops a
long run — recording a corpus, running a search — once it has spent that much.

Any node may take its own `model:` block on the same keys. The server says at
startup when an overlay is in effect, because what is answering questions should
never be a surprise — and `sql-agent config` prints the merged result, shaped
like the file, with every line the overlay decided marked.

## The CLI

`sql-agent` is the client — it talks to the server over HTTP and never touches a
database itself. Against a local `make up` it needs no configuration:

```bash
uv run sql-agent cache
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
# Ask. The subcommand is optional — a bare argument is a question.
sql-agent "how many customers do we have?"
sql-agent -v "how many customers are in the west region?"   # show the work
sql-agent --no-memory "how many customers do we have?"      # ignore the memory

# What it learned and what each turn cost.
sql-agent cache

# Command history
sql-agent turns
# Erase memory
sql-agent reset

# What the server is running: config.yaml under config.local.yaml, overrides marked
sql-agent config
```

With tracing on, `make corpus` files a verdict on each turn's trace as a score
named `correct`. It reads `demo/golden/`, where each case holds a question
and — where `demo/demo.sql` fixes the number — the answer written beside it,
asks each question with the memory off, and compares. Nine of the nineteen get
a verdict that way; the rest have no written answer because theirs moves with
the date, and they are asked anyway because their traces are what a later
harvest reads.

A verdict from `corpus` is **certified** rather than human: somebody wrote the
reference once and a machine applied it. The same endpoint,
`POST /v1/turns/{id}/feedback`, takes a verdict from anyone, and what lands on
the trace is one shape whoever produced it. `sql-agent ask` used to offer a
menu after every answer; it was removed because nothing read what it filed.
The questions are aimed at the traps in `demo/demo.sql`, because a corpus the
agent already answers perfectly measures nothing.

**`--no-memory` asks as though for the first time.** The turn ignores everything
the agent has learned and saves nothing it learns, so it explores and costs what
a first question costs. The memory is exactly as it was afterwards, which is what
makes it safe to run beside a demo. `sql-agent reset` is the other one: that
empties the memory for good.

Give the agent a role holding `SELECT` and nothing else. It only ever reads, and
the session it opens is read-only regardless, but that is a guarantee about the
agent rather than about the credentials you handed it.

**One database, one memory.** `TARGET_DATABASE_URL` is what the agent answers
questions about, and it may name Postgres, MySQL or SQLite — a bare scheme is
mapped onto the driver that is installed, and SQLite is a path:

```bash
TARGET_DATABASE_URL=postgresql://reader:...@db.internal/analytics
TARGET_DATABASE_URL=mysql://reader:...@mysql.internal/reporting
TARGET_DATABASE_URL=sqlite:////data/books.db
```

Pointing it somewhere else means restarting the server, and what the agent
learned about the old database is still in its memory. `sql-agent reset` before
you do.

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

### Telemetry

The token counter says a turn cost 11,500 tokens. Tracing says where they went:
a span per graph node, a generation per model call with its tokens and its cache
reads, and a span per introspection tool call — the 24 a cold turn collapses
into one number.

It captures the question, the prompts, the generated SQL and the rows handed
back to the model. The stack is self-hosted, so none of it leaves the machine,
but it does mean the trace store holds whatever the target database holds. Read
[app/tracing.py](app/tracing.py) before pointing `TARGET_DATABASE_URL` at
something real.

```bash
make langfuse-up   # six containers, ~2GB, UI on http://localhost:3000
```

Then uncomment `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` and run
`docker compose up -d api` — `restart` reuses the environment the container was
created with. The stack seeds itself with that key pair, so there is nothing to
click through first; log in as `dev@sql-agent.local` / `sql-agent-dev`.

To re-record the demo above:

```bash
make demo          # live take
make demo-verify   # check the take from the turn table
```

`make demo` needs [VHS](https://github.com/charmbracelet/vhs).

### GEPA

The cache makes a turn cheaper. This makes the prompt better — a second
flywheel, turning traces the agent has already produced into a search over the
prose that produced them. Three things are searchable: the `extract` node's
prompt, the four tool descriptions, and the per-node effort block. Each is
manual, and a human commits the result.

```bash
make gepa-extract              # read the new prompt
make gepa-extract > new.md     # keep it
make gepa-tools                # the four tool descriptions
make gepa-config               # the per-node effort block
```

Every one of them reads its corpus from Langfuse. `extract` reads its own
recorded calls. `tools` and `config` read whole turns whose trace carries a
reference query — the query the answer should have come from — or a verdict
that the query they ran was right. `make corpus` primes that by asking the
nineteen questions in `demo/golden/` cold and filing each one's reference and,
where the answer is fixed, a verdict; after that the corpus grows from use, and
a verdict or a corrected query filed in the Langfuse UI is in the next harvest.

**One command, needing no other.** It harvests the telemetry, searches, gates
the pool on the probes and prints the winner — fresh corpus and fresh run dir
every time, so a run is never partly made of the last one. `--resume` is the
only way to continue one, and keeps both.

One target per node — `make gepa-answer`, `make gepa-plan` and the rest exist
too, and exit saying what a search for that node would need first.

**stdout is the new prompt and nothing else.** The harvest, the search, the probe
gate and the diff all report on stderr, so the redirect stays readable while it
runs and the file it leaves holds prose with no commentary. Nothing is written
into `config/prompts/` and nothing is committed: you paste the winner in, and
the commit message says which failure the new wording addresses — the one thing
no run can produce.

It reads Langfuse for its corpus, so [Telemetry](#telemetry) has to be on and
some turns have to have happened. It calls a model many times and costs real
tokens.

## Design

See [PLAN.md](PLAN.md) for architecture details and the build plan.