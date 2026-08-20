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
make up && make migrate && make seed   # two databases + API, then 2,000 customers
make demo                              # generate demo video
make cache                             # what it learned, as the model sees it
make turns                             # tokens per turn
```

Copy `.env.example` to `.env` and put your `ANTHROPIC_API_KEY` in it. That file
holds secrets and addresses; everything else is
[config/config.yaml](config/config.yaml), which is tracked and ships pointed at
`claude-opus-5`:

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

Any node may take its own `model:` block on the same keys. The server says at
startup when an overlay is in effect, because what is answering questions should
never be a surprise.

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
sql-agent connections create reporting --driver mysql+asyncmy \
    --hostname mysql.internal --database reporting --username reader
sql-agent connections create books --driver sqlite+aiosqlite \
    --database /data/books.db          # sqlite is a path and nothing else
sql-agent connections ls
sql-agent connect warehouse

# Ask. The subcommand is optional — a bare argument is a question.
sql-agent "how many customers do we have?"
sql-agent -v "how many customers are in the west region?"   # show the work

# What it learned and what each turn cost.
sql-agent cache

# Command history
sql-agent turns
# Erase memory
sql-agent reset
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

### Telemetry

The token counter says a turn cost 11,500 tokens. Tracing says where they went.

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
prose that produced them. It is built for the `extract` node only, it is manual,
and a human commits the result.

```bash
make gepa-extract              # read the new prompt
make gepa-extract > new.md     # keep it
```

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