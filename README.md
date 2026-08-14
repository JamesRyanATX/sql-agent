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
make optim-harvest   # recorded extract calls -> optim/out/extract.jsonl
make optim-probe     # do the current prompts still honour their invariants?
make optim-run       # GEPA over one node's prompt, gated on the probes
make optim-diff      # what the winner changed, beside the invariant checklist
make optim-apply     # write it into config/prompts/extract.md
```

`optim-harvest` reads Langfuse, so [Telemetry](#telemetry) has to be on and some
turns have to have happened. `optim-probe` and `optim-run` call a model and cost
real tokens.

**Why it is cheap.** `llm.complete()` is the only function that talks to a model,
and it records every call under a name that is the graph node. Langfuse therefore
already holds a per-node dataset of exact inputs and outputs — and `extract` is
very nearly a pure function of three recorded strings, so scoring one candidate
is one `low`-effort model call against no database at all. A whole turn is 11.5k
tokens and half a minute; this is neither.

**What it optimises against, and what it cannot.** Five weighted terms, because
every cheap metric here is self-defeating alone: the verification gate accepts a
token subsequence, so a prompt optimised purely for "recipes verified" learns to
emit fragments that verify against anything. The metric is a compromise and is
treated as one.

**The probes are the part that says never.** Four authored cases in
[tests/probes/extract/](tests/probes/extract/) assert the invariants the prompt
is the only home for — no census, recipes grounded in the SQL that ran, no scope
creep, no near-collision with an already-filed name. They are deliberately *not*
GEPA's validation set: they run afterwards, against every candidate in the pool,
and one that regresses a probe the seed passed is discarded whatever it scored.
Weights cannot express "never" — a mean-maximising search trades a rare
catastrophic failure for a broad small gain whenever the arithmetic allows, and
these failures surface turns later, in a poisoned cache, where no metric here can
see them.

**Nothing promotes itself.** `optim-apply` writes
`config/prompts/extract.md` and stops — nothing staged, nothing committed — so
what you review is a diff of prose in a tracked file. It refuses a candidate with
no recorded gate, a target with uncommitted changes, and one that scored below
the prompt it would replace. That last one is not hypothetical: a run here scored
the seed 0.959 and its best survivor 0.928. Clearing every probe does not make a
candidate better than what it replaces.

## Design

See [PLAN.md](PLAN.md) for architecture details and the build plan.