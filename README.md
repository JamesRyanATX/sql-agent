# sql-agent

A self-optimizing SQL agent that gets cheaper every question.

![The agent answering three questions against a database it has never seen](demo/demo.gif)

The first question is expensive. The agent has never seen the database, so it
goes and looks: lists tables, describes columns, samples values, writes SQL,
fixes it when it breaks. Along the way it finds the things that make the obvious
query wrong — customers are soft-deleted, `region` casing is inconsistent,
cancelled orders still sit in `orders`. Then it writes down what it learned, in
English, as cache entries. Every question after that is cheaper, because the next
turn starts from those entries instead of from nothing — including questions it
has never been asked, which get answered by composing what earlier turns found.

What accumulates is a semantic layer, derived from exploration rather than
hand-maintained.

## Measured

End to end on `qwen3.6:27b-mlx`, from a cold cache:

| Turn | Explored | Tokens | Time | Answer |
|---|---|---:|---:|---|
| T1 — how many customers do we have? | 3 tool calls | **10,504** | 293s | 1,840 ✓ |
| T2 — the same question again | **none** | **2,778** | 119s | 1,840 ✓ |
| T3 — how many in the west region? | 2 tool calls | 8,423 | — | 460 ✓ |

T2 is the same question and the same answer, 3.8× cheaper. T3 had never been
asked, and still costs less than T1, because the recipes T1 filed are enough to
write the SQL without exploring again.

These are local-model numbers, and the shape is unusual: nearly all the remaining
cost is *output*, because a reasoning model bills its thinking as output.

## Quickstart

```bash
make up && make migrate && make seed   # postgres, schema, 2,000 customers
make t1                                # ask the first question
make cache                             # what it learned, as the model sees it
make turns                             # tokens per turn
```

Copy `.env.example` to `.env` and pick a provider: `PROVIDER=anthropic` for the
demo model, or `PROVIDER=openai_compat` with `OPENAI_BASE_URL` / `OPENAI_MODEL`
for anything OpenAI-shaped (Ollama, vLLM, LM Studio).

To re-record the demo above:

```bash
make demo          # live take, 17-25 min of real model time
make demo-verify   # check the take from the turn table
```

`make demo` needs [VHS](https://github.com/charmbracelet/vhs); if it isn't
installed, the Makefile pulls it and its dependencies through `nix`.

## Design

[PLAN.md](PLAN.md) has the architecture, the graph, the traps in the seed
schema, the phase status, and the measurements behind the table above.
