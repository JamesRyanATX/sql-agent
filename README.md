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

#### What's happening behind the scenes

The first question is expensive. The agent has never seen the database, so it
goes and looks: lists tables, describes columns, samples values, writes SQL,
fixes it when it breaks. Along the way it finds the things that make the obvious
query wrong - customers are soft-deleted, `region` casing is inconsistent,
cancelled orders still sit in `orders`. Then it writes down what it learned, in
English, as cache entries. Every question after that is cheaper, because the next
turn starts from those entries instead of from nothing - including questions it
has never been asked, which get answered by composing what earlier turns found.

What accumulates is a semantic layer, derived from exploration rather than
hand-maintained.

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
make demo          # live take
make demo-verify   # check the take from the turn table
```

`make demo` needs [VHS](https://github.com/charmbracelet/vhs).

## Design

See [PLAN.md](PLAN.md) for architecture details and the implementation plan from Claude.