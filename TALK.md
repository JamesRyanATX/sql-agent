# TALK: Building AI Data Flywheels with GEPA

Agent Loop Chicago, **Nov 17**. 35 minutes, live, with a computer.

> Every agent is full of text a human wrote once and never revisited: system
> prompts, node instructions, tool descriptions, even the config that picks
> which model runs which step. Meanwhile the agent has produced thousands of
> traces that could tell you exactly where those guesses are wrong.

This is the run sheet: what to say, what to type, what appears on screen, and
what to do when it does not. Read the "Status" column before rehearsing —
what is built and what is not changes as work lands, and the blocking work is
named where it bites. Sections 2, 3 and 6 are blocked on one thing: a `make
corpus` run, which is forty minutes at a keyboard and cannot be done on the
day.

## Live, or pre-baked

Settle this first, because it decides whether the talk fits.

| Runs live | Takes | Why it can be live |
|---|---|---|
| A cold turn (T1) | ~34s | One question, one answer |
| Two cached turns (T2, T3) | ~8s, ~6s | The payoff, and it is fast |
| One corpus question + verdict | ~40s | The human-in-the-loop beat |
| `gepa-extract --probe-only` | ~30s | Four probes, four model calls |
| Reading artifacts and diffs | instant | Files |

| Pre-baked | Takes | Why it cannot be live |
|---|---|---|
| Recording and judging all 23 questions | ~40 min | Twenty-three cold turns |
| `gepa-tools --probe-only` | ~20 min | Nineteen cold turns, one per golden case |
| Every GEPA search | 20 min – 2 hrs | 60–150 metric calls |
| The overfit run | ~15 min | A second search |

**Say this out loud.** A search that takes an hour is not a broken demo, it is
what optimisation costs. The artifacts are the evidence; the live turns are the
proof that the thing being optimised is real.

## Prerequisites

### The night before

```bash
# The model. config.yaml is claude-opus-5; the gitignored overlay must not
# be in the way, or the talk's numbers are somebody else's.
grep ANTHROPIC_API_KEY .env          # must be set
ls config/config.local.yaml          # must NOT exist — move it aside
make up && make migrate && make seed
make langfuse-up                     # verdicts need somewhere to land
make config                          # claude-opus-5, per-node efforts, tracing on
make test                            # green
```

- The corpus recorded and judged: `make corpus`, all 23 questions answered.
  Every question is asked with the memory off, so this leaves the demo's
  memory alone and can be done days ahead without a reset afterwards.
  This takes 40 minutes and cannot be done on the day.
- Every GEPA run finished, with its artifacts committed. Nothing in
  `tools/gepa/out/` survives a `git clone`, so what the talk shows must be
  committed somewhere tracked.

### Ten minutes before

```bash
make health                          # the API is up
make reset                           # T1 is genuinely cold — this is the stage button
```

- Terminal at the recorded take's geometry: 1200x700, 16pt, bare `PS1='$ '`,
  and `.venv/bin` on `PATH` so the commands are `sql-agent …` rather than
  `uv run sql-agent …`. See `demo/demo.tape` lines 55-87 for exactly this.
- Langfuse open at http://localhost:3000 on the traces view, logged in.
- `demo/demo.gif` open in a second window. That is the network fallback for
  section 1, and it is a recording of this same script.

---

## 1. The agent, and why it gets cheaper — 7 min

**Status: this works today, and needs nothing built.**

### Say

- This is a text-to-SQL agent over a deliberately booby-trapped database. Forty
  tables, four that matter. Every number in it derives from modular arithmetic,
  so 1,840 active customers is a fact, not a sample.
- Watch the last line of each turn. That line is the whole product.

### Type

```bash
sql-agent "How many customers do we have?"
```

Exploration scrolls: tables listed, columns described, a column sampled. Then:

```
11,505 tokens (… in / … out) in 34.0s  [explored]
```

The totals here are the measured ones from the README's table, on
`claude-opus-5` from a cold cache. The in/out split moves run to run; the shape
of the line does not.

- It found the trap. 2,000 customer rows, 160 soft-deleted, and the honest
  answer is 1,840. Nobody told it about `deleted_at`; it sampled the column.

```bash
sql-agent cache
```

- Six entries, in English. Read two aloud. This is not embeddings and not a
  vector store — it is notes a colleague would leave you, and you can edit them.

```bash
sql-agent "How many customers do we have?"
```

```
371 tokens (… in / … out) in 8.0s  [no exploration]
```

- Same question, same answer, thirty times cheaper.

```bash
sql-agent "How many customers are in the west region?"
```

```
475 tokens (… in / … out) in 6.0s  [no exploration]
```

- **This is the part that is not memoisation.** That question had never been
  asked. It is answered by composing what earlier turns filed away, including
  that the region column holds `west`, `West` and `WEST`, so the answer is 460
  and not 350.

### Then, the architecture, briefly

```
load_cache → plan → {execute | explore} → generate_sql → execute → {fix} → extract → answer
```

- Eight nodes. One of them, `extract`, decides what gets written back to the
  cache. Exactly one function talks to a model, which is why every call can be
  traced and every prompt can be swapped.

### Fallback

The GIF. It is the same script, recorded, with real numbers.

---

## 2. A corpus from traces, and the human in it — 6 min

**Status: the command is built and committed (`c1b8f0f`), but it has not been
run against the full question list yet.**

### Say

- To improve any of those prompts I need examples with a verdict on them. The
  traces already hold nearly everything: the question, every tool call and its
  arguments, the SQL that ran, which prompt produced the turn, the tokens.
- What a trace does not hold is whether the answer was **right**. That is the
  one thing no machine here can supply, and it is the entire reason a human is
  in this loop.

### Type

```bash
make corpus
```

One question runs cold. When the answer lands:

```
Was that right?
❯ 1. OK
  2. Not OK
```

Choose `2` on a question you know it got wrong, and type a real reason.

### Say, while it files

- An **OK** is not just a thumbs-up: it certifies the SQL that ran, which
  becomes the reference a later metric scores candidates against. Approval
  fills the corpus as a by-product of using the thing.
- A **Not OK** gives prose. That prose lands in the same field the metric's own
  feedback lands in, and GEPA cannot tell which of us wrote it.

Then, in Langfuse: the trace, with the score attached, beside the tool calls and
the SQL. Then the finished corpus: 23 questions, each judged.

- The questions are aimed at the traps on purpose. A corpus the agent already
  answers perfectly measures nothing, because a candidate cannot win where the
  seed already wins.

---

## 3. GEPA on one node's prompt — 7 min

**Status: the tooling works, but it has never run on a real corpus. It is
blocked on section 2's corpus, and then on one search.**

### Say

- GEPA is genetic, not gradient descent. Candidates are mutated and bred, and
  selection keeps a Pareto front rather than stepping down a slope.
- The part that behaves like a gradient is the **Actionable Side Information**:
  the prose describing *why* a candidate scored what it did, which the
  reflection model reads before proposing the next mutation. It points a
  direction the way a gradient vector does. That is the whole argument for GEPA
  over a grid search.

### Type

```bash
make gepa-extract GEPA_ARGS=--probe-only
```

Four probes, each a real model call, each defending an invariant that exists
only as prose:

```
  PASS  census          Neither kind is a census.
  PASS  grounding       Everything you record must be supported by that SQL …
  PASS  scope_creep     Overwriting a general rule with a special case …
  PASS  near_collision  Every entry needs a short, stable `name` …
```

Each line is the invariant as the prompt states it, and under it the file that
says so. A recorded count is right today and wrong forever after. A recipe must
be copied from the SQL that actually ran. Two names one edit apart overwrite
each other silently, and the upsert reports success either way.

### Say, on the metric

- Five weighted terms: grounding 0.35, census 0.25, names 0.20, shape 0.10,
  cost 0.10. They are in one dict so you can argue with them, and the argument
  is the point.
- `grounding` is clamped rather than copied from the production gate, because
  optimising the metric derived from a gate is how you destroy the gate. A
  fragment like `count(*)` verifies against any query that counts.
- The empty-extraction gate is load-bearing: three of the five terms are
  vacuously perfect when nothing is recorded, so without it the metric rewards
  a prompt for doing nothing.

### Then the pre-baked run

Walk the recorded stderr: the harvest line, the split, the search line, the
probe gate, the diff.

- **The gate is outside the objective, deliberately.** Weights cannot express
  "never". A mean-maximising search will trade a rare catastrophic failure for a
  broad small gain whenever the arithmetic allows.
- A candidate that regresses a probe the seed passed is discarded whatever it
  scored. A candidate that merely scores below the seed is also discarded —
  observed in an early run: seed 0.959, best survivor 0.928.
- **stdout is the new prompt and nothing else.** Progress, the diff and the gate
  go to stderr. So promotion is `make gepa-extract > config/prompts/extract.md`
  followed by `git diff` — a review, not a copy-paste.

---

## 4. What a Pareto frontier of prompts looks like — 3 min

**Status: not built yet. GEPA computes the per-objective scores on every
evaluation and hands them back on the result; what is missing is the code that
writes them down. The gap list says where.**

### Say

- Grounding and cost genuinely trade off. A prompt that cites more of the SQL
  costs more tokens. One scalar hides that trade behind a decision somebody
  already made for you.
- The front is the set of candidates where no other candidate is better on
  everything. Picking from it is a judgement call, and it should be visible that
  a judgement is being made.

Show the five-dimensional front over the candidate pool, and pick one, out loud,
for a stated reason.

---

## 5. Things that do not look like prompts — 6 min

**Status: 5a is built — `make gepa-tools` — and has never been run against a
model. 5b is not built and may not be worth building; see the honesty check.**

### Say

- GEPA represents a candidate as a **dict of named text components** and will
  mutate any of them. For `extract` that dict has one key, because a node has
  one prompt. Nothing made it one key except our own code.
- The thesis is that dict. Anything that is text and that a human wrote once is
  a component.

### 5a. Tool descriptions

The agent's four introspection tools carry descriptions somebody wrote once and
never revisited. They are prose, they are in the prompt on every cold turn, and
nobody has ever measured them.

They are now four keys of one candidate, scored by running whole cold turns
against `demo/golden/` — nineteen questions with a known answer. A candidate
that gets a question wrong which the seed got right is discarded whatever it
scored, and nothing is averaged.

```bash
make gepa-tools GEPA_ARGS=--probe-only   # the seed over all 19, ~220k tokens
make gepa-tools                          # the search, ~690k tokens, asks first
```

Show the before and after: T1 tokens and tool-call count, seed versus promoted,
on the same question. The probe-only run is the before half.

### 5b. Config — `max_tool_calls`, currently 24

- One integer, with a legible failure on each side. Too low and the cold turn
  gives up mid-exploration. Too high and it wanders through eleven thousand
  tokens.
- It is coupled to the tool descriptions being searched beside it, which is the
  point of searching them in one candidate: better descriptions need fewer
  calls.
- **The honesty check**: the search space here is tiny, so the argument for GEPA
  over a for-loop is not the search, it is the feedback. If the reflection is
  not visibly reading the tool-call trace, this section should be cut. Show the
  reflection input and output, not just the number that changed.

---

## 6. How it breaks when you let it overfit — 3 min

**Status: this is not built yet. It is a second search, run with the guards
switched off.**

### Say

- Three training cases, the regression gate disabled, and a generous budget.
- The winner scores beautifully and is worse. Here is what the held-out cases
  say about it.
- The code names this failure before it happens: under twelve cases it warns
  that GEPA will fit whichever questions happen to be in there. A metric is a
  proxy, and a search is a machine for finding the gap between your proxy and
  your intent.
- Which is why the gate exists, why it is outside the objective, and why the
  corpus is the expensive part of all of this.

---

## 7. Close — 2 min

- The loop is four things: a corpus, a metric, a gate, and a promotion you can
  review.
- Your agent already produces the first one. It is in your traces right now.
- The metric is the hard part, and it is hard in a way that is specific to your
  product — which is the good news, because it means the thing that makes this
  work is the thing you already understand.

---

## Timing

| # | Section | Min |
|---|---|---:|
| — | Open, the claim | 2 |
| 1 | The agent, and why it gets cheaper | 7 |
| 2 | A corpus from traces, and the human in it | 6 |
| 3 | GEPA on one node's prompt | 7 |
| 4 | The Pareto frontier | 3 |
| 5 | Things that do not look like prompts | 6 |
| 6 | How it breaks when you let it overfit | 3 |
| 7 | Close | 2 |
| — | **Slack** | **–1** |

Thirty-six against a thirty-five minute slot, so something gives. Section 4 is
the one to cut to two minutes; section 6 is the one never to cut, because a talk
that claims a failure mode without showing it has not tested it.

## When it goes wrong

| Symptom | Do this |
|---|---|
| No network, or the API is unreachable | Switch to `demo/demo.gif`. Same script, real numbers, recorded. |
| A turn hangs past ~60s | Keep talking through the architecture; the cost line lands when it lands. Do not Ctrl-C into a dead terminal. |
| A live turn answers **wrong** | Take it. That is section 2's whole argument, arriving early: press `2`, say why, and point out that the verdict just became training data. |
| Langfuse shows nothing | Traces take seconds to ingest, and the read path is not the write path. Move on and show the pre-baked traces. |
| `make corpus` refuses to start | It preflights two things: somebody at a terminal, and tracing on. Run `make config` and read the tracing line. |
| The cache is not empty at T1 | `make reset`. It empties the memory and reseeds the demo database. |

## The gap list

What must exist before this script runs end to end. Each points at the section
of `CHALLENGE.md` that specifies it.

| Blocks | What is missing | Where |
|---|---|---|
| §2 | All 23 questions recorded and judged | `make corpus` — built, never run |
| §3 | `make gepa-extract` against a real corpus, promoted and committed | CHALLENGE §7 |
| §4 | A committed front, and the code that writes it | `tools/gepa/cli.py`, after `_search`. `GEPAResult` carries `val_aggregate_subscores`, `per_objective_best_candidates` and `objective_pareto_front`, populated whenever the adapter returns per-term scores, which ours does. gepa 0.1.4 never writes them anywhere, so about fifty lines must. `--pareto <path>` for the tracked copy. |
| §5 | A real `make gepa-tools` run, promoted into `app/tools.py` and committed | built and tested; never run against a model |
| §5 | Before and after as numbers: T1 tokens and tool calls, seed against promoted | `make gepa-tools GEPA_ARGS=--probe-only` is the before half |
| §6 | A second search with the gate disabled, kept whatever it produces | CHALLENGE, "What the talk has to show" |

Deadline for the promotion that sections 3 and 5 rest on: **Oct 10**.
