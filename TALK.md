# TALK: Building AI Data Flywheels with GEPA

Agent Loop Chicago, **Nov 17**. 35 minutes, live, with a computer.

> Every agent is full of text a human wrote once and never revisited: system
> prompts, node instructions, tool descriptions, even the config that picks
> which model runs which step. Meanwhile the agent has produced thousands of
> traces that could tell you exactly where those guesses are wrong.

This is the run sheet: what to say, what to type, what appears on screen, and
what to do when it does not. Read the "Status" column before rehearsing —
what is built and what is not changes as work lands, and the blocking work is
named where it bites. Sections 1 to 5 have all been run against a live model;
section 6 has not been built. What is missing before Nov 17 is a promotion —
both searches ran and neither produced one worth committing, for reasons the
gap list names.

## Live, or pre-baked

Settle this first, because it decides whether the talk fits.

| Runs live | Takes | Why it can be live |
|---|---|---|
| A cold turn (T1) | ~34s | One question, one answer |
| Two cached turns (T2, T3) | ~8s, ~6s | The payoff, and it is fast |
| One question + a typed verdict | ~40s | The human-in-the-loop beat |
| `gepa-extract --probe-only` | ~30s | Four probes, four model calls |
| Reading artifacts and diffs | instant | Files |

| Pre-baked | Takes | Why it cannot be live |
|---|---|---|
| Recording all nineteen golden cases | ~5 min | Nineteen cold turns, unattended |
| `gepa-tools` or `gepa-config --probe-only` | ~20 min | Nineteen cold turns, one per golden case; the same turns either way |
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

- The corpus recorded: `make corpus`, all nineteen golden cases. Unattended and
  about five minutes, so it can be done on the day, but do it the night before
  anyway so the traces are there to point at. Every question is asked with the
  memory off, so it leaves the demo's memory alone.
- At least one verdict typed by hand through `sql-agent ask`, so the Langfuse
  view has a human comment beside the computed ones. That is the screenshot
  section 2 rests on.
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

**Status: both halves are built. `sql-agent ask` has the menu, `make corpus`
files the computed verdicts, and both have been run against the live model. No
reader consumes the verdicts yet — say so, it is the honest part of the story.**

### Say

- To improve any of those prompts I need examples with a verdict on them. The
  traces already hold nearly everything: the question, every tool call and its
  arguments, the SQL that ran, which prompt produced the turn, the tokens.
- What a trace does not hold is whether the answer was **right**. That is the
  one thing no machine here can supply, and it is the entire reason a human is
  in this loop.

### Type

```bash
sql-agent "how many customers are in the west region?"
```

One question, cold. When the answer lands:

```
Was that right?
❯ 1. OK
  2. Not OK
```

Choose `2` on a question you know it got wrong, and type a real reason.

### Say, while it files

- The verdict goes onto that turn's trace, beside the tool calls and the SQL,
  as a score named `correct` with your prose as its comment.
- That prose lands in the same field the metric's own feedback lands in, which
  is the point: by the time GEPA reads it, nothing distinguishes what a person
  wrote from what a metric wrote.

### Then, the other way to get one

```bash
make corpus
```

- The label is only unautomatable where nobody knows the answer. For the demo I
  do: `demo/golden/` holds a reference query per question, and nine of them
  carry the number `demo/demo.sql` fixes. So this asks all nineteen and
  compares, unattended.
- Be precise about what that means. **I still wrote the labels** — I wrote
  nineteen reference queries instead of pressing `1` nineteen times. The human
  effort moved; it did not vanish. You can automate the verdict exactly when
  you already have ground truth, and getting ground truth is the expensive
  part. A customer's warehouse does not come with it.
- The other ten have no written answer because theirs moves with the date.
  They are asked anyway: their traces are what the harvest reads.

Then, in Langfuse: the scores on the traces, one typed by hand and nine
computed, indistinguishable in the same field.

- **What nothing does yet: read them back.** The corpus the optimiser trains on
  was authored by hand, not harvested from these verdicts. Closing that loop —
  an approved turn becoming a golden case, with the SQL that turn ran as its
  reference — is the next thing on the list and it is the flywheel this talk is
  named after. Better to say that than to imply a loop that is not joined.
- The questions are aimed at the traps on purpose. A corpus the agent already
  answers perfectly measures nothing, because a candidate cannot win where the
  seed already wins. Measured: the seed gets 10 of 19.

---

## 3. GEPA on one node's prompt — 7 min

**Status: run, on a real corpus of 37 cases harvested from the corpus run. The
winner scores 0.989 against the seed's 0.958 and clears all four probes. Not
promoted: it is 3.6 times longer than the seed and the metric cannot see prompt
length. That is section 4's punchline, so hold it until then.**

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

The five terms are the ones named in every artifact this talk shows, so put
them up once and leave them up:

| term | weight | scores | what goes wrong without it |
|---|---:|---|---|
| `grounding` | 0.35 | the share of recipes whose SQL fragment really is a fragment of the query that ran, **and** says enough to be wrong | a recipe claims something the query never did, and every later question composes on it |
| `census` | 0.25 | the share of claims that are not a count or a percentage | "we have 1,840 customers" is right today and wrong forever, and nothing revisits it |
| `names` | 0.20 | the share of entries that do not overwrite, paraphrase or collide with a filed name | the upsert reports success either way, so the only symptom is a later answer built from the wrong recipe |
| `shape` | 0.10 | how close the batch is to 2–6 entries, and claims under 200 characters | the cache is re-sent in full on every turn, so each entry is a bill that recurs |
| `cost` | 0.10 | output tokens against what this same case cost when it was recorded | a prompt that pads what it writes is paid for on every turn, and no other term here notices |

Four things to say over it, in this order:

- **They are in one dict so you can argue with them.** The argument is the
  point. A weight nobody varied is a guess with a decimal point on it.
- **`grounding` is clamped, not copied from the production gate.** The gate
  accepts any token subsequence, so a fragment like `count(*)` verifies against
  any query that counts. Optimising a metric derived from a gate is how you
  destroy the gate, so this one demands the fragment carry the filters and joins
  that make the concept what it is.
- **Two terms are one-sided on purpose.** `cost` can be earned but never
  exceeded, and `shape` is a band rather than a direction. "Fewer entries is
  better" and "cheaper is better" both have the same degenerate optimum:
  record nothing. Which is why —
- **The empty-extraction gate is load-bearing.** `census`, `names` and `cost`
  are all vacuously perfect when nothing was recorded. Without a gate in front
  of them, three of five terms pay a prompt for declining to do its job.

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

**Status: built and run. The front below is measured, from the run committed at
`demo/gepa/extract.pareto.json`. Open the file, or read the table off a slide.**

### Say

- Grounding and cost genuinely trade off. A prompt that cites more of the SQL
  costs more tokens. One scalar hides that trade behind a decision somebody
  already made for you.
- The front is the set of candidates where no other candidate is better on
  everything. Picking from it is a judgement call, and it should be visible that
  a judgement is being made.

### Show

Six candidates, four on the front:

| cand | val | grounding | census | names | shape | cost | chars |
|---|---|---|---|---|---|---|---|
| 0 (seed) | 0.958 | 0.91 | 1.00 | 1.00 | 0.97 | 0.93 | 2,076 |
| 1 | 0.964 | 0.91 | 1.00 | 1.00 | 0.97 | 0.99 | 6,540 |
| 2 | 0.963 | 0.91 | 1.00 | 1.00 | 0.95 | **1.00** | 5,968 |
| 4 | 0.989 | **1.00** | 1.00 | 1.00 | 0.98 | 0.91 | 7,459 |
| best | — | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | |

- **Candidates 2 and 4 are the argument, and they mirror each other exactly.**
  One buys cost at 1.00 and pays grounding down to 0.91; the other buys
  grounding at 1.00 and pays cost down to 0.91. Neither dominates. A single
  weighted number picks between them without telling you it did.
- **The bottom row is reached by nobody.** No candidate in the pool has all five,
  which is what a front is for saying.
- **The seed is on it.** What you already have is not dominated, and a search
  that reports otherwise is a search worth distrusting.

### Pick one, out loud

Candidate 4, because grounding carries 0.35 of the weight and it is the proxy
for the production gate — `grounded_in()` is what stops a recipe claiming
something the query never did. Buying that with cost is the right trade for this
product, and saying so is the judgement the front exists to make visible.

Then say the uncomfortable half: **candidate 4 is 3.6 times longer than the
seed, and no term on that table can see it.** `cost` charges the model's output
tokens, and a prompt is input. Every candidate in the run grew — by 132%, 187%,
215% and 259%. Nothing opposed it. The gate now warns in both directions and the
metric is getting a length term, and until it has one I have not promoted this.

That is not an aside. A metric is a proxy, and a search is a machine for finding
the gap between your proxy and your intent. Here is mine, found by reading the
artifact rather than the exit code.

---

## 5. Things that do not look like prompts — 6 min

**Status: 5a is run; its candidate is at `demo/gepa/tools.candidate.md`, not
promoted. 5b is built and not yet run on Opus; see the gap list.**

### Say

- GEPA represents a candidate as a **dict of named text components** and will
  mutate any of them. For `extract` that dict has one key, because a node has
  one prompt. Nothing made it one key except our own code.
- Three rungs, each less like a prompt than the last. Section 3 was the first.

### 5a. Tool descriptions — the rung that is still prose

The agent's four introspection tools carry descriptions somebody wrote once and
never revisited. They live outside the prompt files, nobody calls them prompts,
and they are in the model's context on every cold turn.

They are four keys of one candidate, scored by running whole cold turns against
`demo/golden/` — nineteen questions with a known answer. A candidate that gets a
question wrong which the seed got right is discarded whatever it scored, and
nothing is averaged.

```bash
make gepa-tools GEPA_ARGS=--probe-only   # the seed over all 19, ~220k tokens
make gepa-tools                          # the search, ~690k tokens, asks first
```

- **Say what it found.** One tool changed, a gain of two cases in nine, six of
  seven proposals rejected. Then say why, because the feedback says why: the
  seed's failures are the revenue questions — cancelled orders, the historical
  price — and no wording of `describe_table` fixes "use the order line's price,
  not the product's". The reflection kept reading SQL semantics, and the
  component could not act on them. A search that finds nothing is a result, and
  this is what one looks like.
- The objection from the room is right: a description is a prompt with a
  different address. Which is why the next rung is not prose at all.

### 5b. Config — the per-node `effort` block

```yaml
plan:         {effort: low}      fix:      {effort: high}
explore:      {effort: high}     extract:  {effort: low}
generate_sql: {effort: high}     answer:   {effort: low}
```

- Six enum values the model never reads. This is the abstract's phrase, "the
  config that picks which model runs which step", and nobody has measured it.
- **One component, not six.** Five legal values over six nodes is 15,625
  configurations, and a grid at 220k tokens an evaluation is billions. A loop
  can move one node at a time. The reflection can move `explore` down and
  `generate_sql` up in one proposal, because it read which node spent the
  tokens and which node wrote the wrong SQL. Per node the space is tiny, so the
  argument for GEPA here is the feedback and not the search — which is why the
  feedback is what gets shown.
- **The gate is the schema.** A candidate is parsed through the same `Node`
  model `config.yaml` is. An illegal level fails with pydantic's own message,
  `none` on Claude is refused with the file validator's reason, and a candidate
  that names a model is refused outright. Nothing new was written to say
  "never".
- Say what the metric cannot see, because the reflection is told the same.
  `plan` makes no call on a cold turn, so its effort is carried and never
  measured. `extract` and `answer` are charged and not judged: one writes to a
  memory these turns never read, the other writes the sentence above the rows
  that are compared. What is measured is `explore`, `generate_sql` and `fix`,
  and `explore` is where most of a cold turn's tokens go — measured on the dev
  model, 73% of the corpus's spend. That is where a search has room.

```bash
make gepa-config GEPA_ARGS=--probe-only  # the seed over all 19, and where its tokens went
make gepa-config                         # the search, ~690k tokens, asks first
```

Show one reflection from `tools/gepa/out/run/config/reflections.jsonl`: the
per-node ledger it read (`explore (high)  6,210 tokens ...`), the YAML it
proposed, and what that scored. That is one of the two ASI triplets CHALLENGE
asks for, and it is the honesty check with an answer: if the proposal did not
move the node the ledger named, say so and cut 5b.

Then the sentence after. The same block takes a model name per node. The metric
counts tokens, and Anthropic-direct reports no dollars, so that search needs a
price table before it is honest. Not built.

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
| `make corpus` refuses to start | It preflights one thing: tracing on, because a verdict needs somewhere to land. Run `make config` and read the tracing line. |
| The cache is not empty at T1 | `make reset`. It empties the memory and reseeds the demo database. |

## The gap list

What must exist before this script runs end to end. Each points at the section
of `CHALLENGE.md` that specifies it.

| Blocks | What is missing | Where |
|---|---|---|
| §2 | Nothing reads a verdict back. An approved turn should become a golden case, with the SQL it ran as the reference | a scores reader in `app/tracing.py`, and a second harvest beside `extract_cases` — CHALLENGE §1 |
| §3, §5 | A promotion, committed. Both searches ran and neither produced one I would promote: `extract`'s winner is 3.6x longer and the metric cannot see length, `tools`' gain is 2 cases in 9 | a `length` term in `metric_extract.WEIGHTS`, then re-run; and a second `tools` run on another split to see whether the gain reproduces |
| §5b | The effort search, run on the talk's model, with `config.local.yaml` moved aside. Nothing has been run yet; the dev overlay sets every node to `none` on a different model, and a profile found there says nothing about Opus | `make gepa-config GEPA_ARGS=--probe-only` (~220k tokens), then `make gepa-config GEPA_ARGS='--pareto demo/gepa/config.pareto.json'` (~690k), then copy `tools/gepa/out/run/config/reflections.jsonl` and the stderr somewhere tracked before the next run wipes them |
| §5 | Before and after as numbers: T1 tokens and tool calls, seed against promoted, and for 5b where the seed's tokens went by node | `--probe-only` on either whole-turn target is the before half; it now prints the per-node split |
| §6 | A second search with the gate disabled, kept whatever it produces | CHALLENGE, "What the talk has to show" |
