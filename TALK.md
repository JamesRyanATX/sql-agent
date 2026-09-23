# TALK: Building AI Data Flywheels with GEPA

Agent Loop Chicago, **Nov 17**. 35 minutes, live, with a computer.

> Every agent is full of text a human wrote once and never revisited: system
> prompts, node instructions, tool descriptions, even the config that picks
> which model runs which step. Meanwhile the agent has produced thousands of
> traces that could tell you exactly where those guesses are wrong.

This is the run sheet: what to say, what to type, what appears on screen, and
what to do when it does not. Read the "Status" column before rehearsing —
what is built and what is not changes as work lands, and the blocking work is
named where it bites. Sections 1 to 4 and 5a have run on the talk's model; 5b
and 6 are built and have run only on the dev model. What is missing before
Nov 17 is those two runs on Opus, and a promotion: three searches have run and
none produced one worth committing, for reasons the gap list names.

## Live, or pre-baked

Settle this first, because it decides whether the talk fits.

| Runs live | Takes | Why it can be live |
|---|---|---|
| A cold turn (T1) | ~34s | One question, one answer |
| Two cached turns (T2, T3) | ~8s, ~6s | The payoff, and it is fast |
| `gepa-extract --probe-only` | ~30s | Four probes, four model calls |
| `gepa-extract --iterations 1` | ~1 min | One proposal, seen being made |
| Reading artifacts and diffs | instant | Files |

| Pre-baked | Takes | Why it cannot be live |
|---|---|---|
| Recording all nineteen golden cases | ~5 min | Nineteen cold turns, unattended |
| `gepa-tools` or `gepa-config --probe-only` | ~20 min | Nineteen cold turns, one per golden case; the same turns either way |
| Every GEPA search | 20 min – 2 hrs | 60–150 metric calls |
| The overfit run | ~15 min | The seed over the corpus, a second search, then every candidate on the held-out cases |

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
- Langfuse open on one of those traces, with the `correct` score visible beside
  the tool calls and the SQL. That is the screenshot section 2 rests on.
- Every GEPA run finished on this laptop, so `make gepa-<target>-pareto` and
  `tools/gepa/out/<target>.run.txt` show last night's run. Then copy each
  run's front and transcript to `demo/gepa/` and commit: nothing in
  `tools/gepa/out/` survives a `git clone`, and the committed copies are the
  fallback for every pre-baked beat.

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

## 2. A corpus from traces, and the human in it — 5 min

**Status: built and run against the live model. `make corpus` asks the nineteen
golden questions and files a verdict on each trace. There was a menu after
every `sql-agent ask` answer; it was removed, because nothing read what it
filed. The human in this loop wrote the reference queries.**

### Say

- To improve any of those prompts I need examples with a verdict on them. The
  traces already hold nearly everything: the question, every tool call and its
  arguments, the SQL that ran, which prompt produced the turn, the tokens. They
  are OpenTelemetry spans; Langfuse is where they land, and where the harvest
  reads them back.
- What a trace does not hold is whether the answer was **right**. That is the
  one thing no machine here can supply, and it is where the human is in this
  loop: not pressing a key after each answer, but writing down what the answer
  should be.

### Type

```bash
make corpus
```

Pre-baked the night before; on stage, show the table it printed and one trace.

- `demo/golden/` holds a reference query per question, and nine of them carry
  the number `demo/demo.sql` fixes. This asks all nineteen with the memory off,
  compares the rows against the written answer, and files the verdict on the
  turn's trace as a score named `correct`, beside the tool calls and the SQL.
- Be precise about what that means. **I still wrote the labels.** I wrote
  nineteen reference queries instead of judging nineteen answers one at a
  time. The human effort moved; it did not vanish. You can automate the verdict
  exactly when you already have ground truth, and getting ground truth is the
  expensive part. A customer's warehouse does not come with it.
- The other ten have no written answer because theirs moves with the date.
  They are asked anyway: their traces are what the harvest reads.
- The questions are aimed at the traps on purpose. A corpus the agent already
  answers perfectly measures nothing, because a candidate cannot win where the
  seed already wins. Measured on the dev model, the seed gets 10 of 19; the
  Opus figure comes from the probe in the gap list.

Then, in Langfuse: one trace, the score on it, in the same field a metric's
own feedback goes into.

- **What nothing does yet: read them back.** The corpus the optimiser trains on
  was authored by hand, not harvested from these verdicts. The verdict is
  recorded where a metric's own feedback is recorded, and that is where a human
  correction would enter. Say that, and no more.

---

## 3. GEPA on one node's prompt — 7 min

**Status: run twice on a corpus of 37 cases harvested from traces. The first
run's winner scored 0.989 against the seed's 0.958, cleared all four probes,
and was 3.6 times longer than the seed, which no term could see. The second
run, with a length term in the metric, is the one at
`demo/gepa/extract.pareto.json`: its survivor scores 0.966 against the seed's
0.966, 64% longer. Neither is promoted. That is section 4's punchline, so hold
it until then.**

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

The six terms are the ones named in every artifact this talk shows, so put
them up once and leave them up:

| term | weight | scores | what goes wrong without it |
|---|---:|---|---|
| `grounding` | 0.35 | the share of recipes whose SQL fragment really is a fragment of the query that ran, **and** says enough to be wrong | a recipe claims something the query never did, and every later question composes on it |
| `census` | 0.25 | the share of claims that are not a count or a percentage | "we have 1,840 customers" is right today and wrong forever, and nothing revisits it |
| `names` | 0.20 | the share of entries that do not overwrite, paraphrase or collide with a filed name | the upsert reports success either way, so the only symptom is a later answer built from the wrong recipe |
| `shape` | 0.05 | how close the batch is to 2–6 entries, and claims under 200 characters | the cache is re-sent in full on every turn, so each entry is a bill that recurs |
| `cost` | 0.05 | output tokens against what this same case cost when it was recorded | a prompt that pads what it writes is paid for on every turn, and no other term here notices |
| `length` | 0.10 | what the model was sent, which is mostly the prompt, against the seed | the prompt is free and nothing opposes adding text: in the run without this term every candidate grew by 132% to 259%, and the winner was 3.6 times the seed with every other term saying it was better |

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
  of them, three of six terms pay a prompt for declining to do its job.

### Then the search, which ran last night

Say what `make gepa-extract` does: harvest the corpus from traces, split it,
run the search, gate the pool, print the diff, and put the new prompt on
stdout. Twenty minutes to two hours, so it ran the night before. Here is what
it left.

If there is a minute to spare, one step of it live: six extract calls and one
reflection, and the proposal scrolls past as it happens.

```bash
make gepa-extract GEPA_ARGS='--iterations 1 --resume'
```

```bash
make gepa-extract-pareto
```

The front, with the time it was written, then one line per candidate on it
showing how its prompt opens, seed first. Read the seed row and the survivor
row: two vals of 0.966, and a `chars` column that says one is 64% longer. The
64% itself is in the transcript's diff, next. Hold the trade-off; section 4 is
about it.

```bash
tail -n 40 tools/gepa/out/extract.run.txt
```

The run's own transcript: the gate's verdict on every candidate, the length
warning, and the diff. `less` it if there is time; the harvest and split lines
are at the top.

Fallback, if the search has not run on the presenting laptop: the committed
copies. `python -m tools.gepa.front demo/gepa/extract.pareto.json` for the
front, and `demo/gepa/extract.run.txt` for the transcript. One line in that
copy is stale: the gate's length warning reads "prompt length is not in the
metric", the wording from before the term landed. The code now says "length is
only 0.10 of the score", and the next run's transcript replaces it.

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

**Status: built and run twice. Two fronts are committed: the run before the
metric had a length term, at `demo/gepa/extract.pareto.before-length.json`,
and the run after, at `demo/gepa/extract.pareto.json`. Both tables below are
read off those two files.**

**On a slide, not the terminal.** Nothing here runs live, and the point is two
rows compared across two tables, which needs both on screen at once. One
terminal beat, so the room sees the numbers come out of a file rather than a
slide:

```bash
make gepa-extract-pareto
```

It prints the front from the last `make gepa-extract` on this machine, as the
table the search printed, with the `chars` column the slide has. That means
the search has to have run on the presenting laptop; if it has not, the
committed copy prints the same way:

```bash
uv run --group gepa python -m tools.gepa.front demo/gepa/extract.pareto.json
```

The same target exists for `tools` and `config`.

### Say

- Grounding and cost genuinely trade off. A prompt that cites more of the SQL
  costs more tokens. One scalar hides that trade behind a decision somebody
  already made for you.
- The front is the set of candidates where no other candidate is better on
  everything. Picking from it is a judgement call, and it should be visible that
  a judgement is being made.

### Show, before the length term

Six candidates, four on the front, five terms:

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
seed, and no term on that table could see it.** `cost` charged the model's
output tokens, and a prompt is input. Every candidate in that run grew — by
132%, 187%, 215% and 259%. Nothing opposed it.

### Show, after

The metric got a sixth term, `length`, at 0.10, with `shape` and `cost` halved
to make room. Same corpus, same budget:

| cand | val | grounding | census | names | shape | cost | length | chars |
|---|---|---|---|---|---|---|---|---|
| 0 (seed) | 0.966 | 0.91 | 1.00 | 1.00 | 0.95 | 0.99 | 1.00 | 2,076 |
| 2 | 0.966 | **1.00** | 1.00 | 1.00 | 0.95 | 0.94 | 0.71 | 3,404 |

- **Growth fell from 259% to 64%.** The term worked.
- **And the aggregate still cannot tell the trade from an improvement.**
  Candidate 2 buys grounding 0.91 to 1.00 and pays length 1.00 to 0.71, and
  the two totals differ by 0.0004. The gate now says exactly this, in yellow:
  length is only 0.10 of the score, read the diff closely. A person reading the
  diff can tell. The number cannot. Not promoted.

That is not an aside. A metric is a proxy, and a search is a machine for finding
the gap between your proxy and your intent. Here is mine, found twice by reading
the artifact rather than the exit code: once when the term was missing, and once
after it was there.

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
  seven proposals rejected; the run's log is `demo/gepa/tools.run.txt`. Then
  say why, because the feedback says why: the
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
per-node ledger it read (of the shape `explore (high)  6,210 tokens ...`; that
number is illustrative until the Opus run supplies one), the YAML it proposed,
and what that scored. That is one of the two ASI triplets CHALLENGE
asks for, and it is the honesty check with an answer: if the proposal did not
move the node the ledger named, say so and cut 5b.

Then the sentence after. The same block takes a model name per node. The metric
counts tokens, and Anthropic-direct reports no dollars, so that search needs a
price table before it is honest. Not built.

---

## 6. How it breaks when you let it overfit — 3 min

**Status: built. `--overfit N` on the search command is the run with the
guards off. Not yet run on Opus; the command and the files it leaves are in
the gap list.**

### Say

- The three cases the seed does worst on as the whole training set, the
  regression gate reporting but deciding nothing, and a generous budget. Same
  prompt, same metric, same corpus as section 3. The worst three, because
  that is what anyone tuning a prompt by hand reaches for, and because a
  random three turned out to be ones the seed already scored 0.996 on, and a
  search with no room finds nothing to overfit.
- The winner scores beautifully and is worse. Here is what the held-out cases
  say about it.
- The code names this failure before it happens: under twelve training cases
  it warns that GEPA will fit whichever questions happen to be in there. A
  metric is a proxy, and a search is a machine for finding the gap between
  your proxy and your intent.
- Which is why the gate exists, why it is outside the objective, and why the
  corpus is the expensive part of all of this.

### Pre-baked

```bash
make gepa-extract-overfit GEPA_ARGS=--resume > demo/gepa/extract-overfit.md
```

`--resume` reuses the corpus section 3 searched; the run itself is fresh and
keeps its own files under the name `extract-overfit`, so section 3's run is
untouched. GEPA needs nothing switched off: it admits a candidate on a
train-minibatch improvement and uses the validation set only to rank parents,
so three cases overfit by default. What is switched off is ours.

On stage:

```bash
make gepa-extract-overfit-pareto          # the front over three, and the held-out table
less tools/gepa/out/extract-overfit.run.txt   # the warning, the gate, the diff
```

### Show, in this order

1. The sweep that picks the three, then the warning: `3 cases is thin — GEPA
   will fit whichever questions happen to be in here`.
2. The front over three cases, in one sentence: the winner looks excellent.
3. The gate's `DISCARDED` lines, then `the gate reported and decided nothing`.
4. The held-out table: training score beside held-out score, and on how many
   unseen cases each candidate lost to the seed. The mean is the number a run
   reports about itself; the count is what the gate would have read.
5. The diff, which is where the prompt is seen naming the three questions it
   trained on.

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
| 2 | A corpus from traces, and the human in it | 5 |
| 3 | GEPA on one node's prompt | 7 |
| 4 | The Pareto frontier | 3 |
| 5 | Things that do not look like prompts | 6 |
| 6 | How it breaks when you let it overfit | 3 |
| 7 | Close | 2 |
| — | **Slack** | **0** |

Thirty-five exactly, with no slack, so if anything runs long section 4 is the
one to cut to two minutes; section 6 is the one never to cut, because a talk
that claims a failure mode without showing it has not tested it.

## When it goes wrong

| Symptom | Do this |
|---|---|
| No network, or the API is unreachable | Switch to `demo/demo.gif`. Same script, real numbers, recorded. |
| A turn hangs past ~60s | Keep talking through the architecture; the cost line lands when it lands. Do not Ctrl-C into a dead terminal. |
| The log says `rate limited`, waiting | Not a hang. The client retries up to six times with waits of up to a minute each. Same advice: keep talking. |
| A live turn answers **wrong** | Take it. Say what the right answer is and why the SQL missed it; that is section 2's argument arriving early, that the answer key is the part only a person can write. |
| Langfuse shows nothing | Traces take seconds to ingest, and the read path is not the write path. Move on and show the pre-baked traces. |
| `make corpus` refuses to start | It preflights one thing: tracing on, because a verdict needs somewhere to land. Run `make config` and read the tracing line. |
| The cache is not empty at T1 | `make reset`. It empties the memory and reseeds the demo database. |

## The gap list

What must exist before this script runs end to end. Each points at the section
of `CHALLENGE.md` that specifies it.

| Blocks | What is missing | Where |
|---|---|---|
| §3, §5 | A promotion, committed. The length term landed and its re-run is committed; the survivor beats the seed by 0.0004 for 64% more prompt, so promoting it is a decision, not a build. `tools`' gain is 2 cases in 9 | decide on `extract`; and a second `tools` run on another split (`--seed 1`) to see whether the gain reproduces |
| §5b | The effort search, run on the talk's model, with `config.local.yaml` moved aside. Nothing has been run yet; the dev overlay sets every node to `none` on a different model, and a profile found there says nothing about Opus | `make gepa-config GEPA_ARGS=--probe-only` (~220k tokens), then `make gepa-config > demo/gepa/config.candidate.yaml` (~690k), then copy `tools/gepa/out/config.pareto.json`, `tools/gepa/out/config.run.txt` and `tools/gepa/out/run/config/reflections.jsonl` to `demo/gepa/` before the next run wipes them |
| §5 | Before and after as numbers: T1 tokens and tool calls, seed against promoted, and for 5b where the seed's tokens went by node | `--probe-only` on either whole-turn target is the before half; it now prints the per-node split |
| §6 | The overfit run on Opus, its three files committed | the command in section 6, after `make gepa-extract` has left a corpus in `tools/gepa/out/extract.jsonl` for `--resume` to reuse; then copy `extract-overfit.pareto.json` and `extract-overfit.run.txt` from `tools/gepa/out/` to `demo/gepa/` |
