# CHALLENGE: GEPA over the four tool descriptions and the config block

**Goal.** Optimise two things in `sql-agent` that are not prompts — the four
`description` strings in `app/tools.py` and the per-node `effort` settings in
`config/config.yaml` — so the Agent Loop Chicago talk (Nov 17) can show a real
before/after diff on each. Today only `extract` is wired
(`tools/gepa/cli.py: WIRED = {"extract"}`), and the `UNWIRED` dict there gives a
written reason for each of the other five nodes. Those reasons are the
requirements for this work, not objections to it.

## What is missing is a whole turn scored against a known answer

Every unwired node and every non-prompt artifact fails for the same reason: the
`extract` metric scores one recorded call in isolation
(`tools/gepa/metric_extract.py`, `tools/gepa/replay.py`), and everything else
needs a **whole turn** scored against a **known answer**. Build that once and
tool descriptions, per-node config, `generate_sql` and `fix` all become
components of the same search.

GEPA already represents a candidate as a dict of named text components and
mutates any of them. The adapter currently pins `COMPONENT = "extract"`
(`tools/gepa/adapter.py`), so that dict has one key. Wiring a tool description is
adding a second key to it. What stops that today is the metric, not GEPA.

### What the feedback text is for

GEPA is genetic, not gradient descent: candidates are mutated and bred, and
selection keeps a Pareto front rather than stepping down a slope. The part that
looks like a gradient is the **Actionable Side Information** — the prose a
candidate's failures are described in, which the reflection model reads before
proposing the next mutation. It points a direction the way a gradient vector
does, which is the entire reason to run GEPA instead of a grid search over a
six-value enum.

In this repo the ASI is already there and already named: `Score.feedback` and
`Score.text()` in `metric_extract.py`, surfaced as the `Feedback` field by
`ExtractAdapter.make_reflective_dataset`. Every metric written below is a metric
*plus* the prose that explains its number, and the prose is the harder half.

That field takes prose from a person as readily as from a metric, and GEPA cannot
tell which wrote it. Item 2 is built on that, and it is the short answer to "how
is this not gradient descent, and why does it still have a direction".

Half of the known-answer warehouse exists: `demo/demo.sql` derives 1,840 and 460
from modular arithmetic, and the `demo-verify` target in the `Makefile` gates on
them, with shape-only predicates where `orders.created` is anchored to `now()`.
Extend that, don't replace it.

## Work items, in order

### 1. Golden set — harvested from traces, then frozen

**Langfuse is the source.** A `turn` trace already holds everything a whole-turn
corpus needs: the question on the span's input, which prose produced it in
`metadata.prompts`, every tool call as a `tool.<name>` span with
arguments and a WARNING level on error, the executed SQL as `sql.execute`'s input
with ERROR on failure, each node's returned delta — `fix_attempts`, `error`,
`explored` — and per-call usage. `turn_scope()` in `tools/gepa/harvest.py`
already reads the prompt fingerprints out of it.

What a trace does not hold is whether the answer was *right*, and that is the
one thing a whole-turn metric needs.

- **The corpus stores a reference SQL, not a number.** Both the reference and the
  candidate's SQL run against the target inside the same rollout, and the two
  result sets are compared. A question anchored to `now()` then stops being a
  special case: both sides run at the same moment. This is `demo-verify`'s
  shape-predicate reasoning taken further, and it keeps its rule — a gate that
  passes today and fails in November is worse than no gate.
- **Where `demo.sql` fixes the number, write the number down too.** 1,840, 460,
  the region and status counts: those do not move, so `expect` records the
  literal answer beside the reference SQL. Reference SQL alone is a comparison
  between two queries, and two queries can be wrong together — the reference was
  subtly wrong when it was captured, or someone edited `demo.sql` and both sides
  moved. A literal number cannot drift quietly. It is the only thing here that
  fails loudly.
- Each line: `question`, `reference_sql`, `expect` where there is a fixed answer,
  `ordered` (whether row order is part of the answer), and `tables` — which
  schema areas it touches, so a split can hold out an area.
- 15–20 cases, and it must clear `THIN_CORPUS = 12` with margin.
- **No traces exist yet.** The first step is `make corpus`: nineteen questions,
  each asked with the memory off, so every turn is cold and the demo's memory is
  left as it was. That run also produces nineteen distinct `extract` calls,
  which is the corpus item 7 has been waiting for. It is unattended: the verdict
  is computed by comparing against the answer written in the case file. A case arrives from it already certified:
  the score says the answer was right, and the `sql.execute` span says what ran.
- `extract` keeps its own trace-based harvest either way (`harvest.py`); this is
  a second harvest beside it, not a replacement.

**Why `demo/golden/` may be committed when `tools/gepa/out/` is not.** That
directory is gitignored because a harvested case holds a recorded prompt, and a
recorded prompt holds whatever the registered warehouse holds — committing a
harvest commits customer data by construction. This corpus comes from
`demo/demo.sql`, which is fabricated, so the reasoning does not reach it. Say
that in the file, or someone will apply the ignore rule by analogy and delete it.

### 2. The label — asked at the prompt, not collected later

The step the abstract promises and the one nobody can automate. It belongs in the
product, one keystroke after the answer it is about:

```
1. OK
2. Not OK
```

and on `2`, one more question: what could be improved about that answer. Both go
to the turn's trace as a score named `correct` and a comment.

- **A `1` certifies the SQL that ran, and that is the reference SQL.** Approval
  produces a labelled case with no second pass and nobody reconstructing an
  afternoon's traces from memory. The corpus fills as a by-product of using the
  agent, which is the reason to build this at all.

  **Not built. The verdict is written and nothing reads it.** `tracing.score`
  has one caller and no counterpart: `harvest.py` filters on span names and
  prompt fingerprints, and `demo/golden/` was authored by hand rather than
  harvested. So the write half is real and the sentence above is still a plan.
  Closing it needs a scores reader in `app/tracing.py` and a second harvest
  beside `extract_cases` that turns an approved turn into a case. Until then,
  say so on stage rather than implying the loop is joined.
- **A `2` produces prose, which is not a scorable case.** "That was wrong" gives a
  metric nothing to compare 150 rollouts against. Such a turn enters the corpus
  only once a corrected `reference_sql` arrives; until then it is *reflection*
  material, which is worth having on its own — it is ASI with a human author,
  landing in the same field the metric writes to.
- Questions the seed got wrong are the valuable ones, because that is where a
  candidate has room to win; a corpus of only approved answers can confirm the
  status quo and nothing else. The five traps in `demo.sql` exist to produce
  exactly this kind of case.
- **Approval is fallible in the way this demo is built to punish.** A trap
  produces a plausible wrong answer, which is precisely what a person presses `1`
  on. That is why item 1 still writes down the numbers `demo.sql` fixes, and it
  is worth saying out loud on stage rather than hoping nobody asks.
- Scoring in the Langfuse UI stays as the fallback for turns taken
  non-interactively, so the harvest reads one shape from two sources.
- The harvest reports what it dropped and why, as `Harvest.report()` already
  does. Unscored is a count, not a silence.

### 3. The feedback path — where that prompt writes to

Small, but real code, and it crosses two boundaries the repo defends.

- `turn_id` and `trace_id` onto the `answer` event (`app/graph.py`), which carries
  neither today. `TurnOut` already exposes `trace_id`, so the precedent is set.
- A scoped endpoint taking the verdict and optional comment. The server writes the
  score through `app/tracing.py`, which grows a write to match its read half.
- **The CLI does not write it.** `tests/test_cli_isolation.py` says the CLI is a
  client of the API and that `langfuse` is import-legal in `app/tracing.py` alone;
  this feature is the first thing that would want to break both, and it does not
  have to. A reader will ask, so answer it in the code.
- **TTY-gated, with a flag to suppress.** `make customer-count`, `--json` and any
  pipe must not sit waiting on a keystroke. When tracing is off there is nowhere
  durable to put a score, so the prompt does not appear — the turn table is wiped
  by `make reset`, which is `harvest.py`'s own argument for Langfuse being the
  home.

### 4. Whole-turn replay — `tools/gepa/replay_turn.py`

Run one golden question through the graph **in-process** with a candidate
injected, and return what happened.

- Injection points: prompt overrides (any of the six in `config/prompts/`),
  tool-description overrides (`app/tools.py`, the four `description` strings
  in the tool spec list), per-node `effort` overrides (`app/config.py`,
  `Node.effort`). Overrides are per-call, never written to disk.
- **Cold every rollout, and nothing kept.** Tool descriptions only matter on the
  cold path, and a warm memory confounds every other component. The turn runs
  with the memory off: it reads nothing and saves nothing, so it behaves like a
  first-ever question every time and leaves your memory exactly as it was.
  Nothing is cleared, which is also what makes concurrent rollouts safe —
  they have nothing to wipe out from under each other. Say this on stage.
- Returns: answer text, result rows, `tokens_in`/`tokens_out` per node,
  tool-call sequence (name + args), fix attempts, errors. Mirror the `Replayed`
  dataclass shape so the metric and adapter code stays thin.
- Generation name must not collide with production node names
  (see `NODE = "extract.replay"` and the reason in `replay.py`), or round two
  trains on round one. The turn span needs the same treatment: a rollout must not
  look like a turn to the next harvest.
- Check `graph.stream_turn` and `build_graph` for the entry point; the prompt
  loader is once-per-process (`config/prompts/README.md`), so the override has
  to go through `prompts.get` or a state field, not the file.

### 5. Metric — `tools/gepa/metric_turn.py`

Same shape as `metric_extract.Score`: a scalar for selection, prose feedback for
reflection.

- Weights in one dict so a reader can argue with them. Start:
  `correct 0.60`, `cost 0.25` (one-sided against the seed baseline, as
  `metric_extract` does), `tool_calls 0.15` (fewer, one-sided).
- **Correctness is result-set comparison** against the reference SQL run in the
  same rollout: a scalar compares numerically, a multi-row result as a multiset
  of value tuples with column names ignored, and an `ordered` case as a list.
  Column names are the model's to choose; the rows are not.
- Wrong answer scores 0 regardless of cost. Cost and tool-call terms are vacuous
  on a wrong answer, and the empty-extraction gate in `metric_extract.score`
  exists for the same reason.
- Feedback text is the point, and it is the ASI. The reflection model needs to
  read *which* tools were called, in what order, what errored, how many fix
  attempts, and which rows differed — that is what makes GEPA better than a grid
  search. Without it the config component is indefensible on stage.
- Hard gate, as the probes do for `extract`: a candidate that gets a golden
  case wrong that the seed got right is discarded whatever it scored.
  Nondeterminism at thinking effort is real; Pareto across cases plus this gate
  absorbs it. **Do not average.**
- **Justify the weights, then test them.** Rerun selection with `cost` at 0.15
  and at 0.35 and report whether the winner changes. If it does, the number is
  load-bearing and the talk owes the audience the argument; if it does not, say
  that too. A weight nobody varied is a guess with a decimal point on it, and
  "the weights are in one dict" is honesty about the guess, not evidence.

### 6. Generalise the adapter — `TurnAdapter`

- `ExtractAdapter` stays. Add `TurnAdapter` with a `components: dict[str, str]`
  seed built from the live prompt files, tool descriptions, and a YAML
  rendering of the per-node config block.
- Wire **tool descriptions first**: prose, GEPA-native, plausible win on
  explore-loop efficiency (fewer `sample_column` / `describe_table` calls on
  T1). Then **config**, as a second component in the same candidate.
- `cli.py`: `WIRED` grows; `UNWIRED` loses entries as they are wired, and the
  remaining entries keep their reasons. `make gepa-tools`, `make gepa-config`
  via the existing `gepa-%` pattern rule. stdout stays the artifact.
- Reflection system prompt (`REFLECT_SYSTEM`) needs a per-component variant:
  "you are improving a tool description" / "you are choosing model effort per
  node" — the generic one will rewrite YAML as prose.
- `EvaluationBatch.objective_scores` is already populated per term, and the
  comment there already argues for `frontier_type="objective"`. That is what the
  front the talk shows is drawn from; keep it populated for the turn metric too.

### 7. Run, promote, show

- `make gepa-extract` with a real corpus first — `tools/gepa/out/` is empty and
  `git log config/prompts/` has no promotion. This is blocking regardless of
  items 1–6. Twenty distinct questions, not the thirteen in the deck notes.
- Then `make gepa-tools`, then `make gepa-config`. Commit each promotion with
  the reason in the message (`config/prompts/README.md`, "Changing one").
- **Nothing is recorded to video.** The talk is 35 minutes, live, with a computer.
  A search is minutes to hours and never runs on stage; a single turn is a live
  beat. So every run happens beforehand and leaves *committed artifacts* — the
  stdout prose, the diff, the front, the ASI triplets, the token and tool-call
  tables — and those are what the audience reads while nothing is running.
- Write the beat list with **measured** durations: run each live beat once, time
  it, and budget the 35 minutes against real numbers rather than an estimate of a
  cold turn that actually takes four minutes. If a turn stalls on stage the
  fallback is the committed numbers for that same question.

## What the talk has to show

Four artifacts, each owed to a specific sentence in the abstract or the review.
None of them fall out of a run by themselves.

- **The Pareto front**, from `objective_scores` over the candidate pool plus
  GEPA's run-dir state. "What a Pareto frontier of prompts looks like" is a
  promise to show one, not to describe one.
- **Two ASI triplets.** The feedback text going in, the rewritten component
  coming out, the score delta. One triplet whose feedback the metric wrote, one
  whose feedback a person typed at the item-2 prompt, shown side by side in the
  same field. The `gepa.reflect` calls are already traced, so this is capture
  rather than instrumentation. This is the slide that answers "how is this not
  gradient descent, and why does it still have a direction".
- **A deliberate overfit run.** Three training cases, regression gate disabled,
  kept whatever it produces, then what the held-out cases say about the winner.
  "How it breaks when you let it overfit" is a demonstration. A talk that claims
  a failure mode without showing it is a talk that has not tested it.
- **The metric rationale**, per term, plus the sensitivity result from item 5.

## Cost and confounds — decide before running

- Cold turns are ~11.5k tokens. 150 rollouts ≈ 1.7M tokens on Opus. Budget 60
  with a 10-case train split is under 700k. Alternative: search on a cheaper
  generation model, validate the winner on Opus — defensible, and a slide.
- If tool descriptions change, `extract` output changes, and the cache changes.
  Cold-every-rollout sidesteps this for the demo. Say so.
- Config as a text component has a tiny search space; the argument for GEPA over
  a loop is the feedback-driven reflection, not the search. If the reflection
  isn't visibly using the tool-call trace, cut config from the talk.
- `harvest.py` notes that Langfuse has inputs and Postgres has outcomes, and only
  Postgres gets reset. Scoring the trace resolves that rather than routing around
  it: the label lives on the trace, beside the inputs, in the store that is not
  reset. Fix the note in `harvest.py` when items 2 and 3 land, so the file stops
  arguing against what it now does.

## Definition of done

1. `demo/golden/`: ≥15 cases, one JSON file each, carrying a reference query
   and — where `demo.sql` fixes the number — the answer beside it. A test
   executes every case against a freshly seeded demo database and fails if one
   errors, comes back empty, or disagrees with its `expect`. **Done**, 19 cases,
   nine with a literal answer. Authored against `demo.sql` rather than harvested
   from traces, because the numbers are reachable without a model and waiting
   for `make corpus` would have blocked the metric and the adapter behind it.
2. `make gepa-tools` exits 0 with a diff on stderr and prose on stdout, from a
   real run, and the promotion is committed.
3. The feedback prompt exists, is TTY-gated, writes a score and comment to the
   turn's trace, and the harvest reads both it and a score set in the UI.
4. Before/after committed as numbers: T1 tokens and tool-call count, seed vs.
   promoted, for the same question.
5. The four talk artifacts above, in the repo.
6. `UNWIRED` still tells the truth for whatever is left. `plan`'s current reason
   will not survive this work — outcome labelling is exactly what items 2 and 3
   build — so it needs the reason that still holds: `plan`'s degenerate optimum
   is the worst in the graph, always say sufficient, and nothing in the golden
   set gates that yet. **Done**, and four of the five needed rewriting rather
   than one: "scoring one call means running the SQL against a warehouse whose
   answers are known" was an argument against building the thing, and
   `demo/golden/` is now a description of it.
