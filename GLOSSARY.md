# Glossary

The words this repository, its command output and the talk expect a reader to know. Where the code uses a different word for the same thing, the code's word is given in brackets.

## The agent

- **turn**: one question asked of the agent and everything the agent does to answer it: loading the cache, exploring the schema, writing SQL, running it, writing the answer, recording what it learned.
- **node**: one step of a turn, as a LangGraph node. The six nodes that call a model are `plan`, `explore`, `generate_sql`, `fix`, `extract` and `answer`. Each has a prompt in `config/prompts/<node>.md` and an `effort` in `config/config.yaml`.
- **effort**: how much reasoning a model call may use. One of `low`, `medium`, `high`, `xhigh`, `max`, set per node.
- **cold turn**: a turn that starts with nothing useful in the cache, so it has to explore the schema and write the SQL from scratch. A **cached turn** finds what it needs in the cache and skips those steps.
- **tool**: one of the four read-only functions the `explore` node may call: `list_tables`, `describe_table`, `sample_column`, `count_distinct`. Each has a **tool description**, the text that tells the model when to call it (`app/tools.py`).
- **agent database**: the Postgres database the agent keeps its own cache and turn log in. The **target database** is the one the questions are about. (The code says `db.agent()` and `db.target()`.)
- **trace**: the record of one turn in Langfuse: every model call, every SQL statement, every tool call.

## The cache

- **cache**: what the agent remembers between turns, stored in the agent database and loaded in full at the start of every turn. The talk also calls it the agent's **memory**.
- **entry**: one row of the cache. Every entry has a `name` and a `claim`. (The prompt calls an entry a "note", a "claim" or a "fact" in places; they all mean an entry.)
- **type** of an entry: `schema_fact` or `recipe`. (The prompt and the code's `kind` column mean this.)
- **schema_fact**: an entry stating something true about the shape of the data: what a table is for, what a column means, which values a status column really holds, which column is the soft-delete flag.
- **recipe**: an entry stating how to express a business concept in SQL: `revenue`, `active customer`. Its `sql_fragment` is a piece of SQL copied from a query that ran.
- **name**: the key an entry is filed under. Writing an entry under an existing name replaces the entry that was there.
- **pinned entry**: an entry a human wrote. The agent never replaces it.
- **tombstone**: an entry marked as withdrawn. It stays in the cache so the agent knows the claim was withdrawn and does not write it again.
- **census** (in the extract prompt): a row count or percentage recorded as if it were a fact about the schema, such as "160 of 2,000 rows are soft-deleted". Forbidden, because it is true only until a row changes.
- **grounded**: a recipe is grounded when its `sql_fragment` appears, token for token and in order, in the SQL that actually ran. `grounded_in()` in `app/graph.py` checks this.

## The demo data

- **trap**: one of five mistakes the demo database is built to provoke: soft-deleted rows, mixed-case region names, cancelled orders, historical versus current prices, a `created` column where `created_at` is expected (`CLAUDE.md`, `tests/test_traps.py`).
- **golden set** (`demo/golden/`): the nineteen demo questions with, for each, the query the answer should have come from. The answer key, not the training set.
- **reference query**: the SQL a question's answer should have come from. Filed on a turn's trace as a Langfuse score named `reference`.
- **verdict**: a human's judgement that a turn's answer was right or wrong. Filed on the trace as a Langfuse score named `correct`.

## The GEPA search

- **GEPA**: the prompt optimiser (`pip install gepa`). It proposes rewrites of a text, scores each one, and keeps the ones that score better. The **reflection model** is the model GEPA asks to write a proposal after reading the feedback.
- **target** (of a search): the text one `make gepa-<target>` command searches over: `extract` is the extract prompt, `tools` is the four tool descriptions, `config` is the six efforts. (Unrelated to the target database.)
- **component**: one text inside a candidate. `extract` and `config` candidates have one component; a `tools` candidate has four, one per description.
- **candidate**: one version of the text being searched. The **seed** is the candidate the search starts from; the **working text** is the text the agent actually runs today. They are the same unless the search starts from a file (`--from`), as `make gepa-extract` and `make gepa-config` do.
- **case**: one recorded question, with what is needed to score an answer to it. A **training case** is one GEPA shows the reflection model; a **validation example** is one used only to score candidates. The **split** is how the harvested cases are divided between the two.
- **harvest**: reading cases out of Langfuse traces. The **corpus** is the result.
- **rollout**: running the agent once, on one case, with a candidate in place of the working text. For `tools` and `config` a rollout is a whole cold turn.
- **metric call**: one scoring of one case. The **budget** is the number of metric calls a search may spend.
- **score**: one number between 0 and 1 for one candidate on one case. `make gepa-<target>-pareto` prints what the number means for each target.
- **feedback**: the text the metric writes beside a score, saying what went wrong. The reflection model reads it.
- **minibatch**: the training cases GEPA scores a new proposal on before deciding whether to keep it. A proposal is kept when its total on the minibatch is higher than its parent's. `extract` uses one case per proposal; the others use three.
- **parent**: the candidate a proposal was written from.
- **pool**: every candidate kept so far, the seed included.
- **best on**: the validation examples a candidate has the top score on. A candidate is **eligible** to be a parent when it is best on at least one example. GEPA calls this set the **Pareto frontier**.
- **val**: a candidate's mean score over the validation examples. The final winner is the candidate with the highest `val`.
- **baseline**: for `tools` and `config`, what the seed's own turn spent on a question, in tokens and tool calls. Every candidate's spend on that question is measured against it.
- **probe**: a test of the extract prompt. It gives the model one made-up turn and checks one rule about what it records. There are four in `tests/probes/extract/`. The **invariant** is the rule a probe checks.
- **gate**: a check a candidate must pass after the search, whatever it scored. The probe gate throws out any candidate that fails a probe the seed passed. A **survivor** is a candidate that passed.
- **finalist**: a candidate printed in full at the end of the report because it is eligible.
- **overfit run** (`make gepa-extract-overfit`): the same search with three cases as both training and validation and the gates only reporting, to show what the guards prevent. **HELD OUT** in its report is the cases it never saw.
- **compare**: the line at the end of a search that started from a file, giving the working text's score next to the winner's.
- **promotion**: a human pasting a winning candidate into the file the agent runs. Nothing does this automatically.

## The turn score (`tools` and `config`)

- **correct**: how closely the rows a turn returned match the rows of the reference query. Worth 0.60.
- **cost**: the turn's tokens against the baseline. Half its weight at the same spend, full at half the spend or less, less as the spend grows. Worth up to 0.25.
- **tool calls**: the turn's number of tool calls against the baseline, on the same scale. Worth up to 0.15.
- **withheld**: cost and tool calls are not paid on a wrong answer, so the cheapest way to score is never to skip the work.
