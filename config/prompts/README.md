# The prompts

Six files, one per node that talks to a model. **The whole file is the prompt** —
what is sent is the file's text, stripped of leading and trailing whitespace and
nothing else. There is no frontmatter and no separator, so there is nothing to
learn and nothing for a loader to get wrong. Notes about a prompt live in this
file, which is why this file exists.

Filenames are the `node=` labels `llm.complete` records as the Langfuse
generation name, so a harvested case, the trace that produced it, the graph node
and the file all name the same thing:

| file | node | called from |
|---|---|---|
| `explore.md` | `explore`, `explore.summary` | the introspection loop — one prompt, two calls |
| `plan.md` | `plan` | sufficiency + the cached-path SQL, in one call |
| `generate_sql.md` | `generate_sql` | the cold path |
| `fix.md` | `fix` | up to `max_fix_attempts` times |
| `extract.md` | `extract` | what gets written to the cache |
| `answer.md` | `answer` | the prose above the result table |

`README.md` is the one non-prompt file the loader tolerates. Any *other* `.md`
here is an error rather than a no-op, and so is a missing or empty prompt: a
file that silently changes nothing means an optimisation run that measures the
seed and reports it as a candidate improvement, which is the kind of wrong that
agrees with itself.

These load **once per process**. `graph.plan` puts its system block behind an
Anthropic cache breakpoint on the promise that the block varies with
`connection_id` alone (PLAN.md §7.1), and a prompt re-read per turn could change
between two turns of one server's life with no symptom but T2 quietly costing
more. Editing a file therefore does nothing until the server restarts —
`docker-compose.yml` puts `config/` in uvicorn's `--reload-dir` so an edit
restarts it in place.

`prompts.fingerprint()` puts 8 hex characters of each file on the turn span, so
a harvest can tell which prose produced which trace.

## Changing one

Edit the file and commit it. The commit message is where the reason goes — which
failure the new wording addresses — because the file body cannot carry a comment.

`make optim-run` searches for a better `extract` prompt and writes the winner to
`optim/out/`; `make optim-diff` shows it beside the invariant checklist; `make
optim-apply` writes it into `extract.md` and leaves it uncommitted for you to
read. Nothing promotes itself.

## Notes

### `answer.md`

This used to say "for someone who did not see the query", and that reader
existed: the whole result was `=> [(1840)]` and the answer text was not printed
at all. Now the rows are a table directly above this, and a model told to "lead
with the number" reads a nine-row result as nine numbers and types the table out
again underneath itself.

Measured on the quarters question: 353 output tokens before, 278 after, so the
saving is real but modest — most of that 353 was thinking, not the recitation.
The reason to change it is the duplication, not the tokens. What a table cannot
say is what was counted and what the shape means, and that is the whole job left
for prose.

Its only cheap automatic proxy is length, and that trade was measured here and
decided the other way, which is why `optim/` will never search this one.

### `extract.md`

The four invariants this prompt is the only home for are asserted by
`tests/probes/extract/*.json`, and `make optim-probe` runs them: no census,
recipes grounded in the SQL that ran, no scope creep, no near-collision with an
already-filed name. A candidate that regresses one the seed passed is discarded
whatever it scored — see `optim/run.py`.

### `plan.md`

Load-bearing: it writes the SQL in the same call as the sufficiency decision.
Splitting them costs a second round trip that re-sends the whole cache, and
sufficiency without SQL is downgraded to insufficient in `graph.py`.

## Promotion log

Appended by `make optim-apply`. A line here is a machine's claim about a score;
the commit beside it is the human's claim about why.

<!-- optim-apply: append below -->
