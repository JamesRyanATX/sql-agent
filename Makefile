.PHONY: up down build migrate seed reset reset-gepa test test-live \
        psql-agent psql-demo logs logs-agent logs-demo logs-api health \
        customer-count west-coast-customer-count cache turns config corpus \
        demo demo-verify langfuse-up langfuse-down langfuse-logs

SHELL := /bin/bash
DC ?= docker compose

# `sql-agent` reads these and nothing else. `-include` so a fresh clone with no
# .env still runs; the names are listed explicitly on `export` so an
# ANTHROPIC_API_KEY sitting in .env does not leak into every recipe.
-include .env
# The default has to precede the `export`: `export FOO` on an undefined variable
# defines it as empty, which then makes `?=` a no-op and leaves every recipe
# calling a CLI with no server to talk to.
SQL_AGENT_URL ?= http://localhost:8000/v1
export SQL_AGENT_URL SQL_AGENT_API_KEY
# /health is unversioned, so strip the suffix rather than keep a second variable.
API := $(SQL_AGENT_URL:/v1=)

# Two servers. The agent's memory and the data it queries are not in the same
# place, so neither are the psql invocations that reach them.
PSQL_AGENT = $(DC) exec -T agent-db psql -U agent -d agent -v ON_ERROR_STOP=1
PSQL_DEMO  = $(DC) exec -T demo-db psql -U business -d business -v ON_ERROR_STOP=1

# VHS needs ttyd and ffmpeg on PATH. Prefer a system install; otherwise pull all
# three through nix so recording needs nothing installed permanently.
VHS ?= $(shell command -v vhs 2>/dev/null || \
         echo 'nix shell nixpkgs\#vhs nixpkgs\#ttyd nixpkgs\#ffmpeg -c vhs')

# --- core: environment, migrations, tests ----------------------------------

up:  ## start both databases and the API, and wait until all three are healthy
	$(DC) up -d --wait

down:
	$(DC) down

build:  ## rebuild the API image — only needed when dependencies change
	$(DC) build api

logs:
	$(DC) logs -f

logs-agent:
	$(DC) logs -f agent-db

logs-demo:
	$(DC) logs -f demo-db

logs-api:  ## the reload log lives here — an edit to app/ restarts in place
	$(DC) logs -f api

# The trace stack is opt-in, like the MySQL container: six more services and
# ~2GB of RAM, and the agent answers questions identically without it. Enabling
# it is these targets plus two uncommented keys in .env — see .env.example.

langfuse-up:  ## start the trace stack, UI on :3000 (six containers, ~2GB)
	$(DC) --profile langfuse up -d --wait
	@echo "Langfuse at http://localhost:3000 — dev@sql-agent.local / sql-agent-dev"

langfuse-down:  ## stop it. The volumes survive, so traces do too.
	$(DC) --profile langfuse stop langfuse-web langfuse-worker \
	  langfuse-db langfuse-clickhouse langfuse-redis langfuse-minio

langfuse-logs:  ## why a trace never showed up
	$(DC) logs -f langfuse-web langfuse-worker

health:  ## is the API up? the CLI needs it to be
	@curl -fsS $(API)/health || { \
	  echo "no API at $(API) — start it with 'make up'"; exit 1; }
	@echo

psql-agent:  ## a shell on what the agent has learned
	$(DC) exec agent-db psql -U agent -d agent

psql-demo:  ## a shell on the data it answers questions about
	$(DC) exec demo-db psql -U business -d business

migrate:  ## apply migrations/*.sql to the agent database, in filename order
	@shopt -s nullglob; \
	files=(migrations/*.sql); \
	if [ $${#files[@]} -eq 0 ]; then echo "no migrations yet"; exit 0; fi; \
	for f in "$${files[@]}"; do \
	  echo "==> $$f"; \
	  $(PSQL_AGENT) < "$$f"; \
	done
	@echo "migrations applied"

seed:  ## build the demo database: role, schema, and 2,000 customers
	@$(PSQL_DEMO) < demo/demo.sql
	@echo "seed complete"

reset:  ## wipe everything the agent has learned, and reseed — the stage button
	uv run sql-agent reset --yes
	$(MAKE) seed

test:
	uv run pytest -q

test-live:  ## includes tests that call the Anthropic API and cost tokens
	uv run pytest -q -m live

# --- GEPA: searching the text a human wrote once -----------------------------
#
# One make target per searchable thing. `make gepa-extract` is a node's prompt;
# `make gepa-tools` is the four tool descriptions in app/tools.py and
# `make gepa-config` is the per-node effort block in config/config.yaml, both
# scored by running whole cold turns against a corpus of scored turns. Anything
# else says why it is not searchable rather than failing. `gepa` is a
# dependency group, so none of this is in the image.
#
# Every target harvests its corpus from Langfuse, which is the flywheel: the
# agent's own turns, read back. `extract` reads its recorded calls; `tools`
# and `config` read whole turns whose trace carries a reference query or a
# verdict that the query they ran was right. `make corpus` primes that with
# the answer key's nineteen questions; after that it grows from use, one
# scored turn at a time. `--days` is the window and `--resume` reuses the
# last harvest, for all three.
#
#   make gepa-tools                                    read it
#   make gepa-tools > new.md                           keep it
#   make gepa-config GEPA_ARGS=--probe-only            the seed over all 19,
#                                                      and where its tokens went
#   make gepa-extract GEPA_ARGS=--probe-only           check the invariants
#   make gepa-extract GEPA_ARGS='--pareto demo/gepa/extract.pareto.json'
#   make gepa-extract GEPA_ARGS='--iterations 1 --resume'  one step, then read
#                                                      the transcript
#   make gepa-extract-overfit GEPA_ARGS=--resume       guards off, on purpose:
#                                                      three cases, the gate
#                                                      reporting only, then
#                                                      the held-out score
#   make gepa-extract-pareto                           the front from the last
#   make gepa-tools-pareto                             run of that target, as
#   make gepa-config-pareto                            a table
#   make gepa-extract-overfit-pareto                   the overfit run's, with
#                                                      its held-out table
#   make gepa-extract-reset                            delete what the last run
#   make gepa-extract-overfit-reset                    of that target left, so
#   make gepa-tools-reset                              the next starts from
#   make gepa-config-reset                             nothing
#   make reset-gepa                                    all four of those
#
# Every run also leaves its whole terminal output in tools/gepa/out/<target>.run.txt:
# the harvest, the split, the gate's verdicts, the diff. Copy that and the
# front to demo/gepa/ to keep them; nothing under out/ survives a clone.
#
# `make gepa-tools` and `make gepa-config` cost real money and ask before they
# spend any. Redirecting means nobody is there to agree, so they refuse unless
# GEPA_ARGS=--yes. What every reflection read is kept beside the run, in
# tools/gepa/out/run/<target>/reflections.jsonl.
#
# `@` because make echoes recipes to stdout, and stdout is the artifact.
# Not `.PHONY`: make skips pattern rules for phony targets, and nothing named
# `gepa-*` is ever a file.
GEPA = uv run --group gepa python -m tools.gepa

# All three above `gepa-%`, and they have to be: this make (3.81) takes the
# first pattern rule that matches, and `gepa-%` matches `gepa-extract-pareto`
# too.
#
# `--overfit` is a flag on the extract search, not a target the CLI knows, so
# this is the one name here that `gepa-%` cannot pass through. It exists so
# the run has a command to be named by, in the talk and in the line below.
gepa-extract-overfit:  ## the extract search with the guards off: three cases, gate reporting only
	@$(GEPA) extract --overfit 3 $(GEPA_ARGS)

# The last run's front, from tools/gepa/out/ — every search writes one there
# whether or not --pareto asked for a tracked copy. The overfit run is its
# own name, so `make gepa-extract-overfit-pareto` is section 6's and this is
# section 3's. A committed copy is a path to the module:
#   uv run --group gepa python -m tools.gepa.front demo/gepa/extract.pareto.json
#
# Before any run, the file is not there. The person who typed this wants to
# know what to type next, not where the file would have been.
gepa-%-pareto:  ## the Pareto front from the last run of gepa-<target>, as a table
	@test -f tools/gepa/out/$*.pareto.json || { \
	  echo "No front for $* yet. Run 'make gepa-$*' first."; exit 1; }
	@uv run --group gepa python -m tools.gepa.front tools/gepa/out/$*.pareto.json

# Everything cli.py writes for one run name: the corpus, the front, the
# transcript, and GEPA's run dir. The overfit run has no corpus of its own —
# it reads extract's — so its reset leaves that alone by construction.
gepa-%-reset:  ## delete what the last run of gepa-<target> left in tools/gepa/out/
	rm -rf tools/gepa/out/$*.jsonl tools/gepa/out/$*.pareto.json \
	       tools/gepa/out/$*.run.txt tools/gepa/out/run/$*

# Phony, unlike the gepa-* rules: it is not a pattern, so the caveat above
# does not apply, and `reset-gepa` does not match `gepa-%`.
reset-gepa: gepa-extract-reset gepa-extract-overfit-reset gepa-tools-reset gepa-config-reset  ## all four

gepa-%:  ## GEPA over one searchable thing: new text on stdout, progress on stderr
	@$(GEPA) $* $(GEPA_ARGS)

# --- demo: presentation & recording -----------------------------------------

customer-count:  ## ask the cold-path question and print the token cost
	uv run sql-agent "how many customers do we have?"

west-coast-customer-count:  ## ask a new question the cache can compose an answer to
	uv run sql-agent "how many customers do we have in the west region?"

cache:  ## show what the agent has learned, as the model sees it
	@uv run sql-agent cache

turns:  ## tokens per turn — the demo chart, as a table
	@uv run sql-agent turns

config:  ## what the server is running — config.yaml under config.local.yaml
	@uv run sql-agent config

# --- the corpus: questions, asked cold, certified against a known answer ----
#
# The searches downstream need turns with a reference query on them. For the
# demo's questions that is written down: `demo/golden/` holds the query each
# answer should come from, and where the fixture fixes the number, the number.
# This asks each question cold, files the reference on the turn's trace, and
# where it can, a verdict too. Nobody is at the keyboard, so it can be left
# running. Everything lands on the traces, so Langfuse has to be up
# (`make langfuse-up`) and the server restarted with both keys.
#
# All nineteen get a reference; nine also get a verdict. The other ten have no
# written answer because theirs moves — revenue and date windows — and the
# reference alone makes each of them a case.
#
# This primes the pump. After it, the corpus is whatever anyone asked and
# scored: a verdict or a corrected query filed in the Langfuse UI is read by
# the next `make gepa-tools` the same way.
#
# Every question is asked with the memory off, so a run reads nothing the demo
# taught the agent and writes nothing back. The turns still land in the turn log
# — which is why `demo-verify` reads the five most recent rather than all of
# them.
corpus:  ## ask demo/golden cold, file the reference and, where known, the verdict (~19 model turns)
	uv run sql-agent corpus demo/golden

demo: health reset  ## record the terminal demo — live, 20-30 min of real model time
	$(VHS) demo/demo.tape

# T1-T3 are gated on the numbers because demo.sql derives them from modular
# arithmetic and 1,840 is a fact. T4 and T5 are gated on *shape* only, and that
# is not laziness: `orders.created` is anchored to now(), so which quarters exist
# and how many orders each holds slide with the recording date. A gate on "2024
# Q3 — 408" would pass today and fail in November, which is the worst kind of
# check — one that reports a bad take when nothing is wrong.
#
# The five *most recent* finished turns, not every finished turn. There is one
# turn log now, so a `make corpus` run earlier in the day sits in the same table
# and would otherwise be counted as the take.
LAST_FIVE = SELECT * FROM turn WHERE answer IS NOT NULL ORDER BY id DESC LIMIT 5

demo-verify:  ## did the last take earn its place? read it from the turn table
	@echo "=== turns ==="
	@$(PSQL_AGENT) -P pager=off -c "SELECT id, left(question, 38) AS question, \
	  explored, tokens_in + tokens_out AS tokens, answer \
	  FROM ($(LAST_FIVE)) r ORDER BY id"
	@echo "=== gate ==="
	@out=$$($(PSQL_AGENT) -P pager=off -t -A -c \
	  "WITH t AS ( \
	     SELECT row_number() OVER (ORDER BY id) AS n, explored, \
	            tokens_in + tokens_out AS tok, answer \
	     FROM ($(LAST_FIVE)) r) \
	   SELECT CASE WHEN ok THEN 'PASS  ' ELSE 'FAIL  ' END || label FROM ( \
	     SELECT 1 AS i, (SELECT count(*) FROM t) = 5 AS ok, \
	            'five turns recorded' AS label \
	     UNION ALL SELECT 2, coalesce((SELECT explored AND answer ~ '1,?840' \
	            FROM t WHERE n = 1), false), 'T1 explored, and answered 1,840' \
	     UNION ALL SELECT 3, coalesce((SELECT NOT explored AND answer ~ '1,?840' \
	            FROM t WHERE n = 2), false), 'T2 used the cache, same answer' \
	     UNION ALL SELECT 4, coalesce(((SELECT tok FROM t WHERE n = 2) \
	            < (SELECT tok FROM t WHERE n = 1)), false), 'T2 cost less than T1' \
	     UNION ALL SELECT 5, coalesce((SELECT answer ~ '460' FROM t WHERE n = 3), \
	            false), 'T3 answered 460' \
	     UNION ALL SELECT 6, coalesce((SELECT explored FROM t WHERE n = 4), false), \
	            'T4 explored — orders is a new area of the schema' \
	     UNION ALL SELECT 7, coalesce((SELECT NOT explored FROM t WHERE n = 5), \
	            false), 'T5 projected without exploring' \
	     UNION ALL SELECT 8, coalesce(((SELECT tok FROM t WHERE n = 5) \
	            < (SELECT tok FROM t WHERE n = 4)), false), 'T5 cost less than T4' \
	   ) x ORDER BY i"); \
	echo "$$out"; \
	if echo "$$out" | grep -q '^FAIL'; then \
	  echo; echo "bad take — re-record with 'make demo'"; exit 1; \
	else echo; echo "good take"; fi
