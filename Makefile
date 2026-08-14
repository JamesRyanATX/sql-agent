.PHONY: up down build migrate seed reset reset-all test test-live connections \
        psql-agent psql-demo logs logs-agent logs-demo logs-api health \
        customer-count west-coast-customer-count cache turns demo demo-verify \
        langfuse-up langfuse-down langfuse-logs

SHELL := /bin/bash
DC ?= docker compose

# SQLAlchemy's asyncio layer needs greenlet, whose manylinux wheel dlopens
# libstdc++. On NixOS that is not on the default search path, so `import
# sqlalchemy.ext.asyncio` fails with "the greenlet library is required" —
# nix-ld has already collected the library, it just is not where dlopen looks.
# A no-op anywhere NIX_LD_LIBRARY_PATH is unset, which is everywhere else.
ifdef NIX_LD_LIBRARY_PATH
export LD_LIBRARY_PATH := $(NIX_LD_LIBRARY_PATH)$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))
endif

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

# Everything the demo does runs against the built-in connection, whose address
# is TARGET_DATABASE_URL. A registered one would be `-c <id>`.
CONN ?= default

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
	@echo "uncomment LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env, then:"
	@echo "  $(DC) up -d api    # not 'restart' — that does not re-read .env"

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

reset:  ## wipe what the demo connection learned, and reseed — the stage button
	uv run sql-agent reset -c $(CONN) --yes
	$(MAKE) seed

reset-all:  ## every connection, plus the registry's turn log. Rarely what you want.
	@$(PSQL_AGENT) -c "TRUNCATE cache_entry, turn, checkpoints, checkpoint_blobs, \
	  checkpoint_writes RESTART IDENTITY CASCADE"
	@echo "all learned state wiped (the connection registry is untouched)"

test:
	uv run pytest -q

test-live:  ## includes tests that call the Anthropic API and cost tokens
	uv run pytest -q -m live

# --- prompts: evaluation and search -----------------------------------------
#
# `optim/` is a development tool, so `gepa` is a dependency group rather than a
# dependency — `uv sync --no-dev` already excludes groups, which is the same
# mechanism that keeps pytest out of the image. Nothing here runs in the
# container.
#
# `optim-apply` writes config/prompts/$(NODE).md and stops — it stages nothing
# and commits nothing. Promotion is the commit you make after reading the diff,
# and its message is where the reason goes: no run can produce that.

NODE ?= extract
OPTIM = uv run --group optim python -m optim.run

optim-probe:  ## do the current prompts still honour their invariants? (costs tokens)
	$(OPTIM) probe --node $(NODE)

optim-harvest:  ## pull recorded extract calls out of Langfuse into optim/out/
	$(OPTIM) harvest --conn $(CONN)

optim-run:  ## GEPA over one node's prompt, gated on the probes (costs tokens)
	$(OPTIM) optimize --node $(NODE)

optim-diff:  ## what the winner changed, beside the invariant checklist
	$(OPTIM) diff --node $(NODE)

optim-apply:  ## write the gated winner into config/prompts/$(NODE).md (nothing is committed)
	$(OPTIM) apply --node $(NODE)

# --- demo: presentation & recording -----------------------------------------

customer-count:  ## ask the cold-path question and print the token cost
	uv run sql-agent -c $(CONN) "how many customers do we have?"

west-coast-customer-count:  ## ask a new question the cache can compose an answer to
	uv run sql-agent -c $(CONN) "how many customers do we have in the west region?"

connections:  ## every database the agent can be pointed at
	@uv run sql-agent connections ls

cache:  ## show what the agent has learned, as the model sees it
	@uv run sql-agent cache -c $(CONN)

turns:  ## tokens per turn — the demo chart, as a table
	@uv run sql-agent turns -c $(CONN)

demo: health reset  ## record the terminal demo — live, 20-30 min of real model time
	$(VHS) demo/demo.tape

# T1-T3 are gated on the numbers because demo.sql derives them from modular
# arithmetic and 1,840 is a fact. T4 and T5 are gated on *shape* only, and that
# is not laziness: `orders.created` is anchored to now(), so which quarters exist
# and how many orders each holds slide with the recording date. A gate on "2024
# Q3 — 408" would pass today and fail in November, which is the worst kind of
# check — one that reports a bad take when nothing is wrong.
demo-verify:  ## did the last take earn its place? read it from the turn table
	@echo "=== turns ==="
	@$(PSQL_AGENT) -P pager=off -c "SELECT id, left(question, 38) AS question, \
	  explored, tokens_in + tokens_out AS tokens, answer \
	  FROM turn WHERE answer IS NOT NULL AND connection_id = '$(CONN)' \
	  ORDER BY id"
	@echo "=== gate ==="
	@out=$$($(PSQL_AGENT) -P pager=off -t -A -c \
	  "WITH t AS ( \
	     SELECT row_number() OVER (ORDER BY id) AS n, explored, \
	            tokens_in + tokens_out AS tok, answer \
	     FROM turn WHERE answer IS NOT NULL AND connection_id = '$(CONN)') \
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
