.PHONY: up down build migrate seed reset reset-all test test-live connections \
        psql-agent psql-demo logs logs-agent logs-demo logs-api health \
        customer-count west-coast-customer-count cache turns demo demo-verify

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

demo: health reset  ## record the terminal demo — live, 17-25 min of real model time
	$(VHS) demo/demo.tape

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
	     SELECT 1 AS i, (SELECT count(*) FROM t) = 3 AS ok, \
	            'three turns recorded' AS label \
	     UNION ALL SELECT 2, coalesce((SELECT explored AND answer ~ '1,?840' \
	            FROM t WHERE n = 1), false), 'T1 explored, and answered 1,840' \
	     UNION ALL SELECT 3, coalesce((SELECT NOT explored AND answer ~ '1,?840' \
	            FROM t WHERE n = 2), false), 'T2 used the cache, same answer' \
	     UNION ALL SELECT 4, coalesce(((SELECT tok FROM t WHERE n = 2) \
	            < (SELECT tok FROM t WHERE n = 1)), false), 'T2 cost less than T1' \
	     UNION ALL SELECT 5, coalesce((SELECT answer ~ '460' FROM t WHERE n = 3), \
	            false), 'T3 answered 460' \
	   ) x ORDER BY i"); \
	echo "$$out"; \
	if echo "$$out" | grep -q '^FAIL'; then \
	  echo; echo "bad take — re-record with 'make demo'"; exit 1; \
	else echo; echo "good take"; fi
