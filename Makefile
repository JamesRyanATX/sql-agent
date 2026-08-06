.PHONY: up down dev migrate seed reset test psql logs demo demo-verify

SHELL := /bin/bash
DC ?= docker compose
PSQL = $(DC) exec -T db psql -U sql_agent -d sql_agent -v ON_ERROR_STOP=1

# VHS needs ttyd and ffmpeg on PATH. Prefer a system install; otherwise pull all
# three through nix so recording needs nothing installed permanently.
VHS ?= $(shell command -v vhs 2>/dev/null || \
         echo 'nix shell nixpkgs#vhs nixpkgs#ttyd nixpkgs#ffmpeg -c vhs')

# --- core: environment, migrations, tests ----------------------------------

up:  ## start postgres and wait for it to accept connections
	$(DC) up -d --wait

down:
	$(DC) down

logs:
	$(DC) logs -f db

psql:
	$(DC) exec db psql -U sql_agent -d sql_agent

dev:  ## run the API with reload
	uv run uvicorn app.main:app --reload --port 8000

migrate:  ## apply migrations/*.sql in filename order
	@shopt -s nullglob; \
	files=(migrations/*.sql); \
	if [ $${#files[@]} -eq 0 ]; then echo "no migrations yet"; exit 0; fi; \
	for f in "$${files[@]}"; do \
	  echo "==> $$f"; \
	  $(PSQL) < "$$f"; \
	done
	@echo "migrations applied"

seed:
	uv run python -m scripts.seed

reset:  ## wipe learned state and reseed — the stage recovery button
	uv run python -m scripts.reset
	$(MAKE) seed

test:
	uv run pytest -q

test-live:  ## includes tests that call the Anthropic API and cost tokens
	uv run pytest -q -m live

# --- demo: presentation & recording -----------------------------------------

t1:  ## ask the cold-path question and print the token cost
	uv run python -m scripts.ask "how many customers do we have?"

cache:  ## show what the agent has learned, as the model sees it
	@uv run python -m scripts.cache

turns:  ## tokens per turn — the demo chart, as a table
	@$(PSQL) -P pager=off -c "SELECT id, left(question, 38) AS question, \
	  explored, tool_calls AS tools, cache_entries AS cached, \
	  tokens_in + tokens_out AS tokens, latency_ms / 1000 AS secs \
	  FROM turn WHERE answer IS NOT NULL ORDER BY id"

demo:  ## record the terminal demo — live, 17-25 min of real model time
	$(VHS) demo/demo.tape

demo-verify:  ## did the last take earn its place? read it from the turn table
	@echo "=== turns ==="
	@$(PSQL) -P pager=off -c "SELECT id, left(question, 38) AS question, \
	  explored, tokens_in + tokens_out AS tokens, answer \
	  FROM turn WHERE answer IS NOT NULL ORDER BY id"
	@echo "=== gate ==="
	@out=$$($(PSQL) -P pager=off -t -A -c \
	  "WITH t AS ( \
	     SELECT row_number() OVER (ORDER BY id) AS n, explored, \
	            tokens_in + tokens_out AS tok, answer \
	     FROM turn WHERE answer IS NOT NULL) \
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
