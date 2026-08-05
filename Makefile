.PHONY: up down dev migrate seed reset test psql logs

SHELL := /bin/bash
DC ?= docker compose
PSQL = $(DC) exec -T db psql -U flyline -d flyline -v ON_ERROR_STOP=1

up:  ## start postgres and wait for it to accept connections
	$(DC) up -d --wait

down:
	$(DC) down

logs:
	$(DC) logs -f db

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

t1:  ## ask the cold-path question and print the token cost
	uv run python -m scripts.ask "how many customers do we have?"

cache:  ## show what the agent has learned, as the model sees it
	@uv run python -m scripts.cache

turns:  ## tokens per turn — the demo chart, as a table
	@$(PSQL) -P pager=off -c "SELECT id, left(question, 38) AS question, \
	  explored, tool_calls AS tools, cache_entries AS cached, \
	  tokens_in + tokens_out AS tokens, latency_ms / 1000 AS secs \
	  FROM turn WHERE answer IS NOT NULL ORDER BY id"

psql:
	$(DC) exec db psql -U flyline -d flyline
