# CLAUDE.md

## Overview
A LangGraph agent that answers business questions on a Postgres DB, caching learned facts as plain‑English entries. Token usage drops after caching.

## Commands
```bash
make up && make migrate && make seed   # start API, apply DB schema, load demo data
make test                              # run pytest (no live model calls)
make test-live                         # include live model tests
make cache                             # `sql-agent cache -c default`
make psql-agent / psql-demo            # shell to the DBs
make reset                             # clear learned state, reseed
```

## Configuration
- `app/settings.py`: reads environment + `.env` (secret/addresses).  
- `app/config.py`: reads `config/config.yaml` (behaviour) merged with `config/config.local.yaml` (git‑ignored).  
  - `config.yaml` provides model, per‑node effort, tool limits, etc.  
  - `extra="forbid"` enforces strict keys.

## Architecture
- **API** (`/v1/*`): only entry point; all logic lives in the server process.  
- `cli/sql_agent/`: HTTP client, no imports from `app`.  
- **Graph** (`app/graph.py`): nodes, schemas, edges; only `llm.complete` talks to models.  
- **Store** (`app/store.py`): connection registry, cache, turn log. Functions prefixed with `reflect_`, `schema_fingerprint`, etc. operate on the **target** DB; everything else uses the **agent** DB.  
- **DB** (`app/db.py`): `db.agent()`, `db.target(cid)`, `db.target_readonly(cid)`. Target engines use `AUTOCOMMIT`; read‑only wraps a transaction for Postgres/MySQL.  
- **Tools** (`app/tools.py`): four read‑only introspection tools used by the explore loop.

## Key Invariants
- `plan` writes SQL to the cache entry in the same call.  
- Cold cache skips the `plan` model call.  
- `extract` runs only on turns needing a fix; fully cached turns skip it.  
- Recipes are verified: `grounded_in()` checks the SQL fragment is a token subsequence of the executed SQL.  
- Pinned cache entries are never overwritten by extraction.  
- Named entries are upserted per‑connection; the unique index is `(connection_id, name)`.  
- All cache entries (including tombstones) load on every turn for the connection.  
- Turn state includes `connection_id`; a resumed turn cannot switch warehouses.  
- Per‑node `effort` controls model usage; there is no global override.  
- Dialect capabilities (read‑only, DML/DDL blocks, statement timeout) live in `app/dialects.py`.  
- SQL is executed via `exec_driver_sql`, never `text()`.  
- Identifiers are quoted using `quoted_name(.., quote=True)`.  
- Target engines run with `AUTOCOMMIT`; read‑only uses transactional settings per dialect.

## Traps (demo data)
The demo (`demo/demo.sql`) encodes five intentional pitfalls; tests in `tests/test_traps.py` assert they remain:
1. Soft‑deleted rows (`customer.deleted_at`).  
2. Varied case in `customer.region`.  
3. Cancelled orders (`orders.status`).  
4. Historical vs current price (`order_item.price` vs `product.unit_price`).  
5. `orders.created` column used instead of `created_at`.

## Testing
- `make test` runs all unit tests (no live model calls).  
- `make test-live` includes tests that hit the model.  
- DB fixtures create isolated `agent_test` and `business_test` schemas per session.  
- Read‑only role defined in `demo/demo.sql` enforces capability checks.

## Observability
Langfuse tracing is enabled only when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. No partial enablement. Traces capture system prompts, model generations, SQL statements, and row counts.

## Prompt Management
Prompts live in `config/prompts/*.md`. Each file is the full prompt for a graph node; loading is memoised per process. Missing or empty prompts raise errors. The loader (`app/prompts.py`) reads from `$CONFIG_DIR/prompts/<node>.md`.

## Miscellaneous
- The agent’s memory resides in a separate Postgres server (`agent-db`).  
- Target connections (`target-db`) may be Postgres, MySQL, or SQLite.  
- Secrets (`Connection.password`) are sealed via `app/secrets.py`.  
- The API enforces `Authorization: Bearer $API_TOKEN`; an empty token disables auth with a warning.

> Keep this document in sync with code changes; it is the source of truth for contributors.
