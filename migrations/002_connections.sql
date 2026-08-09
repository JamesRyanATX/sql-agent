-- Named connections, and per-connection scoping of everything the agent learns.
--
-- The agent used to point at exactly one business database, named once in
-- TARGET_DATABASE_URL. It now holds a registry of them, and the cache belongs
-- to a connection rather than to the process: `revenue` means something
-- different on every warehouse, and one global cache would let two of them
-- overwrite each other silently.
--
-- Idempotent: `make migrate` re-applies every file on every run. Both of
-- cache_entry's indexes are defined here rather than in 001, because both of
-- them gained a connection_id — see the note where they used to live.


CREATE TABLE IF NOT EXISTS connection (
  id          text PRIMARY KEY
                CHECK (id ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
  label       text,
  -- 'api' — registered over HTTP; the columns below are the whole address.
  -- 'env' — the built-in target. Its address is TARGET_DATABASE_URL, resolved
  --         at connect time; the columns here are a copy the app writes back
  --         so `make psql-agent` shows something true. See app/db.py.
  origin      text NOT NULL DEFAULT 'api' CHECK (origin IN ('api', 'env')),
  host        text,
  port        int CHECK (port BETWEEN 1 AND 65535),
  database    text,
  username    text,
  -- Tagged: 'plain:<password>' or 'fernet:<token>' (app/secrets.py). Never
  -- leaves this server — ConnectionOut has no password field at all.
  password    text,
  sslmode     text NOT NULL DEFAULT 'prefer',
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now(),
  -- An api-registered row must be a complete address. An env row need not be:
  -- its address is the environment's, not this table's.
  CONSTRAINT connection_address_complete CHECK (
    origin = 'env' OR (host IS NOT NULL AND port IS NOT NULL
                       AND database IS NOT NULL AND username IS NOT NULL)
  )
);

-- The connection the agent has always had, before it had a name. Inserted here
-- rather than at startup because the foreign keys below need something to point
-- at, and because `make up` starts the API *before* `make migrate` runs — a row
-- created only in the lifespan would not exist yet.
--
-- Its address is deliberately NULL. This file is applied by psql, and by
-- psycopg in tests/conftest.py, neither of which can see TARGET_DATABASE_URL;
-- hardcoding demo-db here would also put the demo's topology into the agent's
-- own schema, which is the split demo/demo.sql exists to keep.
INSERT INTO connection (id, label, origin)
VALUES ('default', 'TARGET_DATABASE_URL', 'env')
ON CONFLICT (id) DO NOTHING;


-- ---------------------------------------------------------------- cache_entry

ALTER TABLE cache_entry ADD COLUMN IF NOT EXISTS connection_id text;
-- Everything learned before the registry existed was learned against the one
-- target, which is now called 'default'.
UPDATE cache_entry SET connection_id = 'default' WHERE connection_id IS NULL;
ALTER TABLE cache_entry ALTER COLUMN connection_id SET NOT NULL;

-- There is no ADD CONSTRAINT IF NOT EXISTS; this is the idiom demo/demo.sql
-- already uses for the same problem.
DO $$
BEGIN
  ALTER TABLE cache_entry ADD CONSTRAINT cache_entry_connection_fk
    FOREIGN KEY (connection_id) REFERENCES connection(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END
$$;

DROP INDEX IF EXISTS cache_entry_load_idx;
DROP INDEX IF EXISTS cache_entry_name_key;

-- load_cache() reads one connection's entries, not disabled, ordered by hits.
CREATE INDEX IF NOT EXISTS cache_entry_conn_load_idx
  ON cache_entry (connection_id, disabled, hits DESC);

-- A named concept is one entry *per connection*, not one per turn that
-- re-learned it. Partial, because schema facts are frequently unnamed.
CREATE UNIQUE INDEX IF NOT EXISTS cache_entry_conn_name_key
  ON cache_entry (connection_id, name) WHERE name IS NOT NULL;


-- ----------------------------------------------------------------------- turn

ALTER TABLE turn ADD COLUMN IF NOT EXISTS connection_id text;
UPDATE turn SET connection_id = 'default' WHERE connection_id IS NULL;
ALTER TABLE turn ALTER COLUMN connection_id SET NOT NULL;

DO $$
BEGIN
  ALTER TABLE turn ADD CONSTRAINT turn_connection_fk
    FOREIGN KEY (connection_id) REFERENCES connection(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END
$$;

-- The turn log, newest first.
CREATE INDEX IF NOT EXISTS turn_connection_idx
  ON turn (connection_id, id DESC);

-- A scoped reset has to reach LangGraph's checkpoints, which are keyed by
-- thread_id — the session UUID — and know nothing about connections. This table
-- records both, so it is the mapping.
CREATE INDEX IF NOT EXISTS turn_conn_session_idx
  ON turn (connection_id, session_id);
