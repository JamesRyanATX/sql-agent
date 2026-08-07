-- The cache and the turn log (PLAN.md §3.1).
--
-- Idempotent: `make migrate` re-applies every file on every run, so everything
-- here is IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS cache_entry (
  id             bigserial PRIMARY KEY,
  kind           text NOT NULL CHECK (kind IN ('schema_fact', 'recipe')),
  name           text,                 -- 'revenue', 'active customer'
  claim          text NOT NULL,        -- the prose the model reads
  sql_fragment   text,                 -- recipes only
  tables         text[] NOT NULL,      -- for drift invalidation
  origin         text NOT NULL CHECK (origin IN ('learned', 'human')),
  pinned         boolean NOT NULL DEFAULT false,
  disabled       boolean NOT NULL DEFAULT false,
  tombstone      boolean NOT NULL DEFAULT false,  -- a learned-wrong thing
  verified       boolean NOT NULL DEFAULT false,
  hits           int NOT NULL DEFAULT 0,
  schema_fp      text,                 -- fingerprint of tables[] at write time
  created_turn   bigint,
  -- Not in §3.1, but §5 compaction needs it: "drop unverified entries with
  -- hits = 0 and no use in 20 turns" is unanswerable from created_turn alone.
  last_used_turn bigint,
  created_at     timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now()
);

-- load_cache() reads everything not disabled, ordered by hits.
CREATE INDEX IF NOT EXISTS cache_entry_load_idx
  ON cache_entry (disabled, hits DESC);

-- A named concept is one entry, not one per turn that re-learned it.
-- Partial, because schema facts are frequently unnamed.
CREATE UNIQUE INDEX IF NOT EXISTS cache_entry_name_key
  ON cache_entry (name) WHERE name IS NOT NULL;

CREATE TABLE IF NOT EXISTS turn (
  id            bigserial PRIMARY KEY,
  session_id    uuid NOT NULL,
  question      text NOT NULL,
  sql           text,
  answer        text,
  tool_calls    int NOT NULL DEFAULT 0,
  explored      boolean NOT NULL DEFAULT false,
  tokens_in     int NOT NULL DEFAULT 0,
  tokens_out    int NOT NULL DEFAULT 0,
  latency_ms    int,
  cache_entries int NOT NULL DEFAULT 0,   -- how many loaded for this turn
  created_at    timestamptz NOT NULL DEFAULT now()
);

-- The demo chart: tokens per turn against created_at.
CREATE INDEX IF NOT EXISTS turn_session_idx
  ON turn (session_id, created_at);
