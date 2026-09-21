-- One memory, one target database. Undoes 002 and most of 003.
--
-- The agent learns about the database `TARGET_DATABASE_URL` names, and what it
-- learns lives in one place. 002 added a registry of databases and partitioned
-- `cache_entry` and `turn` by it; that idea is gone, so the partition key goes
-- with it.
--
-- 002 and 003 stay on disk. They are idempotent and they are history: every
-- `make migrate` replays them and then this file undoes them, which costs a
-- second and keeps the record of what was tried.
--
-- Idempotent like the rest, so the order within the file matters: the foreign
-- keys must go before the table they point at, and the indexes before the
-- column they are built on.

ALTER TABLE cache_entry DROP CONSTRAINT IF EXISTS cache_entry_connection_fk;
ALTER TABLE turn DROP CONSTRAINT IF EXISTS turn_connection_fk;

DROP INDEX IF EXISTS cache_entry_conn_load_idx;
DROP INDEX IF EXISTS cache_entry_conn_name_key;
DROP INDEX IF EXISTS turn_connection_idx;
DROP INDEX IF EXISTS turn_conn_session_idx;

ALTER TABLE cache_entry DROP COLUMN IF EXISTS connection_id;
ALTER TABLE turn DROP COLUMN IF EXISTS connection_id;

DROP TABLE IF EXISTS connection;

-- Two databases were allowed to learn the same name, and now there is one
-- memory, so those rows collide. Keep the most-used of each name and drop the
-- rest — without this the unique index below fails and the whole migration
-- rolls back under ON_ERROR_STOP, on a machine that had used two connections.
-- Ties break on the newest id, which is the entry most recently learned.
DELETE FROM cache_entry a
USING cache_entry b
WHERE a.name IS NOT NULL
  AND a.name = b.name
  AND (a.hits, a.id) < (b.hits, b.id);

-- The two indexes 001 describes and does not create, because 002 owned them
-- once they gained a connection_id. They come back here rather than there: 001
-- is re-applied on every run, and recreating the unique name index *before*
-- this file drops the partitioned one would fail on the first duplicate name.
--
-- The unique index is what makes `write_entries` an upsert: learning more about
-- `revenue` refines the entry rather than filing a second one. Partial, because
-- a schema fact has no name and there may be many of those.
CREATE INDEX IF NOT EXISTS cache_entry_load_idx
  ON cache_entry (disabled, hits DESC);

CREATE UNIQUE INDEX IF NOT EXISTS cache_entry_name_key
  ON cache_entry (name) WHERE name IS NOT NULL;

-- 002's turn index, minus the partition. `sql-agent turns` reads newest first.
CREATE INDEX IF NOT EXISTS turn_recent_idx ON turn (id DESC);
