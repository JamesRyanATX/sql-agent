-- Targets are no longer all Postgres.
--
-- The address columns in 002 were the libpq connection model with the names
-- filed off, and three dialects do not share one: SQLite has a file path and no
-- host, port or user at all. So the row gains the driver that reads it, and the
-- completeness CHECK becomes driver-aware.
--
-- A column rather than a stored SQLAlchemy URL, deliberately. Today the
-- password is one isolated, sealed column and `ConnectionOut` has no field that
-- could carry it (002's note). A URL puts the password back *inside* the
-- address, where every SELECT of that column, every `**row`, every log line and
-- every partial PATCH has to remember to strip it again — and it would break
-- `update_connection`'s "an absent field is left alone" semantics, which are
-- per-column by construction.
--
-- Idempotent: `make migrate` re-applies every file on every run.


ALTER TABLE connection
  ADD COLUMN IF NOT EXISTS driver text NOT NULL DEFAULT 'postgresql+psycopg';

-- An allowlist in the schema, not only in Pydantic. `driver` is handed to
-- create_async_engine, which resolves a drivername by entry-point lookup — an
-- arbitrary string there is a plugin-loading surface, and a row edited by hand
-- at `make psql-agent` never sees a validator.
--
-- Only async-capable dialects. A sync driver (Snowflake, BigQuery) would need
-- every target call to cross into a thread pool, and a thread pool underneath an
-- asyncio graph is how a demo hangs on stage.
DO $$
BEGIN
  ALTER TABLE connection ADD CONSTRAINT connection_driver_known CHECK (
    driver IN ('postgresql+psycopg', 'mysql+asyncmy', 'sqlite+aiosqlite')
  );
EXCEPTION WHEN duplicate_object THEN NULL;
END
$$;

-- Driver-specific connect kwargs the address columns cannot express. **Not** a
-- second way to spell host/port/database: app/db.py filters this against a
-- per-driver allowlist before it reaches create_async_engine, so an unknown key
-- is dropped rather than passed through to a driver that might honour it.
ALTER TABLE connection
  ADD COLUMN IF NOT EXISTS options jsonb NOT NULL DEFAULT '{}'::jsonb;

-- 002's version required host, port, database and username together, which
-- rejects every SQLite row. Replaced rather than added to — and dropped by name
-- first, because 002 recreates it on the next `make migrate` and two CHECKs
-- with the same intent is how one of them silently stops being enforced.
ALTER TABLE connection DROP CONSTRAINT IF EXISTS connection_address_complete;

DO $$
BEGIN
  ALTER TABLE connection ADD CONSTRAINT connection_address_complete CHECK (
    origin = 'env'
    OR (driver = 'sqlite+aiosqlite' AND database IS NOT NULL)
    OR (driver <> 'sqlite+aiosqlite'
        AND host IS NOT NULL AND port IS NOT NULL
        AND database IS NOT NULL AND username IS NOT NULL)
  );
EXCEPTION WHEN duplicate_object THEN NULL;
END
$$;

-- Every fingerprint written before this was computed from information_schema
-- strings — 'character varying', 'YES'. They are now computed from the
-- dialect's own reflection, which spells the same column 'VARCHAR(64)' and a
-- boolean. The two never compare equal, so keeping the old values would report
-- every entry stale, which is a lie in the opposite direction from silence.
-- NULL means "cannot be checked", which is what is true.
UPDATE cache_entry SET schema_fp = NULL WHERE schema_fp IS NOT NULL;
