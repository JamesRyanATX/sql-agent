-- What a turn charged, in dollars, where the backend says so.
--
-- NULL rather than 0 when nothing reported it: Anthropic-direct and every local
-- endpoint bill somewhere else or not at all, and a turn on a free model really
-- does cost nothing. Those two must not look alike in the chart.
--
-- numeric, not double precision: these are fractions of a cent added up over
-- hundreds of rollouts, and binary floating point loses the pennies.
ALTER TABLE turn ADD COLUMN IF NOT EXISTS cost numeric(12, 6);
