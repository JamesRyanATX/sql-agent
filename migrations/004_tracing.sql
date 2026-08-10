-- Which Langfuse trace a turn was recorded as.
--
-- A column and not a join: the trace lives in another system entirely, on the
-- other side of an HTTP exporter, and there is nothing here to join it to. What
-- this buys is the direction that matters — `make turns` shows a turn that cost
-- 11,500 tokens, and this is how you get from that row to the 24 tool calls and
-- six generations that spent them.
--
-- Nullable, and NULL is the common case: tracing is off unless both Langfuse
-- keys are set, so every turn taken before this migration and most turns after
-- it have no trace. NULL means "not recorded", which is what is true — the
-- alternative, an empty string, is a trace id that does not resolve.
--
-- Idempotent: `make migrate` re-applies every file on every run.

ALTER TABLE turn ADD COLUMN IF NOT EXISTS trace_id text;
