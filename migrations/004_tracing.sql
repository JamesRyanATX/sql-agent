-- Which Langfuse trace a turn was recorded as.
ALTER TABLE turn ADD COLUMN IF NOT EXISTS trace_id text;
