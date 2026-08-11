# The terminal demo — PLAN.md §9 steps 1, 3-5 and 8, recorded live.
#
# §9's step 5 ("revenue by region last quarter — curve bumps, doesn't reset") is
# the one this tape went without for a long time, and it is the claim's honest
# half: the agent gets cheaper at what it has seen, not at everything. It lands
# here as two beats rather than one, because the bump and the payoff are
# separate things to watch — step 5 pays for `orders`, step 6 spends what it
# bought. `orders` rather than §9's `order_item` only because a quarterly count
# is a table you can read at a glance.
#
#   make demo          record it (20-30 min of real model time)
#   make demo-verify   check the take was a good one, from the turn table
#
# Nothing here is re-enacted: every token count and latency on screen comes
# from a real turn against a real database. That is why the tape waits on
# output rather than sleeping a guessed duration — turn length varies by
# minutes between runs, and a guess would either truncate a turn or pad the
# recording with dead air.
#
# The tape only records the demo itself — resetting and reseeding is prep,
# not part of the story, so `make demo` does that first (see the `demo`
# target). Question 1's exploration already tells the audience this is a
# cold start; there's no separate beat proving the cache is empty first.
#
# `Wait+Screen` matches whatever is currently visible, so a leftover
# "tokens (" line from an earlier step would satisfy the next wait instantly
# and the tape would type into a still-running turn. Hence Ctrl+L per step.
#
# VHS 0.11.0 can only see the FIRST SCREENFUL — once output scrolls past it,
# `Wait` freezes on stale content and times out regardless of what's actually
# on screen. At 1200x700 and 16pt that screenful is roughly 34 lines, and every
# step here clears first, so the budget is per step rather than cumulative.
#
# A turn prints the question, a psql-shaped result table with its `(N rows)`
# footer, the answer paragraph, and the token line the `Wait` below matches.
# Steps 1, 3 and 4 return a single cell, so they run ~11 lines. Step 5 is the
# long one: two years of `orders` is 9 or 10 quarters, which puts it around 22.
#
# **The ceiling is real.** `sql-agent` will print up to `max_rows` (50) rows,
# which is 55 lines and would scroll `tokens (` clean out of view. Width is
# bounded — `render.CELL` truncates a long value — but height is not. A step
# asking for something genuinely list-shaped (every product, every customer)
# would have to clear and re-print a short tail card rather than waiting on the
# turn's own output. Step 5 gets away with it because a quarter is a coarse
# bucket, and that is luck rather than design.
#
# The bounded default is a contract, not a coincidence: `sql-agent -v` shows
# planning, exploration and what was learned, and any of it printed by default
# would eat the margin the rows have.

Output demo/demo.gif
Output demo/demo.mp4

Require uv

Set Shell bash
Set Theme "Catppuccin Mocha"
Set Width 1200
Set Height 700
Set FontSize 16
Set Padding 20
Set Framerate 10
Set PlaybackSpeed 5
Set WaitTimeout 40m

# A bare prompt, so nothing in the frame depends on whose machine this is,
# and the venv's `sql-agent` on PATH so the commands on screen are the ones a
# user would type rather than `uv run` boilerplate.
#
# Single-quoted, because VHS's parser does not accept \" inside a "..." string —
# it lexes the escape as a bare command and fails with seven errors on one line.
# Nothing here needs SQL_AGENT_URL: it defaults to what `make up` starts.
Hide
Type "PS1='$ '"
Enter
Type 'export PATH="$PWD/.venv/bin:$PATH"'
Enter
Type "sql-agent connect default"
Enter
Type "clear"
Enter
Show
Sleep 5s

# ---------------------------------------------------------------- step 1 ----
# The expensive question. It explores: lists tables, describes columns,
# samples values, and finds the soft-delete trap.

Type "# The initial question implies schema discovery"
Enter
Type 'sql-agent "How many customers do we have?"'
Enter
Wait+Screen@40m /\[explored\]/
Sleep 20s

# ---------------------------------------------------------------- step 3 ----
# Same question, same answer, several times cheaper — and no exploration.
# Typed in full rather than as a make alias, so it is visible that this is
# the same code path as step 1, not a replay.

Ctrl+L
Type "# The same question should now be cheaper:"
Enter
Type 'sql-agent "How many customers do we have?"'
Enter
Wait+Screen@40m /\[no exploration\]/
Sleep 20s

# ---------------------------------------------------------------- step 4 ----
# A question never asked before, answered from what earlier turns learned.
# This is the part that is not memoisation.

Ctrl+L
Type "# Nuanced questions should require less exploration:"
Enter
Type 'sql-agent "How many customers are in the west region?"'
Enter
Wait+Screen@40m /tokens \(/
Sleep 20s

# ---------------------------------------------------------------- step 5 ----
# A different area of the schema, and the price goes back up. Steps 1-4 are all
# about `customer`; nothing the cache holds says anything about `orders`, so
# this explores again — which is the honest shape of the claim. The agent does
# not get cheaper at everything, it gets cheaper at what it has seen.
#
# It also walks into two of the five traps on the way: the date column is
# `created`, not `created_at`, and `status` includes cancelled rows.

Ctrl+L
Type "# A new area of the schema costs again:"
Enter
Type 'sql-agent "list the number of orders per quarter, sorted by date ascending"'
Enter
Wait+Screen@40m /\[explored\]/
Sleep 20s

# ---------------------------------------------------------------- step 6 ----
# The payoff, and the reason step 5 earns its place. Nothing cached is a
# forecast — what step 5 filed away is that orders are dated by `created`, that
# status includes cancellations, and how to bucket them into quarters. From
# those three facts the agent writes a least-squares fit with `regr_slope` and
# projects two quarters forward, without touching the schema.
#
# "the next two quarters", not named ones: demo.sql anchors `orders.created` to
# now(), so the data slides with the recording date and a named quarter would
# quietly become a question about the past.
#
# Measured: 1,416 tokens in 10s against step 5's 14,254 in 48s. The first run of
# this question tripped the fix loop on `interval '1 quarter'`; the cache
# absorbed that, which is the demo happening off-camera.

Ctrl+L
Type "# Composition, not lookup — nothing cached is a forecast:"
Enter
Type 'sql-agent "based on previous quarters, what is the projected order count for the next two quarters?"'
Enter
Wait+Screen@40m /\[no exploration\]/
Sleep 20s

# ------------------------------------------------------------------ close ----
# The curve, as a table.

Ctrl+L
Type "# Dump of token cost per turn:"
Enter
Type "sql-agent turns"
Enter
Wait+Screen@2m /rows\)/
Sleep 20s

# ------------------------------------------------------------ connections ----
# Where all of that was pointed, and how to point it somewhere else. Everything
# above belongs to one registered database — the cache, the turn log, the falling
# number — and `connect` is how you switch to another.
#
# Two commands on one screen: `connected to` cannot appear in the listing above
# it, so the second wait has no stale line to match early.

Ctrl+L
Type "# The agent is pointed at a registered database:"
Enter
Type "sql-agent connections ls"
Enter
# `(1 row)`, singular, while there is one connection — so match the stem, not
# the plural the turns table happens to print.
Wait+Screen@2m /row\)/
Sleep 5s
Type "sql-agent connect default"
Enter
Wait+Screen@2m /connected to/
Sleep 20s
