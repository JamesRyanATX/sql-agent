# The terminal demo — §9 steps 1-4, recorded live.
#
#   make demo          record it (17-25 min of real model time)
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
# on screen. Every step here stays short (`sql-agent` prints 4 lines by
# default; `sql-agent turns` is waited on by its psql-shaped `(N rows)`
# footer, which prints last), so this doesn't currently bite — but a step that
# printed pages of output would need to clear and re-print a short tail card
# instead of waiting directly.
#
# The four-line default is a contract, not a coincidence: `sql-agent -v` shows
# planning, exploration and what was learned, and any of it printed by default
# would scroll the awaited line out of the screenful VHS can see.

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
