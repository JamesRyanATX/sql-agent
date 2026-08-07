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
# on screen. Every step here stays short (scripts.ask prints 4 lines; `make
# cache`/`make turns` are waited on by text that always prints first), so
# this doesn't currently bite — but a step that prints pages of output would
# need to clear and re-print a short tail card instead of waiting directly.

Output demo/demo.gif
Output demo/demo.mp4

Require make
Require uv

Set Shell bash
Set Theme "Catppuccin Mocha"
Set Width 1200
Set Height 700
Set FontSize 16
Set Padding 20
Set Framerate 10
Set PlaybackSpeed 10
Set WaitTimeout 40m

# A bare prompt, so nothing in the frame depends on whose machine this is.
Hide
Type "PS1='$ '"
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
Sleep 20s
Type 'uv run python -m scripts.ask "How many customers do we have?"'
Enter
Wait+Screen@40m /\[explored\]/
Sleep 40s

# ---------------------------------------------------------------- step 2 ----
# What it kept. This is the semantic layer, written by exploration.

Ctrl+L
Type "# What did we learn about the database?"
Enter
Sleep 15s
Type "make cache"
Enter
Wait+Screen@2m /entries \(/
Sleep 45s

# ---------------------------------------------------------------- step 3 ----
# Same question, same answer, several times cheaper — and no exploration.
# Typed in full rather than as a make alias, so it is visible that this is
# the same code path as step 1, not a replay.

Ctrl+L
Type "# The same question should now be cheaper:"
Enter
Sleep 20s
Type 'uv run python -m scripts.ask "How many customers do we have?"'
Enter
Wait+Screen@40m /\[no exploration\]/
Sleep 40s

# ---------------------------------------------------------------- step 4 ----
# A question never asked before, answered from what earlier turns learned.
# This is the part that is not memoisation.

Ctrl+L
Type "# Nuanced questions should require less exploration:"
Enter
Sleep 20s
Type 'uv run python -m scripts.ask "How many customers are in the west region?"'
Enter
Wait+Screen@40m /tokens \(/
Sleep 40s

# ------------------------------------------------------------------ close ----
# The curve, as a table.

Ctrl+L
Type "# tokens per turn:"
Enter
Sleep 15s
Type "make turns"
Enter
Wait+Screen@2m /rows\)/
Sleep 50s
