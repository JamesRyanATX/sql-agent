# The terminal demo — §9 steps 1-5, recorded live.
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
# Two things about VHS 0.11.0 shape this file:
#
# 1. It can only see the FIRST SCREENFUL. Once output scrolls past it, `Wait`
#    freezes on stale content and times out regardless of what the terminal is
#    actually showing. The three turns print well past one screen, so they run
#    through demo/ask.sh, which streams the turn and then clears and re-prints
#    the tail as a card. The wait matches the card. See that file for detail.
#
# 2. `Wait+Screen` matches whatever is currently visible, so a leftover
#    "tokens (" line from an earlier step would satisfy the next wait instantly
#    and the tape would type into a still-running turn. Hence Ctrl+L per step.
#
# The short steps (reset, cache, turns) print inside one screen and are waited
# on directly.

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
# An empty cache, wiped on camera. Nothing up the sleeve.

Type "# the agent has never seen this database. start from nothing:"
Enter
Sleep 20s
Type "make reset"
Enter
Wait+Screen@10m /seed complete/
Sleep 20s

Ctrl+L
Type "# what does it know?"
Enter
Sleep 15s
Type "make cache"
Enter
Wait+Screen@2m /cache is empty/
Sleep 25s

# ---------------------------------------------------------------- step 2 ----
# The expensive question. It explores: lists tables, describes columns,
# samples values, and finds the soft-delete trap.

Ctrl+L
Type "# first question. it has to go and look:"
Enter
Sleep 20s
Type 'bash demo/ask.sh "how many customers do we have?"'
Enter
Wait+Screen@40m /\[explored\]/
Sleep 40s

# ---------------------------------------------------------------- step 3 ----
# What it kept. This is the semantic layer, written by exploration.

Ctrl+L
Type "# it kept what it learned:"
Enter
Sleep 15s
Type "make cache"
Enter
Wait+Screen@2m /entries \(/
Sleep 45s

# ---------------------------------------------------------------- step 4 ----
# Same question, same answer, several times cheaper — and no exploration.
# Typed in full rather than as a make alias, so it is visible that this is
# the same code path as step 2, not a replay.

Ctrl+L
Type "# ask the exact same question again:"
Enter
Sleep 20s
Type 'bash demo/ask.sh "how many customers do we have?"'
Enter
Wait+Screen@40m /\[no exploration\]/
Sleep 40s

# ---------------------------------------------------------------- step 5 ----
# A question never asked before, answered from what earlier turns learned.
# This is the part that is not memoisation.

Ctrl+L
Type "# now something it has never been asked:"
Enter
Sleep 20s
Type 'bash demo/ask.sh "how many customers are in the west region?"'
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
