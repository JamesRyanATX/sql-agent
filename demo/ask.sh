#!/usr/bin/env bash
# Run one turn for the recording, and leave a card the tape can wait on.
#
# Why this exists, rather than the tape typing `scripts.ask` directly:
#
# VHS 0.11.0 can only see the first screenful of the terminal. Once output
# scrolls past it, both `Wait+Screen` and `Wait+Line` freeze on stale content
# and eventually time out — `Wait+Line` reports the initial `$` prompt forever.
# A T1 turn prints well past one screen, so waiting on its `[explored]` line
# times out even though the turn finished minutes earlier. (Measured: a turn
# that completed in 297s hit a 30-minute wait timeout.)
#
# A `clear` resets what VHS can see. So: stream the turn live — the scrolling
# exploration is the part worth watching — then clear and re-print the tail.
# The tape waits on that card, which is always within the first screenful.
set -uo pipefail

question="$1"
log=$(mktemp)
trap 'rm -f "$log"' EXIT

uv run python -m scripts.ask "$question" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

# Let the last streamed lines sit on screen for a moment before wiping them.
sleep 4
clear

# The answer and what it cost. Keeps the ANSI from ask.py, so the card is
# styled the same as the stream it replaces.
tail -n 6 "$log"

exit "$status"
