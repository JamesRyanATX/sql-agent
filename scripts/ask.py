"""Run one turn from the terminal and print what it cost.

    uv run python -m scripts.ask "how many customers do we have?"

Needs the API up (`make up`). The turn runs in the server — this is a renderer
for the event stream, so what you see here is what any client sees.

Output is just the question, the answer, and what it cost — no cache state,
tool-call trace, findings, SQL, or learned entries. That detail lives in the
turn table (`make turns`, `make demo-verify`) for anyone who needs it.

This is how the phase-3 gate gets checked: if T1 lands under ~3,000 tokens the
delta won't sell and the schema needs more decoys.
"""

import sys

from scripts import _client

BOLD, RESET = "\033[1m", "\033[0m"
YELLOW, WHITE, GREY = "\033[33m", "\033[97m", "\033[2m"


def out(*args) -> None:
    # Unbuffered: a turn takes minutes, and redirected to a file Python would
    # hold every line until the process exits — which looks exactly like a hang.
    print(*args, flush=True)


def fmt_rows(rows: list[dict]) -> str:
    # Compact tuple-of-values rendering, no column names.
    if not rows:
        return "[ ]"
    parts = [f"[ {', '.join(repr(v) for v in row.values())} ]" for row in rows]
    return f"[ {', '.join(parts)} ]"


def show(ev: dict) -> None:
    kind = ev.get("type")
    if kind == "rows":
        out(f"{BOLD}{WHITE}=> {fmt_rows(ev['rows'])}{RESET}")
    elif kind == "error":
        # A silent failure with no explanation is worse than the noise it costs.
        out(f"\033[31merror: {ev['message']}{RESET}")
    elif kind == "answer":
        summary = (
            f"{ev['total_tokens']:,} tokens "
            f"({ev['tokens_in']:,} in / {ev['tokens_out']:,} out) "
            f"in {ev['latency_ms'] / 1000:.1f}s"
            f"{'  [explored]' if ev['explored'] else '  [no exploration]'}"
        )
        out(f"\n{GREY}{summary}{RESET}")


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "how many customers do we have?"

    out(f"{BOLD}{YELLOW}{question}{RESET}")
    # No session_id: the server mints one per turn, which is what a one-shot
    # CLI question wants. Turns share the cache regardless — that is the point.
    async for ev in _client.stream_events("/v1/ask", {"question": question}):
        show(ev)


if __name__ == "__main__":
    _client.run(main())
