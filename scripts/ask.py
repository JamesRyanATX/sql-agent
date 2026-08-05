"""Run one turn from the terminal and print what it cost.

    uv run python -m scripts.ask "how many customers do we have?"

This is how the phase-3 gate gets checked: if T1 lands under ~3,000 tokens the
delta won't sell and the schema needs more decoys.
"""

import asyncio
import json
import sys
from uuid import uuid4

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app import db
from app.graph import build_graph, stream_turn

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def out(*args) -> None:
    # Unbuffered: a turn takes minutes, and redirected to a file Python would
    # hold every line until the process exits — which looks exactly like a hang.
    print(*args, flush=True)


def show(ev: dict) -> None:
    kind = ev.get("type")
    if kind == "plan":
        out(f"{DIM}cache: {ev['cache_entries']} entries{RESET}")
    elif kind == "explore":
        args = ", ".join(f"{k}={v!r}" for k, v in ev["input"].items())
        mark = "!" if ev["error"] else " "
        out(f"{DIM}  {ev['calls']:>2}{mark} {ev['tool']}({args}){RESET}")
    elif kind == "findings":
        out(f"\n{BOLD}findings{RESET} ({ev['tool_calls']} tool calls)")
        out(f"{DIM}{ev['text']}{RESET}\n")
    elif kind == "learned":
        out(f"\n{BOLD}learned{RESET} {ev['count']} entries"
            + (f" ({ev['skipped']} skipped)" if ev["skipped"] else ""))
        for e in ev["entries"]:
            tick = "\033[32m✓\033[0m" if e["verified"] else " "
            label = f"{e['name']}: " if e["name"] else ""
            out(f"  {tick} {DIM}[{e['kind']}]{RESET} {label}{e['claim']}")
    elif kind == "sql":
        out(f"{BOLD}sql{RESET}\n{ev['sql']}")
        for a in ev["assumptions"]:
            out(f"{DIM}  assumption: {a}{RESET}")
    elif kind == "rows":
        out(f"{DIM}-> {ev['count']} row(s): {json.dumps(ev['rows'])}{RESET}")
    elif kind == "error":
        out(f"\033[31merror: {ev['message']}{RESET}")
    elif kind == "fix":
        out(f"\033[33mfix {ev['attempt']}: {ev['was_wrong']}{RESET}\n{ev['sql']}")
    elif kind == "answer":
        out(f"\n{BOLD}{ev['text']}{RESET}")
        out(
            f"\n{BOLD}{ev['total_tokens']:,} tokens{RESET} "
            f"({ev['tokens_in']:,} in / {ev['tokens_out']:,} out) "
            f"in {ev['latency_ms'] / 1000:.1f}s"
            f"{'  [explored]' if ev['explored'] else '  [no exploration]'}"
        )


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "how many customers do we have?"
    session = str(uuid4())

    pool = await db.open_pool()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    graph = build_graph(checkpointer)

    out(f"{BOLD}? {question}{RESET}\n")
    try:
        async for ev in stream_turn(graph, session, question):
            show(ev)
    finally:
        await db.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
