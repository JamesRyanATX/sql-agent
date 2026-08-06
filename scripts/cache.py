"""Show what the agent has learned.

    make cache

Reads `GET /v1/cache`, which the server serves through `store.load_cache()` —
so what you see here is exactly what the model sees on the next turn: same
entries, same order. That equivalence is the point: the cache is the product
(PLAN.md §6.2), and beat 2 of the demo is someone reading it aloud.
"""

from scripts import _client

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"


async def main() -> None:
    payload = await _client.get("/v1/cache")
    entries = payload["entries"]
    summary = payload["summary"]

    if not entries:
        print(f"{DIM}cache is empty{RESET}")
        return

    print(
        f"{BOLD}{summary['total']} entries{RESET} "
        f"{DIM}({summary['verified']} verified"
        + (f", {summary['stale']} stale" if summary["stale"] else "")
        + (f", {summary['disabled']} disabled" if summary["disabled"] else "")
        + f"){RESET}\n"
    )

    for e in entries:
        tick = f"{GREEN}✓{RESET}" if e["verified"] else " "
        marks = []
        if e["origin"] == "human":
            marks.append(f"{YELLOW}human{RESET}")
        if e["pinned"]:
            marks.append(f"{YELLOW}pinned{RESET}")
        if e["tombstone"]:
            marks.append(f"{RED}tombstone{RESET}")
        # An entry whose tables have changed shape since it was written can no
        # longer be trusted (§5). Phase 6 acts on this; here it is just visible.
        if e["stale"]:
            marks.append(f"{RED}STALE{RESET}")
        suffix = f"  {' '.join(marks)}" if marks else ""

        print(
            f"{tick} {DIM}[{e['kind']}]{RESET} {BOLD}{e['name'] or '(unnamed)'}{RESET}"
            f"{DIM}  {e['hits']} hits{RESET}{suffix}"
        )
        print(f"    {e['claim']}")
        if e["sql_fragment"]:
            print(f"    {DIM}SQL: {e['sql_fragment']}{RESET}")
        if e["tables"]:
            print(f"    {DIM}tables: {', '.join(e['tables'])}{RESET}")
        print()


if __name__ == "__main__":
    _client.run(main())
