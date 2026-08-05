"""Show what the agent has learned.

    make cache

Reads through `store.load_cache()` rather than querying the table directly, so
what you see here is exactly what the model sees on the next turn — same
entries, same order. That equivalence is the point: the cache is the product
(PLAN.md §6.2), and beat 2 of the demo is someone reading it aloud.
"""

import asyncio

from app import db, store

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"


async def main() -> None:
    await db.open_pool()
    try:
        async with db.connection() as conn:
            entries = await store.load_cache(conn)

            cur = await conn.execute(
                "SELECT count(*) AS n FROM cache_entry WHERE disabled"
            )
            hidden = (await cur.fetchone())["n"]

            # An entry whose tables have changed shape since it was written can
            # no longer be trusted (§5). Phase 6 acts on this; here it is just
            # visible.
            stale = set()
            for e in entries:
                if e.schema_fp and e.tables:
                    now = await store.schema_fingerprint(conn, e.tables)
                    if now != e.schema_fp:
                        stale.add(e.id)

        if not entries:
            print(f"{DIM}cache is empty{RESET}")
            return

        verified = sum(1 for e in entries if e.verified)
        print(
            f"{BOLD}{len(entries)} entries{RESET} "
            f"{DIM}({verified} verified"
            + (f", {len(stale)} stale" if stale else "")
            + (f", {hidden} disabled" if hidden else "")
            + f"){RESET}\n"
        )

        for e in entries:
            tick = f"{GREEN}✓{RESET}" if e.verified else " "
            marks = []
            if e.origin == "human":
                marks.append(f"{YELLOW}human{RESET}")
            if e.pinned:
                marks.append(f"{YELLOW}pinned{RESET}")
            if e.tombstone:
                marks.append(f"{RED}tombstone{RESET}")
            if e.id in stale:
                marks.append(f"{RED}STALE{RESET}")
            suffix = f"  {' '.join(marks)}" if marks else ""

            print(
                f"{tick} {DIM}[{e.kind}]{RESET} {BOLD}{e.name or '(unnamed)'}{RESET}"
                f"{DIM}  {e.hits} hits{RESET}{suffix}"
            )
            print(f"    {e.claim}")
            if e.sql_fragment:
                print(f"    {DIM}SQL: {e.sql_fragment}{RESET}")
            if e.tables:
                print(f"    {DIM}tables: {', '.join(e.tables)}{RESET}")
            print()
    finally:
        await db.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
