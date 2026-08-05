"""Wipe learned state — the stage recovery button (PLAN.md §9, Phase 9).

Truncates the cache, the turn log, and LangGraph's checkpoint tables, leaving
the business schema alone. Tolerant of tables that don't exist yet so it stays
usable as the project is built out phase by phase.
"""

import asyncio

import psycopg

from app.settings import settings

TABLES = [
    "cache_entry",
    "turn",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
]


async def main() -> None:
    async with await psycopg.AsyncConnection.connect(
        settings().database_url, autocommit=True
    ) as conn:
        for table in TABLES:
            cur = await conn.execute("SELECT to_regclass(%s)", (f"public.{table}",))
            row = await cur.fetchone()
            if row is None or row[0] is None:
                print(f"  skip  {table} (does not exist)")
                continue
            await conn.execute(f'TRUNCATE TABLE "{table}" RESTART IDENTITY CASCADE')
            print(f"  wiped {table}")
    print("reset complete")


if __name__ == "__main__":
    asyncio.run(main())
