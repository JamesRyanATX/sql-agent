"""Wipe learned state — the stage recovery button (PLAN.md §9, Phase 9).

    make reset

`DELETE /v1/cache`. The server truncates the cache, the turn log and LangGraph's
checkpoint tables, leaving the business schema alone — `make reset` follows this
with `make seed` to put the business data back.
"""

from scripts import _client


async def main() -> None:
    wiped = (await _client.delete("/v1/cache"))["wiped"]
    for table, rows in wiped.items():
        print(f"  wiped {table} ({rows} rows)")
    if not wiped:
        # Only happens before the migrations have run, and silence would read
        # as success.
        print("  nothing to wipe — has `make migrate` run?")
    print("reset complete")


if __name__ == "__main__":
    _client.run(main())
