"""Graph output → SSE frames.

Event types the UI binds to (PLAN.md §6.1):
    plan, explore, sql, error, fix, learned, answer, usage, done
"""

import json
from typing import Any

_USAGE_KEYS = ("tokens_in", "tokens_out")


def sse(ev: dict[str, Any]) -> str:
    return f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n"


def to_events(mode: str, chunk: Any) -> list[dict[str, Any]]:
    """Translate one astream chunk into zero or more UI events.

    `custom` chunks are already events — nodes emit them via get_stream_writer.
    `updates` chunks are {node: state_delta}; the only thing we surface from
    those is the running token count, which is what the counter binds to.
    """
    if mode == "custom":
        return [chunk] if isinstance(chunk, dict) and "type" in chunk else []

    if mode == "updates" and isinstance(chunk, dict):
        events = []
        for node, delta in chunk.items():
            if not isinstance(delta, dict):
                continue
            if any(k in delta for k in _USAGE_KEYS):
                events.append(
                    {
                        "type": "usage",
                        "node": node,
                        "tokens_in": delta.get("tokens_in", 0),
                        "tokens_out": delta.get("tokens_out", 0),
                    }
                )
        return events

    return []
