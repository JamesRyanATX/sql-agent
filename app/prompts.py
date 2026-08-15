"""Reads `config/prompts/<node>.md`, the prose half of what a node sends.

Only prose that is a function of nothing — not the question, not the cache, not
the dialect. Anything composed per turn stays in `graph.py`, as do the JSON
schemas, which are wire contracts rather than prose.

Blocks resolve **once per process**: `graph.plan` puts its system block behind an
Anthropic cache breakpoint promising it varies with `connection_id` alone, so a
prompt changing mid-process would show up only as T2 costing more.

Imports stdlib and `app.settings` only, so `optim/` can read a seed candidate
without pulling in langgraph and sqlalchemy.

Keys are the `node=` labels `llm.complete` records as Langfuse generation names.
`explore` covers `explore.summary` too: one prompt, two calls.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from app.settings import settings


# Six files named for these, and nothing else, is the directory contract.
NODES = ("explore", "plan", "generate_sql", "fix", "extract", "answer")

# The one non-prompt file `config/prompts/` may hold. Allowed by name, because
# tolerating anything unrecognised is the rule this loader exists to not have.
NOTES = "README.md"


def directory() -> Path:
    """Where the prose lives. Named separately because errors quote it."""
    return Path(settings().config_dir) / "prompts"


@lru_cache
def _loaded() -> dict[str, str]:
    """`config/prompts/<node>.md`, one entry per node.

    Missing, empty and misnamed files are all errors — each one is a prompt that
    silently is not what you think it is. Tests call `_loaded.cache_clear()`.
    """
    root = directory()
    if not root.is_dir():
        raise ValueError(
            f"no prompt directory at {root} — CONFIG_DIR is "
            f"{settings().config_dir!r}, and it must hold a prompts/ directory "
            f"with {len(NODES)} files: {', '.join(f'{n}.md' for n in NODES)}"
        )

    expected = {f"{n}.md" for n in NODES} | {NOTES}
    unknown = {p.name for p in root.glob("*.md")} - expected
    if unknown:
        raise ValueError(
            f"{root} holds .md files naming no prompt: {sorted(unknown)} — "
            f"expected some of {sorted(f'{n}.md' for n in NODES)}"
        )

    blocks: dict[str, str] = {}
    missing: list[str] = []
    for name in NODES:
        path = root / f"{name}.md"
        text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
        if not text:
            missing.append(f"{name}.md")
        blocks[name] = text
    if missing:
        raise ValueError(
            f"{root} is missing or empty for: {sorted(missing)} — every node "
            f"needs prose, and an empty file is a placeholder, not a prompt"
        )
    return blocks


def get(name: str) -> str:
    """The instruction block for a node."""
    return _loaded()[name]


def fingerprint() -> dict[str, str]:
    """node -> 8 hex chars of its prompt, for the turn span's metadata.

    Which prose produced a trace. Without it a harvest cannot tell one revision
    from the next, and round two of an optimisation trains on round one's output.
    """
    return {
        name: hashlib.sha256(text.encode()).hexdigest()[:8]
        for name, text in sorted(_loaded().items())
    }
