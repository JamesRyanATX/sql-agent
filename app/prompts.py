"""The seam an optimiser writes through, and the loader behind it.

The prose lives in `config/prompts/<node>.md` and this module reads it. It used
to live here as six string constants, and moving it out was the point: a GEPA
winner is now written into the file it came from, so promotion is a reviewed
commit rather than a copy-paste into a Python string literal — the one step in
the loop that had no diff. See `config/prompts/README.md` for the contract and
for the notes the constants' comments used to carry.

What is in those files is prose that is a function of *nothing* — not the
question, not the cache, not the dialect. `render_cache()` and `dialect_note()`
stay in `graph.py`, because they compose per turn and are the content this prose
tells the model what to do with. The JSON schemas stay there too: they are the
node's wire contract, not prose.

A module rather than six reads and an `if`, for two reasons.

`optim/` needs the seed candidate without importing the world. `from app.graph
import EXTRACT_SYSTEM` drags in langgraph, sqlalchemy, `app.db`, `app.tools` and
`app.store`; this module imports stdlib and `app.settings`. It deliberately does
**not** import `app.config` either — where the prompts are is an environment
question (`CONFIG_DIR`), what the model does with them is a config-file one, and
only the first is needed to read a file.

And the blocks have to resolve **once per process**. `graph.plan` puts its system
block behind an Anthropic cache breakpoint on the promise that the block varies
with `connection_id` alone (§7.1). A prompt re-read per turn could change between
two turns of one server's life, and the only symptom would be T2 quietly costing
more. `_loaded` is memoised for that reason, not for speed — prose that is
constant for a whole process is strictly stronger than the invariant asks for,
which is why reading a file here is safe and would not be in `TurnState` or in
`config["configurable"]`.

Keys are the `node=` labels `llm.complete` records as the Langfuse generation
name, so a harvested trace and its prompt file name the same thing. `explore`
covers `explore.summary` as well: one prompt, two calls.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from app.settings import settings


# The vocabulary, and the only thing this module still hardcodes. Six files
# named for these, and nothing else, is the whole directory contract.
NODES = ("explore", "plan", "generate_sql", "fix", "extract", "answer")

# The one non-prompt file `config/prompts/` may hold. It is where the notes that
# used to be comments above the constants went, so it has to be allowed — and it
# has to be allowed by name, because "tolerate anything unrecognised" is exactly
# the rule this loader exists to not have.
NOTES = "README.md"


def directory() -> Path:
    """Where the prose lives. Named separately because errors quote it."""
    return Path(settings().config_dir) / "prompts"


@lru_cache
def _loaded() -> dict[str, str]:
    """`config/prompts/<node>.md`, one entry per node.

    Memoised deliberately — see the module docstring. Tests and the optimiser
    call `_loaded.cache_clear()` after moving the directory, the same way they
    already call `settings.cache_clear()`.

    Three things are errors rather than no-ops, and they are the same error:
    a prompt that silently is not what you think it is. A missing file would
    leave a node with nothing to say; an empty one is the placeholder somebody
    forgot to fill; a stray `extrct.md` is a typo that changes nothing, which
    means an optimisation run that measures the seed and reports it as a
    candidate improvement. That last one is the kind of wrong that agrees with
    itself, and it is why this loader has never had a lenient mode.
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

    The one thing Langfuse's own prompt management would have given for free:
    which prose produced this trace. Without it a harvest cannot tell a run
    under one revision from a run under the next, and round two of an
    optimisation trains on round one's output. Eight chars because this
    distinguishes revisions, it does not authenticate them.
    """
    return {
        name: hashlib.sha256(text.encode()).hexdigest()[:8]
        for name, text in sorted(_loaded().items())
    }
