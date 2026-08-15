"""`config/config.yaml`: which model, at what effort, within what bounds.

Behaviour. Secrets and addresses are `app/settings.py`, and they are two objects
so a call site says which it reads.

    config/config.yaml        tracked. The demo defaults: Anthropic, claude-opus-5.
    config/config.local.yaml  gitignored, deep-merged over it. Your local model.
    CONFIG_DIR                where both of those are.

`extra="forbid"` at every level, which is what makes it safe to put node names at
the top level beside settings: a typo is an error, not a key configuring nothing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.settings import settings

FILE = "config.yaml"
LOCAL = "config.local.yaml"


class Model(BaseModel):
    """Where a call goes. The key is not here — see app/settings.py."""

    model_config = ConfigDict(extra="forbid")

    # `openai_compat` is any OpenAI-shaped endpoint (Ollama, vLLM, LM Studio).
    # A Literal so a misspelled provider is an error rather than a silent fallback.
    provider: Literal["anthropic", "openai_compat"] = "anthropic"
    model: str = "claude-opus-5"

    # openai_compat only; ignored by the Anthropic backend, which has one address.
    url: str | None = None

    # For a thinking model the ceiling on output tokens is the ceiling on wall clock.
    max_tokens: int = 16_000
    # A 27B local model takes minutes, not seconds.
    timeout: float = 900.0

    @model_validator(mode="after")
    def _endpoint_is_known(self) -> Model:
        """`openai_compat` has no default address, and must not invent one."""
        if self.provider == "openai_compat" and not self.url:
            raise ValueError(
                "provider: openai_compat needs a url — there is no default "
                "endpoint. It must not be localhost: the API runs in a "
                "container, where localhost is the container."
            )
        return self


class Node(BaseModel):
    """One graph node's overrides. Everything here is optional."""

    model_config = ConfigDict(extra="forbid")

    # PLAN.md §7.1. Lowering effort is how a node is made cheap; disabling
    # thinking is not, and no value here does it — on Opus 5 that can turn a tool
    # call into visible text that never runs, breaking the explore loop.
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    model: Model | None = None


class Config(BaseModel):
    """`config/config.yaml`, whole."""

    model_config = ConfigDict(extra="forbid")

    # The default for every node that does not override it.
    model: Model = Field(default_factory=Model)

    # Bounds, so T1 doesn't run forever on stage.
    max_tool_calls: int = 24
    max_fix_attempts: int = 3
    # Milliseconds as an integer, the only unit all three dialects can be told.
    # See app/dialects.py for what each one does with it.
    statement_timeout_ms: int = 5_000
    max_rows: int = 50  # rows handed back to the model from execute

    # The six nodes that talk to a model. `load_cache` reads a table, so it is
    # not one. Each of these must have a prompt file, and vice versa.
    plan: Node = Field(default_factory=Node)
    explore: Node = Field(default_factory=Node)
    generate_sql: Node = Field(default_factory=Node)
    fix: Node = Field(default_factory=Node)
    extract: Node = Field(default_factory=Node)
    answer: Node = Field(default_factory=Node)

    # GEPA's teacher, which proposes candidate prompts. Its own block because it
    # is the one call that is not part of a turn, and usually wants a stronger
    # model. `optim/adapter.py` calls it with node="gepa.reflect".
    gepa: Node = Field(default_factory=lambda: Node(effort="max"))

    # --- resolution --------------------------------------------------------

    def node(self, name: str) -> Node:
        """The block for a `node=` label, or an empty one.

        A dotted label falls back to its prefix — `explore.summary` to `explore`,
        `gepa.reflect` to `gepa`. An unknown label gets the global defaults.
        """
        block = getattr(self, name, None)
        if block is None and "." in name:
            block = getattr(self, name.split(".", 1)[0], None)
        return block if isinstance(block, Node) else Node()

    def model_for(self, name: str) -> Model:
        """Which model a `node=` label goes to. The override, or the global."""
        return self.node(name).model or self.model

    def effort_for(self, name: str, default: str = "medium") -> str:
        """How hard it should think. See PLAN.md §7.1 before lowering one."""
        return self.node(name).effort or default


# ------------------------------------------------------------------- the loader


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` wins, one key at a time, recursing into dicts.

    Deep, so a local file naming only `model.model` keeps the tracked file's
    `model.provider` rather than resetting the whole block.
    """
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a mapping — it parsed as {type(loaded).__name__}")
    return loaded


def overlay() -> Path | None:
    """`config.local.yaml`, if there is one.

    Its own function because the presence of an untracked overlay is worth a
    startup warning: the server is not running what the repository says it runs.
    """
    path = Path(settings().config_dir) / LOCAL
    return path if path.is_file() else None


@lru_cache
def config() -> Config:
    """`config/config.yaml`, merged with `config.local.yaml` and validated.

    Memoised: a model that changed between two turns of one process would
    invalidate `plan`'s cache prefix with no symptom but the token counter going
    up. Tests call `config.cache_clear()`.

    A missing `config.yaml` is an error rather than a fall-through to defaults —
    the file is tracked, so its absence means CONFIG_DIR is wrong.
    """
    root = Path(settings().config_dir)
    path = root / FILE
    if not path.is_file():
        raise ValueError(
            f"no config at {path} — CONFIG_DIR is {settings().config_dir!r}. "
            f"The file is tracked in git; if it is genuinely missing, copy it "
            f"from the repository rather than letting defaults apply."
        )

    data = _read(path)
    local = overlay()
    if local is not None:
        data = _merge(data, _read(local))

    return Config.model_validate(data)
