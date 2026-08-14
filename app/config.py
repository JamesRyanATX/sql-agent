"""`config/config.yaml`: which model, at what effort, within what bounds.

The other half of `app/settings.py`, and separate from it on purpose. Two
sources became two objects so that a call site says which one it reads:
`config().max_tool_calls` came out of a tracked file a human edits and reviews,
`settings().api_token` came out of the environment. Folding the yaml in as a
low-priority pydantic-settings source would have made those indistinguishable at
every call site, and would have left a stale `EFFORT_PLAN=low` in somebody's
`.env` silently beating the file they were editing.

Which is why the environment variables this replaced are *gone* rather than
demoted. `Settings` has `extra="ignore"`, so a retired name is dropped in
silence and the default quietly applies — `app/main.py` warns about every one of
them at startup for that reason.

Three layers, and the third is optional:

    config/config.yaml        tracked. The demo defaults: Anthropic, claude-opus-5.
    config/config.local.yaml  gitignored, deep-merged over it. Your local model.
    CONFIG_DIR                where both of those are. An env var, because where
                              the config lives cannot itself be config.

`extra="forbid"` at every level, so a key naming nothing is an error. That is
what makes it safe to put node names at the top level beside settings: `explor:`
cannot be mistaken for a node this file does not know about, because there is no
such thing — the six are declared fields.
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
    # A Literal because `llm.py` used to test `== "anthropic"` and treat
    # everything else as OpenAI-compatible, which made `openai_compatible` work
    # by accident and `anthropi` work by accident in the other direction.
    provider: Literal["anthropic", "openai_compat"] = "anthropic"
    model: str = "claude-opus-5"

    # openai_compat only; ignored by the Anthropic backend, which has one address.
    url: str | None = None

    # A thinking model handed an open-ended question will happily spend ten
    # minutes deliberating before its first tool call, so the ceiling on output
    # tokens is the ceiling on wall clock.
    max_tokens: int = 16_000
    # A 27B local model takes minutes, not seconds.
    timeout: float = 900.0

    @model_validator(mode="after")
    def _endpoint_is_known(self) -> Model:
        """`openai_compat` has no default address, and must not invent one.

        There is no such thing as *the* OpenAI-compatible endpoint, and the
        obvious default is the worst one: `localhost` from inside the api
        container is the api container, so a turn would fail on connect with an
        error about the wrong machine. Saying so while reading the file costs a
        line; finding it out mid-turn costs the demo.
        """
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

    # See PLAN.md §7.1: lowering effort is how a node is made cheap. Disabling
    # thinking is not — on Opus 5 that can turn a tool call into plain visible
    # text that never runs, which would silently break the explore loop. There
    # is deliberately no value here that disables it.
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
    # Milliseconds, an integer, because that is the only unit all three
    # supported dialects can be told. It was the string "5s" — a Postgres
    # interval literal, which MySQL parses as neither a number nor an error.
    # See app/dialects.py for what each one does with it.
    statement_timeout_ms: int = 5_000
    max_rows: int = 50  # rows handed back to the model from execute

    # The six nodes that talk to a model. `load_cache` is not among them and
    # never will be: it reads a table. A node here with no prompt file, or a
    # prompt file with no node here, is the same bug seen from two sides.
    plan: Node = Field(default_factory=Node)
    explore: Node = Field(default_factory=Node)
    generate_sql: Node = Field(default_factory=Node)
    fix: Node = Field(default_factory=Node)
    extract: Node = Field(default_factory=Node)
    answer: Node = Field(default_factory=Node)

    # GEPA's reflection model — the teacher that proposes candidate prompts.
    # Its own block because it is the one call in the system that is not part of
    # a turn, and because giving it a stronger model than production runs is the
    # normal thing to want. `optim/adapter.py` calls it with node="gepa.reflect".
    gepa: Node = Field(default_factory=lambda: Node(effort="max"))

    # --- resolution --------------------------------------------------------

    def node(self, name: str) -> Node:
        """The block for a `node=` label, or an empty one.

        Dotted labels fall back to their prefix, which is the whole mapping
        table: `explore.summary` resolves to `explore` (one loop, one prompt,
        two calls), `extract.replay` to `extract` (so `optim/`'s harness runs
        under production's own effort rather than measuring a configuration
        nobody ships), and `gepa.reflect` to `gepa`.

        An unknown label is not an error: `llm.complete`'s default is the
        literal `"model"`, and a call that names nothing should get the global
        defaults rather than a traceback.
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

    Deep rather than shallow so that a local file naming only `model.model`
    keeps the tracked file's `model.provider`. A shallow merge would replace the
    whole block, which is how a local override of one field silently resets the
    other three.
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

    Its own function because the *presence* of an overlay is a fact worth
    saying out loud at startup: it is untracked, so what the server is running
    is not what the repository says it runs, and "which model is this?" has a
    surprising answer exactly when this file exists.
    """
    path = Path(settings().config_dir) / LOCAL
    return path if path.is_file() else None


@lru_cache
def config() -> Config:
    """`config/config.yaml`, merged with `config.local.yaml` and validated.

    Memoised like `settings()`, and for a stronger reason than either: `plan`'s
    system block sits behind an Anthropic cache breakpoint, and a model that
    changed between two turns of one process would invalidate the prefix with no
    symptom but the token counter going back up. Tests call `config.cache_clear()`.

    A missing `config.yaml` is an error, not a fall-through to defaults. The
    file is tracked, so its absence means CONFIG_DIR is pointed somewhere wrong,
    and defaulting there would run the demo against a model nobody chose.
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
