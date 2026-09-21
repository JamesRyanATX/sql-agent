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
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# The function, not the module: `overrides()` below is already taken — it is
# the overlay's key list, which is a different question entirely.
from app.overrides import current as current_overrides
from app.settings import settings

# The two providers that speak OpenAI's wire shape over plain HTTP, and so need
# an address and a key rather than an SDK.
HTTP_PROVIDERS = ("openai_compat", "openrouter")
OPENROUTER_URL = "https://openrouter.ai/api/v1"

FILE = "config.yaml"
LOCAL = "config.local.yaml"


class Model(BaseModel):
    """Where a call goes. The key is not here — see app/settings.py."""

    model_config = ConfigDict(extra="forbid")

    # `openai_compat` is any OpenAI-shaped endpoint (Ollama, vLLM, LM Studio).
    # `openrouter` is OpenAI-shaped too, and is its own value because three
    # things differ and all three cost money to get wrong: effort goes in a
    # `reasoning` object rather than a flat field, prompt caching needs an
    # explicit marker, and the response reports what the call charged.
    # A Literal so a misspelled provider is an error rather than a silent fallback.
    provider: Literal["anthropic", "openai_compat", "openrouter"] = "anthropic"
    model: str = "claude-opus-5"

    # Both OpenAI-shaped providers; ignored by the Anthropic backend, which has
    # one address.
    url: str | None = None

    # For a thinking model the ceiling on output tokens is the ceiling on wall clock.
    max_tokens: int = 16_000
    # A 27B local model takes minutes, not seconds.
    timeout: float = 900.0

    @model_validator(mode="after")
    def _endpoint_is_known(self) -> Model:
        """An OpenAI-shaped provider has no default address, and must not invent
        one. OpenRouter has a published address but still names it here, because
        a url in the file is the one place somebody looks to answer "where did
        this call go, and who charged me for it"."""
        if self.provider in HTTP_PROVIDERS and not self.url:
            raise ValueError(
                f"provider: {self.provider} needs a url — there is no default "
                "endpoint. It must not be localhost: the API runs in a "
                "container, where localhost is the container."
                + (f" For OpenRouter that is {OPENROUTER_URL}."
                   if self.provider == "openrouter" else "")
            )
        return self

    def is_claude(self) -> bool:
        """Whether this ends up at a Claude model, however it is reached.

        Not `provider == "anthropic"`: through a gateway a Claude model arrives
        as an OpenAI-shaped request, and the rules that exist because of how
        Claude behaves have to follow the model rather than the wire format.
        OpenRouter spells the family as a prefix, which is what makes this
        answerable at all.
        """
        return self.provider == "anthropic" or self.model.startswith("anthropic/")


class Node(BaseModel):
    """One graph node's overrides. Everything here is optional."""

    model_config = ConfigDict(extra="forbid")

    # PLAN.md §7.1. Lowering effort is how a node is made cheap; disabling
    # thinking is not — on Opus 5 that can turn a tool call into visible text
    # that never runs, breaking the explore loop. `none` exists because OpenAI's
    # chat completions endpoint refuses function tools on a reasoning model at
    # any other setting, and structured output is a forced tool call. It is an
    # openai_compat value only; Config rejects it on an Anthropic node.
    effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
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

    # Dollars a single command may spend before it stops. Applies to the long
    # unattended ones — recording a corpus, running a search — not to one
    # question asked by hand. Zero means no ceiling, which is the old
    # behaviour and has to be asked for.
    #
    # It can only be enforced where the backend reports what it charged, which
    # today is OpenRouter alone. Elsewhere a run reports nothing and stops at
    # nothing, and the commands say so rather than implying a guard they do not
    # have.
    max_spend: float = 5.0

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
    # model. `tools/gepa/adapter.py` calls it with node="gepa.reflect".
    gepa: Node = Field(default_factory=lambda: Node(effort="max"))

    NODES: ClassVar[tuple[str, ...]] = (
        "plan", "explore", "generate_sql", "fix", "extract", "answer", "gepa",
    )

    @model_validator(mode="after")
    def _no_thinking_only_where_it_is_the_endpoints_rule(self) -> Config:
        """`effort: none` is an OpenAI constraint, not a cost lever (PLAN.md §7.1)."""
        offenders = [
            name
            for name in self.NODES
            if self.node(name).effort == "none" and self.model_for(name).is_claude()
        ]
        if offenders:
            raise ValueError(
                f"effort: none on {', '.join(offenders)} — Claude has no such "
                f"level, and disabling thinking on Opus 5 breaks tool calls: it "
                f"writes one as visible text that never runs. Lower the effort "
                f"instead. (The check follows the model, not the provider, so a "
                f"Claude model reached through a gateway is caught too.)"
            )
        return self

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
        """How hard it should think. See PLAN.md §7.1 before lowering one.

        An optimisation run may put a different effort in force for one turn.
        `none` is refused here for the same reason the file-time validator
        refuses it: on Opus 5 it can turn a tool call into visible text that
        never runs, and a candidate proposing it would score as a broken graph
        rather than as the bad idea it is.
        """
        # The same fallback `node()` does, so an override on `explore` covers
        # the `explore.summary` call the way a config block would.
        proposed = current_overrides().efforts.get(name)
        if proposed is None and "." in name:
            proposed = current_overrides().efforts.get(name.split(".", 1)[0])
        if proposed is not None:
            if proposed == "none" and self.model_for(name).is_claude():
                raise ValueError(
                    f"effort: none on {name} — Claude has no such level, and "
                    f"disabling thinking on Opus 5 breaks tool calls"
                )
            return proposed
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


def leaves(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Every dotted path in a mapping that `_merge` would write as a value.

    A scalar, or anything that replaces a block wholesale — `explore: null`
    is a leaf named `explore`, because that is what the merge does with it.
    """
    out: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict) and value:
            out.extend(leaves(value, f"{prefix}{key}."))
        else:
            out.append(f"{prefix}{key}")
    return out


@lru_cache
def overrides() -> tuple[str, ...]:
    """Which keys `config.local.yaml` set, sorted.

    Memoised for the same reason `config()` is: the answer must describe the
    process, not the disk. An overlay edited since boot would otherwise mark
    keys the running config never saw.
    """
    local = overlay()
    return tuple(sorted(leaves(_read(local)))) if local is not None else ()


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
