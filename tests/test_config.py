"""`config/config.yaml`: the layers, the resolution, and what must not be quiet.

The theme is the same one `app/prompts.py` has: a key that names nothing must
say so. `Settings` has `extra="ignore"` and a mistyped env var is dropped in
silence — that is the failure mode this file is built to not repeat, which is
why every negative case here asserts on the *message* as well as the raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app import config as config_module
from app.config import Config, Model, config, leaves, overrides
from app.settings import settings
from tests.conftest import DEMO, clear_config

# ----------------------------------------------------------------- the layers


def test_the_tracked_file_is_what_ships():
    """A fresh clone runs the demo: Anthropic, and the effort PLAN.md §7.1 names.

    Reads the real config/config.yaml, deliberately — this is the assertion that
    the file in git is the demo's configuration and not somebody's local one.
    Note it may be overlaid: config.local.yaml is gitignored, so this asserts on
    the tracked layer alone.
    """
    path = Path(__file__).resolve().parent.parent / "config" / config_module.FILE
    assert path.is_file(), "config/config.yaml is tracked and must be here"
    loaded = Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    assert loaded.model.provider == "anthropic"
    assert loaded.model.model == "claude-opus-5"
    assert loaded.effort_for("plan") == "low"
    assert loaded.effort_for("explore") == "high"
    assert loaded.effort_for("generate_sql") == "high"


def test_the_local_overlay_wins_key_by_key(config_dir):
    """Deep, not shallow. A local file naming only `model.model` keeps the
    tracked file's provider — a shallow merge would silently reset it."""
    config_dir(
        {"model": {"provider": "anthropic", "model": "claude-opus-5"},
         "max_tool_calls": 24},
        local={"model": {"model": "claude-sonnet-5"}, "max_tool_calls": 3},
    )

    assert config().model.model == "claude-sonnet-5"
    assert config().model.provider == "anthropic"
    assert config().max_tool_calls == 3


def test_no_overlay_is_the_normal_case(config_dir):
    config_dir(DEMO)
    assert config().model.model == "claude-opus-5"


def test_a_missing_config_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    """The file is tracked, so its absence means CONFIG_DIR is pointed
    somewhere wrong — and defaulting there runs the demo against a model
    nobody chose."""
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))
    settings.cache_clear()
    clear_config()
    try:
        with pytest.raises(ValueError, match="no config at"):
            config()
    finally:
        monkeypatch.delenv("CONFIG_DIR", raising=False)
        settings.cache_clear()
        clear_config()


# ------------------------------------------------------------- the resolution


def test_a_node_falls_back_to_the_global_model():
    loaded = Config.model_validate(DEMO)
    assert loaded.model_for("extract").model == "claude-opus-5"


def test_a_node_may_name_its_own_model():
    """The reason `assistant_turn` and `tool_results` take a node: a per-node
    override can change the *backend*, and the two disagree about wire format."""
    loaded = Config.model_validate(
        {**DEMO, "explore": {"effort": "high",
                             "model": {"provider": "openai_compat",
                                       "model": "qwen3", "url": "http://x/v1"}}}
    )

    assert loaded.model_for("explore").provider == "openai_compat"
    assert loaded.model_for("extract").provider == "anthropic"


@pytest.mark.parametrize(
    "label,node",
    [
        ("explore.summary", "explore"),   # one prompt, two calls
        ("extract.replay", "extract"),    # tools/gepa/'s single-node harness
        ("gepa.reflect", "gepa"),         # the teacher, not part of a turn
    ],
)
def test_a_dotted_label_resolves_to_its_prefix(label, node):
    """`tools/gepa/`'s harness must run under production's own effort, or it is
    measuring a configuration nobody ships."""
    loaded = Config.model_validate(
        {**DEMO, "extract": {"effort": "low"}, "gepa": {"effort": "max"}}
    )
    assert loaded.effort_for(label) == loaded.effort_for(node)


def test_an_unknown_label_gets_the_global_defaults():
    """`llm.complete`'s default node is the literal "model". A call that names
    nothing should not raise — it should get the defaults."""
    loaded = Config.model_validate(DEMO)
    assert loaded.model_for("model").model == "claude-opus-5"
    assert loaded.effort_for("model") == "medium"


# ----------------------------------------------- what must not be a no-op


def test_a_key_naming_nothing_is_an_error():
    """`extra="forbid"` is what makes it safe to put node names at the top
    level beside settings. Without it this typo would configure nothing and
    say nothing."""
    with pytest.raises(ValidationError, match="explor"):
        Config.model_validate({**DEMO, "explor": {"effort": "high"}})


def test_load_cache_is_not_a_node():
    """It reads a table. There is no model call to set an effort on, and
    accepting the key would imply there was."""
    with pytest.raises(ValidationError, match="load_cache"):
        Config.model_validate({**DEMO, "load_cache": {"effort": "medium"}})


def test_the_provider_spelling_is_checked():
    """`llm.py` used to test `== "anthropic"` and treat everything else as
    OpenAI-compatible, which made `openai_compatible` work by accident — and
    `anthropi` silently route to an endpoint that does not exist."""
    with pytest.raises(ValidationError):
        Model.model_validate({"provider": "openai_compatible", "model": "x"})


def test_openai_compat_must_name_its_endpoint():
    """There is no default address, and the obvious one is the worst: localhost
    from inside the api container is the api container."""
    with pytest.raises(ValidationError, match="needs a url"):
        Model.model_validate({"provider": "openai_compat", "model": "qwen3"})


OPENAI = {"provider": "openai_compat", "model": "gpt-5.6-luna",
          "url": "https://api.openai.com/v1"}


def test_effort_none_is_allowed_on_an_openai_compat_node():
    """OpenAI's chat completions endpoint refuses function tools on a reasoning
    model unless `reasoning_effort` is `none` — and structured output is a
    forced tool call, so every node but `answer` hits it."""
    loaded = Config.model_validate({**DEMO, "model": OPENAI,
                                    "explore": {"effort": "none"}})
    assert loaded.effort_for("explore") == "none"


def test_effort_none_is_refused_on_an_anthropic_node():
    """PLAN.md §7.1: on Opus 5 disabling thinking can turn a tool call into
    visible text that never runs. `none` is the endpoint's rule, not a cost
    lever, and the check is per node — a global `none` with one Anthropic
    override is still an error."""
    with pytest.raises(ValidationError, match="effort: none on explore"):
        Config.model_validate({**DEMO, "explore": {"effort": "none"}})
    with pytest.raises(ValidationError, match="effort: none on gepa"):
        Config.model_validate({
            **DEMO, "model": OPENAI,
            "gepa": {"effort": "none", "model": DEMO["model"]},
        })


def test_a_statement_timeout_is_milliseconds():
    """It was the string "5s" — a Postgres interval literal, which MySQL parses
    as neither a number nor an error. See app/dialects.py."""
    with pytest.raises(ValidationError):
        Config.model_validate({**DEMO, "statement_timeout_ms": "5s"})

    assert Config.model_validate(DEMO).statement_timeout_ms == 5_000


def test_effort_cannot_be_turned_off_on_anthropic():
    """PLAN.md §7.1: on Opus 5, thinking disabled can turn a tool call into
    plain visible text that never runs, which silently breaks the explore
    loop. Cost is controlled with effort there. `none` exists for OpenAI's
    endpoint, which demands it — see the openai_compat tests above."""
    with pytest.raises(ValidationError):
        Config.model_validate({**DEMO, "plan": {"effort": "none"}})


# ---------------------------------------------------------- what the overlay set


def test_leaves_are_the_paths_the_merge_would_write():
    assert sorted(leaves({"model": {"provider": "x", "url": None}, "max_rows": 1})) == [
        "max_rows", "model.provider", "model.url",
    ]
    # A block replaced wholesale is a leaf: that is what `_merge` does with it.
    assert leaves({"explore": None}) == ["explore"]
    assert leaves({"explore": {}}) == ["explore"]
    assert leaves({}) == []


def test_overrides_are_the_overlay_keys_sorted(config_dir):
    write = config_dir
    assert overrides() == ()

    write(DEMO, local={"model": {"model": "qwen3", "provider": "openai_compat",
                                 "url": "http://h/v1"},
                       "explore": {"effort": "low"}})
    assert overrides() == (
        "explore.effort", "model.model", "model.provider", "model.url",
    )


def test_a_yaml_that_is_not_a_mapping_says_so(config_dir, tmp_path):
    (tmp_path / config_module.FILE).write_text("- just\n- a list\n")
    clear_config()

    with pytest.raises(ValueError, match="not a mapping"):
        config()


# ------------------------------------------------------- reached through a gateway


ROUTER = {"provider": "openrouter", "url": "https://openrouter.ai/api/v1"}


def test_a_gateway_still_needs_its_address_named():
    """It has a published one, but the file is where somebody looks to answer
    "where did this call go, and who charged me for it"."""
    with pytest.raises(ValidationError, match="needs a url"):
        Model.model_validate({"provider": "openrouter", "model": "x/y"})


def test_effort_none_is_refused_on_claude_however_it_is_reached():
    """The rule exists because of how Claude behaves, so it follows the model.
    Keyed on the provider it would go quiet here, and `none` would reach Opus
    through a gateway — thinking off, a tool call written as visible text, and
    an explore loop that never runs one."""
    with pytest.raises(ValidationError, match="effort: none on explore"):
        Config.model_validate({
            **DEMO,
            "model": {**ROUTER, "model": "anthropic/claude-opus-5"},
            "explore": {"effort": "none"},
        })


def test_effort_none_is_allowed_on_a_model_that_is_not_claude():
    """Gemini through the same gateway. The guard is about one model family,
    not about gateways."""
    loaded = Config.model_validate({
        **DEMO,
        "model": {**ROUTER, "model": "google/gemini-2.5-flash"},
        "explore": {"effort": "none"},
    })

    assert loaded.effort_for("explore") == "none"


def test_a_per_node_claude_model_is_caught_under_a_cheap_default():
    """The configuration this project is heading for: cheap rollouts, a better
    model for the one node that proposes rewrites. The guard has to see the
    node's own model, not the global one."""
    with pytest.raises(ValidationError, match="effort: none on gepa"):
        Config.model_validate({
            **DEMO,
            "model": {**ROUTER, "model": "google/gemini-2.5-flash"},
            "gepa": {
                "effort": "none",
                "model": {**ROUTER, "model": "anthropic/claude-sonnet-5"},
            },
        })
