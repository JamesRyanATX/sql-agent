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
from app.config import Config, Model, config
from app.settings import settings

DEMO = {
    "model": {"provider": "anthropic", "model": "claude-opus-5"},
    "max_tool_calls": 24,
    "extract": {"effort": "low"},
    "explore": {"effort": "high"},
}


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """A scratch CONFIG_DIR. `write` puts a config.yaml or an overlay in it."""

    def write(data: dict, *, local: dict | None = None) -> None:
        (tmp_path / config_module.FILE).write_text(yaml.safe_dump(data))
        if local is not None:
            (tmp_path / config_module.LOCAL).write_text(yaml.safe_dump(local))
        config.cache_clear()

    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))
    settings.cache_clear()
    config.cache_clear()
    write(DEMO)
    yield write
    monkeypatch.delenv("CONFIG_DIR", raising=False)
    settings.cache_clear()
    config.cache_clear()


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
    config.cache_clear()
    try:
        with pytest.raises(ValueError, match="no config at"):
            config()
    finally:
        monkeypatch.delenv("CONFIG_DIR", raising=False)
        settings.cache_clear()
        config.cache_clear()


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


def test_a_statement_timeout_is_milliseconds():
    """It was the string "5s" — a Postgres interval literal, which MySQL parses
    as neither a number nor an error. See app/dialects.py."""
    with pytest.raises(ValidationError):
        Config.model_validate({**DEMO, "statement_timeout_ms": "5s"})

    assert Config.model_validate(DEMO).statement_timeout_ms == 5_000


def test_effort_cannot_be_turned_off():
    """PLAN.md §7.1: on Opus 5, thinking disabled can turn a tool call into
    plain visible text that never runs, which silently breaks the explore
    loop. Cost is controlled with effort, so there is no value here for none."""
    with pytest.raises(ValidationError):
        Config.model_validate({**DEMO, "plan": {"effort": "none"}})


def test_a_yaml_that_is_not_a_mapping_says_so(config_dir, tmp_path):
    (tmp_path / config_module.FILE).write_text("- just\n- a list\n")
    config.cache_clear()

    with pytest.raises(ValueError, match="not a mapping"):
        config()
