"""Different text for one turn, and nothing leaking out of it.

The load-bearing test here is the one that runs a real graph and asserts on what
the model was *sent*. Everything downstream — the whole-turn metric, the
multi-component search — is measuring whichever text actually reached the node,
and if that is the file's text rather than the candidate's, every number the
optimisation produces is about the wrong thing and nothing fails.

Requires Postgres up (`make up && make migrate && make seed`), like
test_coldpath.py, which this borrows its scripted model from. No model calls.
"""

from __future__ import annotations

import hashlib

import pytest

from app import llm, overrides, prompts, tools
from app.config import Config, config
from tests.conftest import DEMO
from tests.test_coldpath import (  # noqa: F401 — `pool` is a fixture
    ScriptedModel,
    json_result,
    no_entries,
    pool,
    run,
    text_result,
    tool_result,
)

CANDIDATE = "You are a different instruction block entirely."


# ---------------------------------------------------------------- on their own


def test_nothing_is_overridden_by_default():
    """What production runs under, asserted rather than assumed: every reader
    falls through to the file."""
    assert overrides.current() == overrides.NONE
    assert prompts.get("extract") == prompts._loaded()["extract"]
    assert tools.schemas() is tools.SCHEMAS


def test_a_prompt_override_is_in_force_only_inside_the_block():
    file_text = prompts.get("plan")

    with overrides.using(overrides.Overrides(prompts={"plan": CANDIDATE})):
        assert prompts.get("plan") == CANDIDATE
        # One node, not all of them: a candidate that changed `plan` must not
        # silently change what `extract` was measured under.
        assert prompts.get("extract") == prompts._loaded()["extract"]

    assert prompts.get("plan") == file_text


def test_the_prompt_cache_is_never_touched():
    """The override sits above the memoised read, so a candidate cannot leave
    its text behind for the next turn in this process."""
    with overrides.using(overrides.Overrides(prompts={"plan": CANDIDATE})):
        assert prompts._loaded()["plan"] != CANDIDATE


def test_the_fingerprint_still_describes_the_files():
    """The turn span records which committed prose the process is running. If a
    candidate's text changed that, a harvest could not tell a run under the seed
    from a run under a mutation."""
    from_files = hashlib.sha256(prompts._loaded()["plan"].encode()).hexdigest()[:8]

    with overrides.using(overrides.Overrides(prompts={"plan": CANDIDATE})):
        assert prompts.fingerprint()["plan"] == from_files


def test_a_tool_description_is_replaced_and_nothing_else_is():
    """Only the description. The name and the input schema are what `run_tool`
    dispatches and splats, so a candidate that changed those would break the
    call rather than score badly for it."""
    with overrides.using(overrides.Overrides(tools={"list_tables": "Say the tables."})):
        swapped = {s["name"]: s for s in tools.schemas()}
        original = {s["name"]: s for s in tools.SCHEMAS}

        assert swapped["list_tables"]["description"] == "Say the tables."
        assert swapped["list_tables"]["input_schema"] == original["list_tables"]["input_schema"]
        assert set(swapped) == set(original)
        # The three it said nothing about keep theirs.
        for name in ("describe_table", "sample_column", "count_distinct"):
            assert swapped[name] == original[name]

    assert tools.schemas() is tools.SCHEMAS


def test_a_tool_that_does_not_exist_is_ignored():
    """A candidate describing a fifth tool is wrong, and the feedback should say
    so — but the rollout has to finish in order to say anything."""
    with overrides.using(overrides.Overrides(tools={"drop_table": "no"})):
        assert [s["name"] for s in tools.schemas()] == [s["name"] for s in tools.SCHEMAS]


def test_an_effort_override_covers_the_dotted_label_too():
    """`explore.summary` is the second call the explore loop makes, and it
    resolves to the `explore` block. An override has to do the same, or half of
    exploring is measured at one effort and half at another."""
    with overrides.using(overrides.Overrides(efforts={"explore": "low"})):
        assert config().effort_for("explore") == "low"
        assert config().effort_for("explore.summary") == "low"
        assert config().effort_for("extract") == config().node("extract").effort


def test_effort_none_is_still_refused_on_anthropic():
    """The file-time validator cannot see a candidate, so the check is here as
    well. PLAN.md §7.1: on Opus 5 this turns a tool call into visible text that
    never runs, which reads as a broken graph rather than a bad candidate.

    Built rather than read from disk: whether this raises depends on the
    provider, and the provider depends on whichever `config.local.yaml` the
    developer happens to have.
    """
    anthropic = Config.model_validate(DEMO)

    with (
        overrides.using(overrides.Overrides(efforts={"plan": "none"})),
        pytest.raises(ValueError, match="effort: none on plan"),
    ):
        anthropic.effort_for("plan")


def test_effort_none_is_allowed_where_the_endpoint_demands_it():
    """The other half, and the reason the value exists: OpenAI's chat
    completions endpoint refuses function tools on a reasoning model at any
    other setting."""
    openai = Config.model_validate({
        **DEMO,
        "model": {"provider": "openai_compat", "model": "gpt-5.6-luna",
                  "url": "https://api.openai.com/v1"},
    })

    with overrides.using(overrides.Overrides(efforts={"plan": "none"})):
        assert openai.effort_for("plan") == "none"


# --------------------------------------------------------- inside a real graph
#
# A cold turn never calls `plan` — with nothing in the cache there is nothing to
# judge sufficient, and skipping that call is what makes T1 cheaper than it
# looks. So these override `explore` and `generate_sql`, which do run.


def cold_turn() -> ScriptedModel:
    """The five calls a cold turn makes: two exploring, then SQL, extract, answer."""
    return ScriptedModel(
        tool_result("list_tables", {}),
        text_result("customer holds 2,000 rows"),
        json_result({"sql": "SELECT count(*) AS n FROM customer", "assumptions": []}),
        no_entries(),
        text_result("1,840 customers."),
    )


async def test_the_override_reaches_a_node_inside_the_graph(monkeypatch, pool):
    """The one that matters.

    LangGraph runs each node as its own task, and a task gets a *copy* of the
    context. If that copy did not carry the candidate, every reader above would
    still pass and the optimisation would be scoring the file's text under a
    candidate's name. So this asserts on what the model was handed, several
    nodes deep, rather than on a return value.
    """
    model = cold_turn()
    monkeypatch.setattr(llm, "complete", model)

    with overrides.using(
        overrides.Overrides(
            prompts={"explore": CANDIDATE},
            tools={"list_tables": "Say the tables, briefly."},
            efforts={"explore": "low"},
        )
    ):
        await run()

    exploring = [c for c in model.calls if c.get("node", "").startswith("explore")]
    assert exploring, [c.get("node") for c in model.calls]
    assert exploring[0]["system"].startswith(CANDIDATE)
    assert exploring[0]["effort"] == "low"

    offered = {t["name"]: t["description"] for t in exploring[0]["tools"]}
    assert offered["list_tables"] == "Say the tables, briefly."
    assert offered["describe_table"].startswith("Show a table's columns")

    # A node the candidate said nothing about ran under the committed prose.
    sql_call = next(c for c in model.calls if c.get("node") == "generate_sql")
    assert sql_call["system"].startswith(prompts._loaded()["generate_sql"])


async def test_a_turn_outside_the_block_is_unaffected(monkeypatch, pool):
    """Two turns in one process: the second must not inherit the first's text.
    That leak would make a whole run's numbers meaningless, and it would show up
    as candidates that all score alike."""
    first = cold_turn()
    monkeypatch.setattr(llm, "complete", first)
    with overrides.using(overrides.Overrides(prompts={"explore": CANDIDATE})):
        await run()

    second = cold_turn()
    monkeypatch.setattr(llm, "complete", second)
    await run()

    def explore_system(model: ScriptedModel) -> str:
        return next(c for c in model.calls if c.get("node") == "explore")["system"]

    assert explore_system(first).startswith(CANDIDATE)
    assert explore_system(second).startswith(prompts._loaded()["explore"])
