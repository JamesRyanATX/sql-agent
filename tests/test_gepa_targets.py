"""What a target promises the command, checked without spending anything.

`cli.py` is now target-generic, which means it can no longer notice that a
target got something wrong. If the seed dict does not match what the adapter
overrides, or the renderer drops a component, or a reflection template is
missing a placeholder, the symptom is a run that costs money and produces a
document with a hole in it.

All of it is structure, so none of it needs a model or a database.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("gepa", reason="uv run --group gepa")

from gepa.strategies.instruction_proposal import (  # noqa: E402
    InstructionProposalSignature,
)

from app import tools  # noqa: E402
from tools.gepa import targets  # noqa: E402

TOOL_NAMES = {"list_tables", "describe_table", "sample_column", "count_distinct"}


# ------------------------------------------------------------------ the seed


def test_the_tools_seed_is_exactly_what_the_model_is_sent_today():
    """The seed has to be the live text, or the first generation is measured
    against something the agent never ran. A fifth tool added to SCHEMAS without
    a thought for this would otherwise be searched around rather than searched.
    """
    seed = targets.TOOLS.seed()

    assert set(seed) == TOOL_NAMES
    assert seed == {s["name"]: s["description"] for s in tools.SCHEMAS}


def test_the_extract_seed_is_the_prompt_on_disk():
    from app import prompts

    assert targets.EXTRACT.seed() == {"extract": prompts.get("extract")}


# ------------------------------------------------------- reaching the graph


def test_a_tools_candidate_reaches_the_graph_as_tool_descriptions_only():
    """A search over descriptions that quietly also moved a prompt or an effort
    would be measuring something nobody asked about, and the run would still
    look fine."""
    applied = targets._tools_overrides({"list_tables": "something else"})

    assert applied.tools == {"list_tables": "something else"}
    assert applied.prompts == {}
    assert applied.efforts == {}


def test_a_candidate_cannot_reach_a_tool_that_does_not_exist():
    """`app/tools.py` ignores an unknown name on purpose, so a candidate naming
    a tool that is not there fails silently rather than loudly. That is right at
    run time and wrong here: the seed is what decides which names GEPA will ever
    propose, and it comes from SCHEMAS."""
    assert set(targets.TOOLS.seed()) <= {s["name"] for s in tools.SCHEMAS}


# ------------------------------------------------------- reflection templates


def test_every_tool_template_is_one_gepa_will_accept():
    """`gepa.optimize` validates these when the run is constructed, so a missing
    placeholder fails before a token is spent — but only if the dict reaches it.
    Checked here against GEPA's own validator rather than a copy of its rule."""
    for name, template in targets.TOOLS.templates().items():
        InstructionProposalSignature.validate_prompt_template(template)
        assert name in template, "a template that does not name its own tool"


def test_a_template_names_every_argument_the_tool_takes_and_no_others():
    """The template tells the reflection model it may not invent an argument.
    Listing one the tool does not take would teach exactly the mistake it exists
    to prevent, and leaving one out invites a description that never mentions
    it. Both are caught by reading the names out of `input_schema`, so the
    template cannot drift from the tool.
    """
    schemas = {s["name"]: s for s in tools.SCHEMAS}

    for name, template in targets.TOOLS.templates().items():
        args = set(schemas[name]["input_schema"].get("properties") or {})
        line = next(l for l in template.splitlines() if l.startswith("It takes"))
        listed = set(re.findall(r"`(\w+)`", line))

        assert listed == args, f"{name}: template says {listed}, tool takes {args}"
        if not args:
            assert line == "It takes no arguments."


def test_extract_uses_gepa_s_own_reflection_prompt():
    """One component of node instructions is what GEPA's generic template is
    for. A per-component one here would be prose for its own sake."""
    assert targets.EXTRACT.templates is None


# ------------------------------------------------------------- the registry


def test_a_target_is_either_searchable_or_has_a_reason():
    """`WIRED` is gone: being in TARGETS is what wired means. These two must not
    overlap, or a name resolves to a target and to an excuse at once."""
    assert not set(targets.TARGETS) & set(targets.UNWIRED)


def test_every_prompt_node_is_accounted_for():
    """A seventh prompt file added to config/prompts/ without a line in either
    registry would make `make gepa-<it>` say "no target named", which reads as a
    typo rather than as work nobody has done."""
    from app import prompts

    accounted = set(targets.TARGETS) | set(targets.UNWIRED)
    assert set(prompts.NODES) <= accounted


def test_a_whole_turn_target_says_what_a_rollout_costs():
    """The spend confirmation is keyed off this. A target that forgot it would
    silently stop asking before spending."""
    assert targets.TOOLS.rollout_tokens > 0
    assert targets.TOOLS.blurb


def test_every_target_says_what_it_is_for_a_newcomer():
    """The ABOUT section at the top of the front display: what the thing is
    responsible for, and what GEPA does to it. Two sentences, no jargon a
    line further down would have to explain."""
    for target in targets.TARGETS.values():
        assert target.about.count(". ") >= 1, f"{target.name}: two sentences"
        assert "GEPA optimizes" in target.about, target.name
    # 180 rollouts, which allows about twenty-two proposals. It was 60, which
    # allowed seven, and the run that used it kept one new candidate. The
    # default, so `make gepa-tools` needs no arguments.
    assert targets.TOOLS.budget == 180
    assert targets.EXTRACT.rollout_tokens == 0, "one model call — do not ask"


def test_each_target_says_what_its_score_means():
    """The SCORE section of the report prints this. It ends in a full stop,
    and every line of it is a statement of its own, not a sentence broken
    across lines."""
    for target in targets.TARGETS.values():
        text = target.score()
        assert text.endswith("."), target.name
        for line in text.splitlines():
            assert line.strip(), target.name
            assert line.count(". ") == 0, (target.name, line)


def test_the_extract_text_states_the_rule_one_statement_per_line():
    assert targets.EXTRACT.score() == (
        "The share of recipes whose SQL fragment is copied from the query that ran.\n"
        "A fragment that is only a table or function name does not count as copied.\n"
        "0 when the call failed or nothing was recorded.\n"
        "0 when no recipe was recorded.\n"
        "0 when a note records a count or a percentage.\n"
        "0 when a note takes a name already filed, or two notes share a name."
    )


def test_the_turn_sentence_is_built_from_the_turn_metric_weights(monkeypatch):
    """Both turn targets share the metric, so they share the sentence, and
    its numbers come from `metric_turn.WEIGHTS`: changing a weight changes
    the sentence, so the report cannot describe weights the metric no longer
    uses."""
    from tools.gepa import metric_turn

    assert targets.TOOLS.score() == targets.CONFIG.score()
    assert targets.TOOLS.score() == (
        "A right answer scores 0.60.\n"
        "  + tokens, against the seed's turn on the same question: 0.125 at "
        "the same spend, 0.25 at half or less, less the more it spends\n"
        "  + tool calls, on the same scale: 0.075 at the same count, 0.15 at "
        "half or fewer\n"
        "A wrong answer scores 0.60 times how closely its rows match the known "
        "answer.\n"
        "0 if the turn failed, wrote no SQL, or returned no rows."
    )

    monkeypatch.setitem(metric_turn.WEIGHTS, "cost", 0.30)
    assert "0.150 at the same spend, 0.30 at half or less" in targets.TOOLS.score()


def test_the_extract_adapter_refuses_a_prompt_past_the_growth_limit():
    from app import prompts

    adapter = targets.EXTRACT.adapter(None)
    assert adapter.max_chars == int(targets.MAX_GROWTH * len(prompts.get("extract")))


def test_every_target_offers_a_cheap_check():
    """--probe-only is how you find out something is wrong for the price of a
    few model calls rather than a whole search. A target without one silently
    turns that into an error message."""
    for target in targets.TARGETS.values():
        assert target.check is not None, target.name


# ------------------------------------------------------------------- config
#
# Six enum values the model never reads, as one YAML component. The gate is the
# config schema: a candidate goes through the same `Node` model the file does.


NODES = {"plan", "explore", "generate_sql", "fix", "extract", "answer"}


def test_the_config_seed_is_what_is_in_force_today():
    """Resolved, not read: a node the file leaves out is `medium` by default,
    and a seed that left it out would be searching around it."""
    from app.config import config

    seed = targets.CONFIG.seed()

    assert set(seed) == {"efforts"}
    parsed = targets._config_parse(seed["efforts"])
    assert set(parsed) == NODES
    assert parsed == {n: config().effort_for(n) for n in NODES}


def test_the_turn_nodes_are_the_prompt_nodes_in_turn_order():
    """Two lists of the same six names. `Config.NODES` adds `gepa`, which is the
    reflection model and not part of a turn."""
    from app import prompts

    assert set(targets.TURN_NODES) == set(prompts.NODES)
    assert targets.TURN_NODES[0] == "plan" and targets.TURN_NODES[-1] == "answer"


def test_the_effort_levels_are_the_ones_the_config_accepts():
    """Spelled in targets.py for the template; the truth is the Literal on
    `Node.effort`. This is the assertion that keeps them one list."""
    from typing import get_args

    from app.config import Node

    literal = next(
        a for a in get_args(Node.model_fields["effort"].annotation) if get_args(a)
    )
    assert tuple(get_args(literal)) == targets.EFFORT_LEVELS


def test_a_config_candidate_reaches_the_graph_as_efforts_only():
    applied = targets._config_overrides({"efforts": "explore:\n  effort: low\n"})

    assert applied.efforts == {"explore": "low"}
    assert applied.prompts == {}
    assert applied.tools == {}


@pytest.mark.parametrize(
    "text, reason",
    [
        ("- explore", "expected a mapping"),
        ("", "expected a mapping"),
        ("nope:\n  effort: low\n", "no node named nope"),
        ("plan:\n  effort: bananas\n", "Input should be"),
        ("plan:\n  effort: low\n  model:\n    model: gpt\n", "effort only"),
        ("plan: low\n", "one key, `effort`"),
        ("plan:\n  effort:\n", "has no value"),
        ("plan: [\n", "not YAML"),
    ],
)
def test_a_candidate_the_config_would_refuse_is_rejected_with_the_reason(text, reason):
    """The reason is what the reflection reads next round, so it has to say
    what the shape that runs is. Pydantic's own message for a bad level, the
    node list for a bad node, and "effort only" for a candidate that tried to
    choose a model."""
    from tools.gepa.metric_turn import Rejected

    with pytest.raises(Rejected, match=reason):
        targets._config_parse(text)


def test_none_is_refused_where_the_model_is_claude(monkeypatch):
    """The file validator's rule, applied to a candidate: on Opus, `none`
    disables thinking and a tool call becomes visible text that never runs. A
    candidate proposing it is rejected before a rollout, not scored as a broken
    graph after one."""
    from tools.gepa.metric_turn import Rejected

    monkeypatch.setattr(targets, "_is_claude", lambda node: True)
    with pytest.raises(Rejected, match="Claude has no such level"):
        targets._config_parse("explore:\n  effort: none\n")

    monkeypatch.setattr(targets, "_is_claude", lambda node: False)
    assert targets._config_parse("explore:\n  effort: none\n") == {"explore": "none"}


def test_the_config_template_is_one_gepa_will_accept_and_names_every_node():
    template = targets.CONFIG.templates()["efforts"]

    InstructionProposalSignature.validate_prompt_template(template)
    for node in NODES:
        assert f"- {node}" in template, f"the template does not describe {node}"
    assert "`effort`" in template
    assert "Do not name a model" in template


def test_the_template_tells_the_truth_about_none(monkeypatch):
    """Telling a reflection `none` is legal on Opus buys one rejected candidate
    per round. Telling it `none` is illegal on a model where it is the cheapest
    setting hides the setting that scored 10/19 in development."""
    monkeypatch.setattr(targets, "_is_claude", lambda node: True)
    on_claude = targets.CONFIG.templates()["efforts"]
    assert "cheapest first: low, medium, high, xhigh, max." in on_claude
    assert "rejected" in on_claude

    monkeypatch.setattr(targets, "_is_claude", lambda node: False)
    elsewhere = targets.CONFIG.templates()["efforts"]
    assert "cheapest first: none, low," in elsewhere
    assert "`none` is legal on this model" in elsewhere


def test_the_config_focus_is_the_per_node_ledger():
    from tools.gepa.replay_turn import TurnReplayed

    r = TurnReplayed(
        question="q", per_node={"explore": (500, 50)}, efforts={"explore": "high"}
    )

    (key, text), = targets._config_focus("efforts", r).items()
    assert "per node" in key
    assert "explore (high)" in text and "550 tokens" in text


def test_the_config_target_is_a_whole_turn_target():
    """Same budget, same rollout cost, same gate and check as `tools`: a
    rollout is a cold turn whichever component it is measuring."""
    assert targets.CONFIG.budget == targets.TOOLS.budget == targets.TURN_BUDGET
    assert targets.CONFIG.rollout_tokens == targets.TOOLS.rollout_tokens
    assert targets.CONFIG.check is not None
    assert targets.CONFIG.templates is not None
    assert "config" not in targets.UNWIRED
