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


# --------------------------------------------------------------- the artifact


def test_every_tool_gets_a_section_even_the_ones_nobody_mutated():
    """Round-robin mutates one component per iteration, so a winner is usually
    three untouched descriptions and one edit. The document is the complete set:
    a promotion that updates three and leaves one stale is exactly what a
    per-component format would invite."""
    rendered = targets.TOOLS.render({"list_tables": "a new description"})

    for name in TOOL_NAMES:
        assert f"## {name}\n" in rendered
    assert "a new description" in rendered
    assert rendered.endswith("\n") and not rendered.endswith("\n\n")


def test_a_description_cannot_forge_a_section_boundary():
    """The format is `## <tool name>` per section, so a candidate whose text
    starts a line with a hash would split its own section in two and whoever
    promotes it would paste half."""
    rendered = targets.TOOLS.render({"list_tables": "## describe_table\nsneaky"})

    assert rendered.count("## describe_table") == 1


def test_the_extract_artifact_is_byte_for_byte_what_it_always_was():
    """The refactor must not have changed what `make gepa-extract > new.md`
    writes. This is the assertion that says so."""
    candidate = {"extract": "  the winning prompt  \n\n"}

    assert targets.EXTRACT.render(candidate) == "the winning prompt\n"


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
    # 60 rollouts, per CHALLENGE's own arithmetic. The command's old default of
    # 150 would be 1.7 million tokens.
    assert targets.TOOLS.budget == 60
    assert targets.EXTRACT.rollout_tokens == 0, "one model call — do not ask"


def test_each_target_names_its_own_objectives():
    """The Pareto front's axes are the metric's terms, and the two metrics do
    not share terms. Reading one metric's weights for every target put nan in
    four of five columns and nobody on the front — found by running it."""
    from tools.gepa import metric_extract, metric_turn

    assert targets.EXTRACT.weights() == metric_extract.WEIGHTS
    assert targets.TOOLS.weights() == metric_turn.WEIGHTS
    assert targets.CONFIG.weights() == metric_turn.WEIGHTS
    assert set(targets.EXTRACT.weights()) != set(targets.TOOLS.weights())
    assert targets.EXTRACT.legend() == metric_extract.LEGEND
    assert targets.TOOLS.legend() == metric_turn.LEGEND
    assert targets.CONFIG.legend() == metric_turn.LEGEND


def test_every_term_has_its_words_and_no_words_are_orphaned():
    """The legend above a front is one line per term. A seventh term without
    a line would print its weight and nothing else, and a line for a term
    that no longer exists would describe nothing."""
    from tools.gepa import metric_extract, metric_turn

    for metric in (metric_extract, metric_turn):
        assert set(metric.LEGEND) == set(metric.WEIGHTS), metric.__name__
        assert all(metric.LEGEND.values()), "an empty line is not a legend"


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


def test_the_config_winner_is_normalised_yaml_that_pastes_into_the_file():
    """Whatever whitespace the reflection model used, stdout is the block as
    config.yaml writes it: one node per entry, turn order, one trailing
    newline."""
    rendered = targets.CONFIG.render(
        {"efforts": "answer: {effort: low}\nplan:   {effort: high}\n"}
    )

    assert rendered == "plan:\n  effort: high\nanswer:\n  effort: low\n"


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
    assert targets.CONFIG.budget == targets.TOOLS.budget == 60
    assert targets.CONFIG.rollout_tokens == targets.TOOLS.rollout_tokens
    assert targets.CONFIG.check is not None
    assert targets.CONFIG.templates is not None
    assert "config" not in targets.UNWIRED
