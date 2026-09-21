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
    # 60 rollouts, per CHALLENGE's own arithmetic. The command's old default of
    # 150 would be 1.7 million tokens.
    assert targets.TOOLS.budget == 60
    assert targets.EXTRACT.rollout_tokens == 0, "one model call — do not ask"


def test_every_target_offers_a_cheap_check():
    """--probe-only is how you find out something is wrong for the price of a
    few model calls rather than a whole search. A target without one silently
    turns that into an error message."""
    for target in targets.TARGETS.values():
        assert target.check is not None, target.name
