"""The prompt seam: where the prose lives and what happens when it doesn't.

Not tests of what the prompts *say* — that is
[tests/test_prompt_invariants.py](tests/test_prompt_invariants.py), which spends
tokens and is marked `live`. These are about the mechanism: the files resolve,
they resolve once, and nothing here can silently do nothing.
"""

from __future__ import annotations

import pytest

from app import graph, prompts
from app.settings import settings


@pytest.fixture
def prompt_dir(tmp_path, monkeypatch):
    """A scratch CONFIG_DIR holding a full set of prompts, both caches cleared.

    Full rather than empty: the files are the source of truth now, so a
    directory missing one is an error rather than a fallback, and a fixture that
    started empty would be testing that error and nothing else.
    """
    directory = tmp_path / "prompts"
    directory.mkdir()
    for node in prompts.NODES:
        (directory / f"{node}.md").write_text(f"instructions for {node}\n")

    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))
    settings.cache_clear()
    prompts._loaded.cache_clear()
    yield directory
    monkeypatch.delenv("CONFIG_DIR", raising=False)
    settings.cache_clear()
    prompts._loaded.cache_clear()


def test_every_node_that_takes_a_prompt_has_one():
    """The names are the `node=` labels llm.complete records as generation names.

    One vocabulary across the graph, the trace, the config file and the prompt
    file, so a harvested case and the prose that produced it name the same
    thing. `explore` covers `explore.summary` too: one prompt, two calls.
    """
    assert set(prompts.NODES) == {
        "explore", "plan", "generate_sql", "fix", "extract", "answer"
    }
    for node in prompts.NODES:
        assert prompts.get(node).strip()


def test_the_tracked_files_are_what_ships():
    """No constants, no override layer: config/prompts/ is the prose."""
    assert settings().config_dir == "config"
    assert prompts.directory().is_dir()
    on_disk = (prompts.directory() / "extract.md").read_text(encoding="utf-8")
    assert prompts.get("extract") == on_disk.strip()


def test_a_file_is_the_whole_prompt(prompt_dir):
    """No frontmatter, no separator, no header — what is sent is what is there.

    Only the surrounding whitespace goes, because a text file ends with a
    newline and a prompt does not.
    """
    (prompt_dir / "extract.md").write_text("\n  write down nothing at all  \n\n")

    assert prompts.get("extract") == "write down nothing at all"
    assert prompts.get("plan") == "instructions for plan"


def test_a_md_naming_no_prompt_is_an_error_rather_than_a_no_op(prompt_dir):
    """A typo'd file that silently changes nothing means an optimisation run
    that measures the seed and reports it as a candidate improvement."""
    (prompt_dir / "extrct.md").write_text("oops")
    prompts._loaded.cache_clear()

    with pytest.raises(ValueError, match="naming no prompt"):
        prompts.get("extract")


def test_the_notes_file_is_the_one_tolerated_exception(prompt_dir):
    """README.md is where the comments above the old constants went."""
    (prompt_dir / prompts.NOTES).write_text("# The prompts\n\nnotes about them\n")
    prompts._loaded.cache_clear()

    assert prompts.get("extract") == "instructions for extract"
    assert "notes" not in prompts.get("extract")


def test_a_missing_prompt_is_an_error(prompt_dir):
    """There is no constant left to fall back to, and silence would ship a
    node with no instructions at all."""
    (prompt_dir / "extract.md").unlink()
    prompts._loaded.cache_clear()

    with pytest.raises(ValueError, match="extract.md"):
        prompts.get("plan")


def test_an_empty_prompt_is_an_error(prompt_dir):
    """An empty file is the placeholder somebody forgot to fill."""
    (prompt_dir / "answer.md").write_text("   \n")
    prompts._loaded.cache_clear()

    with pytest.raises(ValueError, match="answer.md"):
        prompts.get("answer")


def test_a_config_dir_with_no_prompts_directory_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "nope"))
    settings.cache_clear()
    prompts._loaded.cache_clear()
    try:
        with pytest.raises(ValueError, match="no prompt directory"):
            prompts.get("plan")
    finally:
        monkeypatch.delenv("CONFIG_DIR", raising=False)
        settings.cache_clear()
        prompts._loaded.cache_clear()


def test_the_blocks_resolve_once_per_process(prompt_dir):
    """Memoised deliberately, and this is the invariant it protects.

    `graph.plan` puts its system block behind an Anthropic cache breakpoint on
    the promise that the block varies with connection_id alone. A prompt re-read
    per turn could change between two turns of one server's life, and the only
    symptom would be T2 quietly costing more. This is also why editing a file
    needs a restart, and why config/ is in uvicorn's --reload-dir.
    """
    first = prompts.get("plan")
    (prompt_dir / "plan.md").write_text("a different instruction entirely")

    assert prompts.get("plan") == first, "a live edit must not reach a running turn"

    prompts._loaded.cache_clear()
    assert prompts.get("plan") == "a different instruction entirely"


def test_the_fingerprint_moves_only_when_the_prose_does(prompt_dir):
    """What a harvest filters on to keep round two off round one's output."""
    before = prompts.fingerprint()
    assert set(before) == set(prompts.NODES)
    assert all(len(v) == 8 for v in before.values())

    (prompt_dir / "extract.md").write_text("something else")
    prompts._loaded.cache_clear()
    after = prompts.fingerprint()

    assert after["extract"] != before["extract"]
    assert {k: v for k, v in after.items() if k != "extract"} == {
        k: v for k, v in before.items() if k != "extract"
    }


# ------------------------------------------------- what the harness replays


def test_the_extract_message_carries_its_anchors():
    """`optim/` replays a recorded message verbatim and splits the SQL back out
    of it with these. Duplicating the literals there is how they drift."""
    message = graph.extract_message(
        question="how many customers do we have?",
        sql="SELECT count(*) FROM customer WHERE deleted_at IS NULL",
        findings="customer holds 2,000 rows",
        cache=[{"name": "revenue", "claim": "qty x price"}],
    )

    assert graph.EXTRACT_SQL_ANCHOR in message
    assert graph.EXTRACT_FILED_ANCHOR in message
    recovered = message.split(graph.EXTRACT_SQL_ANCHOR, 1)[1].split("\n\n", 1)[0]
    assert recovered == "SELECT count(*) FROM customer WHERE deleted_at IS NULL"


def test_a_cold_cache_files_nothing_and_says_nothing_about_filing():
    message = graph.extract_message(
        question="q", sql="SELECT 1", findings="f", cache=[]
    )
    assert graph.EXTRACT_FILED_ANCHOR not in message


def test_entries_from_applies_the_verification_gate():
    """The gate lives in one place so the optimiser scores what production
    writes, not a re-implementation that agrees with it until someone edits."""
    sql = "SELECT sum(qty * price) FROM order_item WHERE status <> 'cancelled'"
    entries = graph.entries_from(
        [
            {"kind": "recipe", "name": "revenue", "claim": "qty x price",
             "sql_fragment": "sum(qty * price)", "tables": ["order_item"]},
            {"kind": "recipe", "name": "pending", "claim": "excludes pending",
             "sql_fragment": "status <> 'pending'", "tables": ["orders"]},
            {"kind": "schema_fact", "name": "order_item", "claim": "one row per line",
             "tables": []},
        ],
        sql,
        fallback_tables=["order_item"],
    )

    assert [e.verified for e in entries] == [True, False, False]
    # A schema_fact has no fragment, so the gate has nothing to say about it.
    assert entries[2].sql_fragment is None
    # An omitted `tables` falls back to what the SQL actually named.
    assert entries[2].tables == ["order_item"]
