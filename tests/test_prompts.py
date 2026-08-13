"""The prompt seam: what an optimiser writes a candidate through.

Not tests of what the prompts *say* — that is
[tests/test_prompt_invariants.py](tests/test_prompt_invariants.py), which spends
tokens and is marked `live`. These are about the mechanism: the override
resolves, it resolves once, and a typo cannot silently do nothing.
"""

from __future__ import annotations

import pytest

from app import graph, prompts
from app.settings import settings


@pytest.fixture
def prompt_dir(tmp_path, monkeypatch):
    """Point `prompt_dir` at a scratch directory, both caches cleared."""
    monkeypatch.setenv("PROMPT_DIR", str(tmp_path))
    settings.cache_clear()
    prompts._loaded.cache_clear()
    yield tmp_path
    monkeypatch.delenv("PROMPT_DIR", raising=False)
    settings.cache_clear()
    prompts._loaded.cache_clear()


def test_every_node_that_takes_a_prompt_can_get_one():
    """The keys are the `node=` labels llm.complete records as generation names.

    One vocabulary across the graph, the trace and an override file, so a
    harvested case and the prompt that produced it name the same thing.
    `explore` covers `explore.summary` too: one prompt, two calls.
    """
    for node in ("explore", "plan", "generate_sql", "fix", "extract", "answer"):
        assert prompts.get(node).strip()


def test_the_deployed_state_is_the_prose_in_git():
    """Empty `prompt_dir` is what ships. An override is a development tool."""
    assert settings().prompt_dir == ""
    assert prompts.get("extract") == prompts.EXTRACT_SYSTEM


def test_an_override_replaces_one_block_and_leaves_the_rest(prompt_dir):
    (prompt_dir / "extract.txt").write_text("write down nothing at all\n")

    assert prompts.get("extract") == "write down nothing at all"
    # Trailing newline stripped: a text file ends with one and a prompt does not.
    assert not prompts.get("extract").endswith("\n")
    assert prompts.get("plan") == prompts.PLAN_SYSTEM


def test_a_txt_naming_no_prompt_is_an_error_rather_than_a_no_op(prompt_dir):
    """A typo'd override that silently changes nothing means an optimisation
    run that measures the seed and reports it as a candidate improvement."""
    (prompt_dir / "extrct.txt").write_text("oops")

    with pytest.raises(ValueError, match="naming no prompt"):
        prompts.get("extract")


def test_a_prompt_dir_that_is_not_a_directory_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPT_DIR", str(tmp_path / "nope"))
    settings.cache_clear()
    prompts._loaded.cache_clear()
    try:
        with pytest.raises(ValueError, match="not a directory"):
            prompts.get("plan")
    finally:
        monkeypatch.delenv("PROMPT_DIR", raising=False)
        settings.cache_clear()
        prompts._loaded.cache_clear()


def test_the_blocks_resolve_once_per_process(prompt_dir):
    """Memoised deliberately, and this is the invariant it protects.

    `graph.plan` puts its system block behind an Anthropic cache breakpoint on
    the promise that the block varies with connection_id alone. A prompt re-read
    per turn could change between two turns of one server's life, and the only
    symptom would be T2 quietly costing more.
    """
    first = prompts.get("plan")
    (prompt_dir / "plan.txt").write_text("a different instruction entirely")

    assert prompts.get("plan") == first, "a live edit must not reach a running turn"

    prompts._loaded.cache_clear()
    assert prompts.get("plan") == "a different instruction entirely"


def test_the_fingerprint_moves_only_when_the_prose_does(prompt_dir):
    """What a harvest filters on to keep round two off round one's output."""
    before = prompts.fingerprint()
    assert set(before) == {"explore", "plan", "generate_sql", "fix", "extract", "answer"}
    assert all(len(v) == 8 for v in before.values())

    (prompt_dir / "extract.txt").write_text("something else")
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
