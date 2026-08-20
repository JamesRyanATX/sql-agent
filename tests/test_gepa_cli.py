"""The command's contract: stdout is the prompt, and nothing else ever is.

A stray `click.echo` without `err=True` or a GEPA log line both produce the same
failure — a prompt file with commentary in it, which the next run reads back as
the seed. Invisible until someone pastes a progress bar into `extract.md`.

Scripted model, no database, no search: `_search` is replaced by a pool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

pytest.importorskip("gepa", reason="uv run --group gepa")

from app import llm, prompts, tracing  # noqa: E402
from click.testing import CliRunner  # noqa: E402
from tools.gepa import cli as gepa  # noqa: E402
from tools.gepa.adapter import COMPONENT  # noqa: E402
from tools.gepa.cases import ExtractCase  # noqa: E402
from tests.test_gepa_adapter import GOOD_OUTPUT, SQL  # noqa: E402

BETTER = "a better instruction, which also honours every invariant"


@dataclass
class FakeResult:
    candidates: list[dict[str, str]]
    val_aggregate_scores: list[float] = field(default_factory=list)
    total_metric_calls: int = 42


@pytest.fixture
def well_behaved(monkeypatch):
    """A model that honours the invariants whatever prompt it is given."""

    async def fake_complete(**kwargs):
        return llm.Result(
            text=json.dumps(GOOD_OUTPUT),
            stop_reason="end_turn", tokens_in=400, tokens_out=120,
        )

    monkeypatch.setattr(llm, "complete", fake_complete)


@pytest.fixture
def a_run(monkeypatch, tmp_path, well_behaved):
    """Everything up to the gate, faked: a corpus on disk and a pool of two."""
    seed = prompts.get("extract")
    monkeypatch.setattr(gepa, "OUT", tmp_path)
    monkeypatch.setattr(tracing, "enabled", lambda: False)
    monkeypatch.setattr(
        gepa, "_corpus", lambda node, **kwargs: [_case(i) for i in range(8)]
    )
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            [{COMPONENT: seed}, {COMPONENT: BETTER}], [0.80, 0.95]
        ),
    )
    return seed


def _case(i: int) -> ExtractCase:
    return ExtractCase.authored(
        name=f"case-{i}",
        question="what was revenue last quarter?",
        sql=SQL,
        findings="[(1234.50,)]",
    )


def test_stdout_is_the_prompt_and_stderr_is_everything_else(a_run):
    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == 0
    assert result.stdout == BETTER + "\n", "stdout must be prose and nothing else"
    # Not lost — a redirect leaves it on the terminal.
    assert "candidate 1" in result.stderr
    assert "the invariants this prompt is the only home for" in result.stderr


def test_a_node_with_no_metric_exits_two_and_says_what_is_missing(monkeypatch):
    """A target that exists because a pattern rule cannot know which nodes are
    wired. What it must not do is look like a broken run."""
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: pytest.fail("a search was started")
    )

    result = CliRunner().invoke(gepa.cli, ["answer"])

    assert result.exit_code == gepa.UNWIRED_NODE
    assert result.stdout == "", "nothing on stdout means nothing to paste"
    assert "no metric" in result.stderr
    assert "decided the other way" in result.stderr, "the reason, not just a no"
    assert "tests/probes/answer/*.json" in result.stderr


def test_a_node_that_is_not_a_prompt_is_a_usage_error():
    result = CliRunner().invoke(gepa.cli, ["extarct"])

    assert result.exit_code == 1
    assert "no prompt named 'extarct'" in result.output


def test_a_winner_that_scored_below_the_seed_is_not_offered(a_run, monkeypatch):
    """Observed: a run scored the seed 0.959 and its best survivor 0.928.
    Clearing every probe is not the same as being better."""
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            [{COMPONENT: a_run}, {COMPONENT: BETTER}], [0.959, 0.928]
        ),
    )

    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == gepa.NO_IMPROVEMENT
    assert result.stdout == ""
    assert "nothing better" in result.stderr


def test_an_empty_pool_is_reported_as_a_result_rather_than_a_crash(a_run, monkeypatch):
    """A seed at the top of the metric is never mutated. That says something
    about the metric, not about the run."""
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: FakeResult([{COMPONENT: a_run}], [1.0])
    )

    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == gepa.NO_IMPROVEMENT
    assert result.stdout == ""
    assert "GEPA proposed nothing" in result.stderr
