"""The probe gate: the thing standing between a good score and a bad prompt.

This is the most load-bearing code in `optim/`, and it is the part GEPA cannot
do for us. Weights cannot express "never": a mean-maximising search will trade a
rare catastrophic failure for a broad small gain whenever the arithmetic allows,
and the failures the probes defend are exactly the ones the trainset metric
cannot see — a deleted census paragraph costs nothing today and poisons the
cache next week.

So the gate is structural, outside GEPA's objective, and tested here against a
scripted model rather than trusted to work when it matters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

# See tests/test_optim_adapter.py — `gepa` is a dependency group, not a dev dep.
pytest.importorskip("gepa", reason="uv run --group optim")

from app import llm  # noqa: E402
from optim.adapter import COMPONENT, Loop  # noqa: E402
from optim.run import _gate  # noqa: E402
from tests.test_optim_adapter import CENSUS_OUTPUT, GOOD_OUTPUT  # noqa: E402

SEED = "the seed instruction, which honours every invariant"
DEGENERATE = "DEGENERATE: record the row counts you were told"
ALSO_FINE = "a differently worded instruction that also honours them"


@dataclass
class FakeResult:
    """Just enough GEPAResult for the gate: a pool and its val scores."""

    candidates: list[dict[str, str]]
    val_aggregate_scores: list[float] = field(default_factory=list)


@pytest.fixture
def scripted(monkeypatch):
    """A model whose behaviour depends on the prompt it is given.

    That dependence is the whole point — a gate can only be tested by a
    candidate that genuinely behaves worse than the seed.
    """

    async def fake_complete(**kwargs):
        payload = CENSUS_OUTPUT if "DEGENERATE" in kwargs["system"] else GOOD_OUTPUT
        return llm.Result(
            text=json.dumps(payload),
            stop_reason="end_turn", tokens_in=400, tokens_out=120,
        )

    monkeypatch.setattr(llm, "complete", fake_complete)


@pytest.fixture
def loop():
    with Loop() as l:
        yield l


def test_the_seed_passes_every_probe_under_a_well_behaved_model(loop, scripted, capsys):
    """The gate's baseline. If the seed failed a probe, that probe could never
    disqualify anything — it would be excluded from `seed_passing` and every
    candidate would inherit the failure for free."""
    _gate(loop, FakeResult([{COMPONENT: SEED}], [1.0]), SEED, "extract")
    out = capsys.readouterr().out
    assert "seed passes 4/4" in out


def test_a_candidate_that_regresses_a_probe_is_discarded(loop, scripted, capsys):
    survivors = _gate(
        loop,
        FakeResult(
            [{COMPONENT: SEED}, {COMPONENT: DEGENERATE}],
            # The degenerate candidate scores *better* on the trainset metric.
            # That is the scenario the gate exists for, not a hypothetical.
            [0.80, 0.95],
        ),
        SEED,
        "extract",
    )

    assert survivors == [], "a probe regression is disqualifying at any score"
    out = capsys.readouterr().out
    assert "DISCARDED" in out
    assert "census" in out
    # The reason has to be readable, because a human decides what to do next.
    assert "Neither kind is a census" in out


def test_a_good_candidate_survives_and_carries_its_score(loop, scripted):
    survivors = _gate(
        loop,
        FakeResult(
            [{COMPONENT: SEED}, {COMPONENT: DEGENERATE}, {COMPONENT: ALSO_FINE}],
            [0.80, 0.95, 0.88],
        ),
        SEED,
        "extract",
    )

    assert [s["text"] for s in survivors] == [ALSO_FINE]
    assert survivors[0]["score"] == 0.88


def test_survivors_come_back_best_first(loop, scripted):
    a, b = ALSO_FINE, ALSO_FINE + " (a second wording)"
    survivors = _gate(
        loop,
        FakeResult([{COMPONENT: SEED}, {COMPONENT: a}, {COMPONENT: b}], [0.5, 0.6, 0.9]),
        SEED,
        "extract",
    )
    assert [s["score"] for s in survivors] == [0.9, 0.6]


def test_the_seed_itself_is_never_offered_as_a_candidate(loop, scripted):
    survivors = _gate(loop, FakeResult([{COMPONENT: SEED}], [1.0]), SEED, "extract")
    assert survivors == []


def test_a_much_shorter_candidate_is_flagged_for_reading(loop, scripted, capsys):
    """Not rejected. ANSWER_SYSTEM's comment records this team shortening a
    prompt on purpose and measuring the result — brevity can be right. But a
    candidate that won by deleting most of the instruction must be *seen*."""
    terse = "DEGENER"  # short, and not the degenerate behaviour trigger
    survivors = _gate(
        loop, FakeResult([{COMPONENT: SEED}, {COMPONENT: terse}], [0.5, 0.99]), SEED, "extract"
    )

    assert survivors, "brevity alone is not disqualifying"
    assert "read the diff closely" in capsys.readouterr().out


# ------------------------------------------------------- the resume guard


def test_a_leftover_run_dir_is_refused_rather_than_silently_resumed(tmp_path, monkeypatch):
    """GEPA resumes from run_dir without saying so, and its state is keyed to
    the seed it started from. A resume after editing app/prompts.py or
    re-harvesting would report a candidate as descended from a seed that never
    produced it — a result you would believe."""
    from click.testing import CliRunner

    from optim import run

    monkeypatch.setattr(run, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(run, "CORPUS", tmp_path / "extract.jsonl")
    (tmp_path / "extract.jsonl").write_text("")
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "state.bin").write_text("a previous run")

    result = CliRunner().invoke(run.cli, ["optimize", "--node", "extract"])

    assert result.exit_code != 0
    assert "would resume from it" in result.output
    assert "--resume" in result.output


def test_an_empty_run_dir_is_not_a_previous_run(tmp_path, monkeypatch):
    """The other direction: a directory GEPA created and left empty must not
    block the next run forever."""
    from click.testing import CliRunner

    from optim import run

    monkeypatch.setattr(run, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(run, "CORPUS", tmp_path / "missing.jsonl")
    (tmp_path / "run").mkdir()

    result = CliRunner().invoke(run.cli, ["optimize", "--node", "extract"])

    # Falls through to the corpus check, which is the *next* guard.
    assert "run `harvest` first" in result.output
