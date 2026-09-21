"""The probe gate: the thing standing between a good score and a bad prompt.

This is the most load-bearing code in `tools/gepa/`, and it is the part GEPA cannot
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

# See tests/test_gepa_adapter.py — `gepa` is a dependency group, not a dev dep.
pytest.importorskip("gepa", reason="uv run --group gepa")

from app import llm  # noqa: E402
from tools.gepa.adapter import COMPONENT, Loop  # noqa: E402
from tools.gepa.gates import probe_gate, turn_gate  # noqa: E402
from tests.test_gepa_adapter import CENSUS_OUTPUT, GOOD_OUTPUT  # noqa: E402

SEED = "the seed instruction, which honours every invariant"
DEGENERATE = "DEGENERATE: record the row counts you were told"
ALSO_FINE = "a differently worded instruction that also honours them"


@dataclass
class FakeResult:
    """Just enough GEPAResult for the gate: a pool and its val scores."""

    candidates: list[dict[str, str]]
    val_aggregate_scores: list[float] = field(default_factory=list)


def gate(loop, result, seed: str):
    """`probe_gate` takes the whole seed candidate, as every target's does."""
    return probe_gate(loop, result, {COMPONENT: seed}, node="extract")


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
    gate(loop, FakeResult([{COMPONENT: SEED}], [1.0]), SEED)
    err = capsys.readouterr().err
    assert "seed passes 4/4" in err


def test_a_candidate_that_regresses_a_probe_is_discarded(loop, scripted, capsys):
    survivors = gate(
        loop,
        FakeResult(
            [{COMPONENT: SEED}, {COMPONENT: DEGENERATE}],
            # The degenerate candidate scores *better* on the trainset metric.
            # That is the scenario the gate exists for, not a hypothetical.
            [0.80, 0.95],
        ),
        SEED,
    )

    assert survivors == [], "a probe regression is disqualifying at any score"
    err = capsys.readouterr().err
    assert "DISCARDED" in err
    assert "census" in err
    # The reason has to be readable, because a human decides what to do next.
    assert "Neither kind is a census" in err


def test_a_good_candidate_survives_and_carries_its_score(loop, scripted):
    survivors = gate(
        loop,
        FakeResult(
            [{COMPONENT: SEED}, {COMPONENT: DEGENERATE}, {COMPONENT: ALSO_FINE}],
            [0.80, 0.95, 0.88],
        ),
        SEED,
    )

    assert [s.candidate[COMPONENT] for s in survivors] == [ALSO_FINE]
    assert survivors[0].score == 0.88


def test_survivors_come_back_best_first(loop, scripted):
    a, b = ALSO_FINE, ALSO_FINE + " (a second wording)"
    survivors = gate(
        loop,
        FakeResult([{COMPONENT: SEED}, {COMPONENT: a}, {COMPONENT: b}], [0.5, 0.6, 0.9]),
        SEED,
    )
    assert [s.score for s in survivors] == [0.9, 0.6]


def test_the_seed_itself_is_never_offered_as_a_candidate(loop, scripted):
    survivors = gate(loop, FakeResult([{COMPONENT: SEED}], [1.0]), SEED)
    assert survivors == []


def test_a_much_shorter_candidate_is_flagged_for_reading(loop, scripted, capsys):
    """Not rejected. ANSWER_SYSTEM's comment records this team shortening a
    prompt on purpose and measuring the result — brevity can be right. But a
    candidate that won by deleting most of the instruction must be *seen*."""
    terse = "DEGENER"  # short, and not the degenerate behaviour trigger
    survivors = gate(
        loop, FakeResult([{COMPONENT: SEED}, {COMPONENT: terse}], [0.5, 0.99]), SEED
    )

    assert survivors, "brevity alone is not disqualifying"
    assert "read the diff closely" in capsys.readouterr().err


# --------------------------------------------------- one command, needing none


def test_a_leftover_run_dir_is_wiped_rather_than_refused(tmp_path, monkeypatch):
    """GEPA resumes from run_dir without saying so, and its state is keyed to
    the seed and trainset it started from — so the leftovers of one run are
    never what the next one wants. Refusing made `make gepa-extract` a target
    that hands you a `rm -rf` and stops."""
    from click.testing import CliRunner

    from app import tracing
    from tools.gepa import cli as gepa

    monkeypatch.setattr(gepa, "OUT", tmp_path)
    monkeypatch.setattr(tracing, "enabled", lambda: False)
    stale = tmp_path / "run" / "extract"
    stale.mkdir(parents=True)
    (stale / "state.bin").write_text("a previous run")

    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert not stale.exists(), "the previous run's state must not survive"
    # Then straight on to the corpus, which is the next thing it does itself.
    assert "nothing to harvest" in result.output


def test_resume_keeps_the_run_dir_and_the_corpus_it_belongs_to(tmp_path, monkeypatch):
    """The one escape hatch, and it has to keep both: GEPA's state references
    its trainset, so resuming onto a re-harvested corpus would be a run made of
    two different populations."""
    from click.testing import CliRunner

    from app import tracing
    from tools.gepa import cli as gepa

    monkeypatch.setattr(gepa, "OUT", tmp_path)
    monkeypatch.setattr(
        tracing, "enabled", lambda: pytest.fail("--resume must not harvest")
    )
    kept = tmp_path / "run" / "extract"
    kept.mkdir(parents=True)
    (tmp_path / "extract.jsonl").write_text("")

    CliRunner().invoke(gepa.cli, ["extract", "--resume"])

    assert kept.exists()


def test_nothing_reaches_the_search_before_there_is_a_corpus(tmp_path, monkeypatch):
    """Whatever it refuses over, it refuses before a token is spent: `apply`
    used to check its preconditions at the end of a ten-minute search."""
    from click.testing import CliRunner

    from app import tracing
    from tools.gepa import cli as gepa

    monkeypatch.setattr(gepa, "OUT", tmp_path)
    monkeypatch.setattr(tracing, "enabled", lambda: False)
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: pytest.fail("the search was reached")
    )

    assert CliRunner().invoke(gepa.cli, ["extract"]).exit_code != 0


# ------------------------------------------------- the gate on known answers
#
# `turn_gate` reads GEPA's own per-case validation scores rather than re-running
# anything, so these need no model, no database and no loop. That is unusually
# lucky and worth exploiting: the gate that decides what gets promoted is the
# cheapest thing in this directory to test.


@dataclass
class TurnResult:
    """Enough GEPAResult for the turn gate: a pool, its means, and per case."""

    candidates: list[dict[str, str]]
    val_aggregate_scores: list[float] = field(default_factory=list)
    val_subscores: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class FakeCase:
    name: str
    question: str


CASES = [
    FakeCase("customers_active", "how many customers do we have?"),
    FakeCase("customers_west", "how many customers are in the west region?"),
    FakeCase("revenue_total", "what was our total revenue?"),
]

SEED_TOOLS = {"list_tables": "the seed description"}
RIVAL = {"list_tables": "a rival description"}


def scores(*values: float) -> dict[str, float]:
    return dict(zip([c.name for c in CASES], values))


def test_a_candidate_that_lost_a_case_the_seed_answered_is_discarded(capsys):
    """CHALLENGE item 5, and the reason the docstring there says "do not
    average". This candidate's mean is higher — 0.60 against 0.47 — and it got
    a question wrong that the seed got right. A mean-maximising search takes
    that trade every time, which is exactly why the gate is outside the
    objective.
    """
    survivors = turn_gate(
        None,
        TurnResult(
            [SEED_TOOLS, RIVAL],
            [0.47, 0.60],
            [scores(0.9, 0.5, 0.0), scores(0.9, 0.0, 0.9)],
        ),
        SEED_TOOLS,
        cases=CASES,
    )

    assert survivors == [], "a lost case is disqualifying at any score"
    err = capsys.readouterr().err
    assert "DISCARDED" in err
    # The question, not the case id: a person decides what to do next.
    assert "how many customers are in the west region?" in err


def test_a_candidate_that_lost_nothing_survives_with_its_score():
    """The other half. It gains a case and loses none, which is the shape a
    promotion is supposed to have."""
    survivors = turn_gate(
        None,
        TurnResult(
            [SEED_TOOLS, RIVAL],
            [0.47, 0.70],
            [scores(0.9, 0.5, 0.0), scores(0.9, 0.6, 0.6)],
        ),
        SEED_TOOLS,
        cases=CASES,
    )

    assert [s.candidate for s in survivors] == [RIVAL]
    assert survivors[0].score == 0.70


def test_a_case_the_seed_also_failed_is_not_a_regression():
    """The seed never answered `revenue_total`, so failing it cannot be a thing
    this candidate broke. Otherwise a permanently hard case would empty the pool
    on every run and the gate would be saying nothing."""
    survivors = turn_gate(
        None,
        TurnResult(
            [SEED_TOOLS, RIVAL],
            [0.47, 0.50],
            [scores(0.9, 0.5, 0.0), scores(0.9, 0.6, 0.0)],
        ),
        SEED_TOOLS,
        cases=CASES,
    )

    assert [s.candidate for s in survivors] == [RIVAL]


def test_a_candidate_with_no_validation_row_is_discarded_rather_than_assumed():
    """Unscored is not the same as unregressed. Keeping it would promote a
    candidate nothing ever checked."""
    survivors = turn_gate(
        None,
        TurnResult([SEED_TOOLS, RIVAL], [0.47, 0.99], [scores(0.9, 0.5, 0.0)]),
        SEED_TOOLS,
        cases=CASES,
    )

    assert survivors == []


def test_the_seed_is_never_offered_as_its_own_improvement():
    survivors = turn_gate(
        None,
        TurnResult([SEED_TOOLS], [0.47], [scores(0.9, 0.5, 0.0)]),
        SEED_TOOLS,
        cases=CASES,
    )

    assert survivors == []


def test_survivors_come_back_best_first():
    third = {"list_tables": "a third description"}
    survivors = turn_gate(
        None,
        TurnResult(
            [SEED_TOOLS, RIVAL, third],
            [0.40, 0.60, 0.90],
            [scores(0.9, 0, 0), scores(0.9, 0.5, 0), scores(0.9, 0.9, 0.9)],
        ),
        SEED_TOOLS,
        cases=CASES,
    )

    assert [s.score for s in survivors] == [0.90, 0.60]


def test_a_seed_that_never_reached_the_valset_says_so(capsys):
    """Then there is nothing to have regressed against. Keeping the pool and
    saying the gate is not checking beats an empty result that looks like every
    candidate failed."""
    survivors = turn_gate(
        None, TurnResult([RIVAL], [0.6], [scores(0.9, 0.9, 0.9)]), SEED_TOOLS,
        cases=CASES,
    )

    assert [s.candidate for s in survivors] == [RIVAL]
    assert "the gate is not checking" in capsys.readouterr().err
