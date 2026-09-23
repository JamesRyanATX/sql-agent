"""The command's contract: stdout is the prompt, and nothing else ever is.

A stray `click.echo` without `err=True` or a GEPA log line both produce the same
failure — a prompt file with commentary in it, which the next run reads back as
the seed. Invisible until someone pastes a progress bar into `extract.md`.

Scripted model, no database, no search: `_search` is replaced by a pool.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import pytest

pytest.importorskip("gepa", reason="uv run --group gepa")

from app import llm, prompts, tracing  # noqa: E402
from click.testing import CliRunner  # noqa: E402
from tools.gepa import cli as gepa  # noqa: E402
from tools.gepa import metric_extract, targets  # noqa: E402
from tools.gepa.adapter import COMPONENT  # noqa: E402
from tools.gepa.cases import ExtractCase  # noqa: E402
from tests.test_gepa_adapter import GOOD_OUTPUT, SQL  # noqa: E402

BETTER = "a better instruction, which also honours every invariant"


@dataclass
class FakeResult:
    """Enough of `GEPAResult` for the command. The three per-objective fields
    default to None exactly as they do on the real type, which is how a run
    with no objective scores reaches `_pareto`."""

    candidates: list[dict[str, str]]
    val_aggregate_scores: list[float] = field(default_factory=list)
    total_metric_calls: int = 42
    val_aggregate_subscores: list[dict[str, float]] | None = None
    per_objective_best_candidates: dict[str, set[int]] | None = None
    objective_pareto_front: dict[str, float] | None = None
    # Per candidate, per validation case. The turn gate's whole evidence base.
    val_subscores: list[dict[str, float]] = field(default_factory=list)


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
    """Everything up to the gate, faked: a corpus in memory and a pool of two.

    The corpus is replaced on the target rather than on the command, because
    where a corpus comes from is a property of what is being searched. `Target`
    is frozen, so the registry entry is swapped for a copy.
    """
    seed = prompts.get("extract")
    monkeypatch.setattr(gepa, "OUT", tmp_path)
    monkeypatch.setattr(tracing, "enabled", lambda: False)
    monkeypatch.setitem(
        targets.TARGETS,
        "extract",
        dataclasses.replace(
            targets.EXTRACT, corpus=lambda **kwargs: [_case(i) for i in range(8)]
        ),
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
    assert "not searchable" in result.stderr
    assert "decided the other way" in result.stderr, "the reason, not just a no"
    assert "tools/gepa/targets.py" in result.stderr, "and where the work is"


def test_a_name_that_is_not_a_target_is_a_usage_error():
    """It lists both halves of the registry, because "extarct" is a typo and
    "config" is a thing somebody will reasonably try."""
    result = CliRunner().invoke(gepa.cli, ["extarct"])

    assert result.exit_code == 1
    assert "no target named 'extarct'" in result.output
    assert "extract" in result.output
    assert "generate_sql" in result.output


def test_probe_only_checks_the_seed_and_spends_nothing_else(well_behaved, monkeypatch):
    """The cheap pre-check. Four model calls against the prompt on disk, and it
    must not reach the search — the whole reason to have it is to find a broken
    invariant before paying for a run."""
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: pytest.fail("a search was started")
    )

    result = CliRunner().invoke(gepa.cli, ["extract", "--probe-only"])

    assert result.exit_code == 0
    assert result.stdout == "", "a check is not a promotion"
    assert "all 4 probes pass" in result.stderr


def test_probe_only_exits_one_when_the_seed_fails_an_invariant(monkeypatch):
    """A failing probe is a preflight failure, not a search result. Exit 1, and
    the reason readable, because a person decides what to do about it."""
    from tests.test_gepa_adapter import CENSUS_OUTPUT

    async def records_a_census(**kwargs):
        return llm.Result(
            text=json.dumps(CENSUS_OUTPUT),
            stop_reason="end_turn", tokens_in=400, tokens_out=120,
        )

    monkeypatch.setattr(llm, "complete", records_a_census)

    result = CliRunner().invoke(gepa.cli, ["extract", "--probe-only"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "probes failed" in result.stderr
    assert "Neither kind is a census" in result.stderr


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


# ----------------------------------------------------------------- the front
#
# The metric collapses five weighted terms into one number, which settles the
# grounding-against-cost trade on the reader's behalf. These say the front
# survives that collapse, lands in a file, and never reaches stdout.

# Read from the metric, not spelled again. A hardcoded list here went stale the
# moment a sixth term was added, and the failure was four front tests rather
# than anything pointing at the cause.
TERMS = tuple(metric_extract.WEIGHTS)


def _row(*values: float) -> tuple[float, ...]:
    """One candidate's terms, padded with 1.0 to however many the metric has.

    The last value is repeated into any extra terms so a candidate that gives
    up the final term keeps giving it up. Written this way because a literal
    five-tuple went stale the moment a sixth term was added.
    """
    padding = (values[-1],) * (len(TERMS) - len(values))
    return values + padding


def _subscores(*rows: tuple[float, ...]) -> list[dict[str, float]]:
    return [dict(zip(TERMS, row)) for row in rows]


@pytest.fixture
def a_front(a_run, monkeypatch):
    """Four candidates. 0 is the seed, 1 dominates it outright, 2 is worst
    everywhere, 3 buys the cheapest turn by giving up grounding."""
    pool = [
        {COMPONENT: a_run},
        {COMPONENT: BETTER},
        {COMPONENT: "worse on every term"},
        {COMPONENT: "terse, and ungrounded"},
    ]
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            pool,
            [0.80, 0.95, 0.40, 0.72],
            # Candidate 1 dominates the seed, 2 is worst everywhere, and 3
            # buys the last term outright by giving up the first. Padded to
            # however many terms the metric has, so the shape of the front —
            # not the term count — is what these assert.
            val_aggregate_subscores=_subscores(
                _row(0.80, 1.00, 1.00, 0.75, 0.90),
                _row(0.95, 1.00, 1.00, 0.90, 0.90),
                _row(0.50, 0.90, 0.90, 0.50, 0.50),
                _row(0.30, 1.00, 1.00, 1.00, 1.00),
            ),
            per_objective_best_candidates={
                TERMS[0]: {1},
                TERMS[1]: {0, 1, 3},
                TERMS[2]: {0, 1, 3},
                TERMS[-2]: {3},
                TERMS[-1]: {3},
            },
            objective_pareto_front=dict(
                zip(TERMS, _row(0.95, 1.0, 1.0, 1.0, 1.0))
            ),
        ),
    )
    return a_run


def test_the_front_is_written_without_touching_stdout(a_front, tmp_path):
    """The whole command's contract, applied to a second artifact: a file is
    written, a table is printed, and the prompt on stdout is untouched."""
    path = tmp_path / "artifacts" / "extract.pareto.json"

    result = CliRunner().invoke(gepa.cli, ["extract", "--pareto", str(path)])

    assert result.exit_code == 0
    assert result.stdout == BETTER + "\n", "the front must not reach stdout"

    document = json.loads(path.read_text())
    assert document["target"] == "extract"
    assert document["objectives"] == list(TERMS)
    assert document["pool"]["seed_index"] == 0
    # The whole argument for a front: no single candidate has all of it.
    assert document["best_per_objective"]["grounding"] == 0.95
    assert document["best_per_objective"]["cost"] == 1.0

    for term in TERMS:
        assert term in result.stderr


def test_a_dominated_candidate_is_not_on_the_front(a_front, tmp_path):
    """Candidate 2 is worse than candidate 1 on all five terms, so no reading
    of "better" puts it on a front. Candidate 3 is worse on grounding and
    better on shape and cost, so it belongs there — and it is the one that
    makes the trade-off visible, which is the point of showing a front at all.

    This is also what stops somebody quietly substituting GEPA's
    `per_objective_best_candidates`, which is a different, smaller set.
    """
    path = tmp_path / "front.json"

    CliRunner().invoke(gepa.cli, ["extract", "--pareto", str(path)])

    on_front = {entry["index"] for entry in json.loads(path.read_text())["front"]}
    assert 2 not in on_front, "dominated on every term"
    assert {1, 3} <= on_front, "neither beats the other on everything"


def test_the_front_is_byte_stable_across_runs(a_front, tmp_path):
    """The file is committed and read in a diff. GEPA hands back sets, and one
    unsorted set reaching the JSON makes every rerun a spurious change."""
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    CliRunner().invoke(gepa.cli, ["extract", "--pareto", str(first)])
    CliRunner().invoke(gepa.cli, ["extract", "--pareto", str(second)])

    assert first.read_bytes() == second.read_bytes()


def test_the_front_is_always_written_to_the_run_directory(a_front, tmp_path):
    """`--pareto` names a tracked copy. Not passing it must not mean losing the
    evidence — `out/` is where every other artifact of a run already lands."""
    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == 0
    assert json.loads(gepa.pareto("extract").read_text())["front"]


def test_a_result_with_no_objective_scores_says_so_rather_than_crashing(
    a_run, monkeypatch
):
    """An adapter that returns no per-term breakdown is a run with no front,
    not a run that failed. Says so in one line and writes nothing."""
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult([{COMPONENT: a_run}, {COMPONENT: BETTER}], [0.8, 0.9]),
    )

    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == 0
    assert result.stdout == BETTER + "\n"
    assert "no per-objective scores" in result.stderr
    assert not gepa.pareto("extract").exists()


def test_a_candidate_that_scored_nothing_anywhere_is_left_off(a_run, monkeypatch):
    """`metric_extract.score` returns empty terms on both its gates, so a
    candidate whose every validation case errored arrives with no terms at all.
    It is ineligible for the front rather than a KeyError.

    The run also ends NO_IMPROVEMENT, and the front is on disk anyway. That is
    the reason `_pareto` runs before the pool is judged: a run that produced
    nothing promotable is the one whose evidence is most worth keeping.
    """
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            [{COMPONENT: a_run}, {COMPONENT: BETTER}],
            [0.8, 0.0],
            val_aggregate_subscores=[dict.fromkeys(TERMS, 0.9), {}],
            per_objective_best_candidates={term: {0} for term in TERMS},
            objective_pareto_front=dict.fromkeys(TERMS, 0.9),
        ),
    )

    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == gepa.NO_IMPROVEMENT
    assert result.stdout == ""
    on_front = {
        e["index"] for e in json.loads(gepa.pareto("extract").read_text())["front"]
    }
    assert on_front == {0}


# --------------------------------------------------------- more than one thing
#
# The command is target-generic now. These are the properties that only show up
# once a second target exists, and the one about money.

TOOLS_SEED = None  # filled by the fixture; the live descriptions


@pytest.fixture
def a_tools_run(monkeypatch, tmp_path, well_behaved):
    """A finished tools search, faked from the gate backwards.

    The corpus is the real `demo/golden/` — it is tracked, it costs nothing to
    read, and using it means the split and the gate see the shape they will see
    for real.
    """
    seed = targets.TOOLS.seed()
    better = {**seed, "list_tables": "Every table, with its column count."}
    monkeypatch.setattr(gepa, "OUT", tmp_path)
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            [seed, better],
            [0.60, 0.80],
            val_subscores=[
                {"customers_active": 1.0, "customers_west": 0.0},
                {"customers_active": 1.0, "customers_west": 0.6},
            ],
        ),
    )
    return seed, better


def budget_of(monkeypatch, seed, argv: list[str]) -> int:
    """What the search was actually handed, as opposed to what the help says."""
    seen = {}

    def record(gepa_module, **kwargs):
        seen.update(kwargs)
        return FakeResult([seed], [1.0])

    monkeypatch.setattr(gepa, "_search", record)
    CliRunner().invoke(gepa.cli, argv)
    return seen["budget"]


def test_a_whole_turn_target_brings_its_own_budget(a_tools_run, monkeypatch):
    """`--budget` used to default to 150 for everything. A whole-turn rollout is
    about 11,500 tokens, so that default silently means 1.7 million — the most
    expensive thing an unnoticed default could do in this repo."""
    assert budget_of(monkeypatch, a_tools_run[0], ["tools", "--yes"]) == 60


def test_the_cheap_target_keeps_the_budget_it_always_had(a_run, monkeypatch):
    assert budget_of(monkeypatch, {COMPONENT: a_run}, ["extract"]) == 150


def test_an_explicit_budget_still_wins(a_tools_run, monkeypatch):
    assert budget_of(monkeypatch, a_tools_run[0], ["tools", "--budget", "4", "--yes"]) == 4


def test_an_expensive_run_says_what_it_will_spend_before_it_spends(a_tools_run):
    result = CliRunner().invoke(gepa.cli, ["tools", "--yes"])

    assert result.exit_code == 0
    assert "60 rollouts" in result.stderr
    assert "690,000 tokens" in result.stderr
    assert "components  4" in result.stderr
    # Not on stdout, where the artifact is.
    assert "rollouts" not in result.stdout


def test_an_expensive_run_refuses_a_pipe_rather_than_spending_into_it(
    a_tools_run, monkeypatch
):
    """`make gepa-tools > new.md` and every CI job: nobody is there to agree to
    the money, and the run would spend it anyway."""
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: pytest.fail("it spent the money")
    )

    result = CliRunner().invoke(gepa.cli, ["tools"])

    assert result.exit_code == 1
    assert "nobody at a terminal" in result.output
    assert result.stdout == ""


def test_a_cheap_run_does_not_ask(a_run, monkeypatch):
    """One model call a rollout. Asking about it would train people to type
    `--yes`, which is how a confirmation stops being read."""
    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == 0
    assert "Start?" not in result.stderr


def test_the_tools_winner_is_a_document_with_every_tool_in_it(a_tools_run):
    """Four components cannot be one prompt file, so stdout becomes a section
    per tool — all four, including the three nobody mutated."""
    result = CliRunner().invoke(gepa.cli, ["tools", "--yes"])

    assert result.exit_code == 0
    for name in a_tools_run[0]:
        assert f"## {name}\n" in result.stdout
    assert "Every table, with its column count." in result.stdout
    # The diff names where a person would go to change it.
    assert "app/tools.py SCHEMAS[list_tables]" in result.stderr


def test_an_unwired_target_gives_the_reason_that_is_still_true(monkeypatch):
    """`plan`'s old reason was that it needed outcome labelling. It has that
    now, so the reason had to be replaced rather than left to read like one."""
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: pytest.fail("a search was started")
    )

    result = CliRunner().invoke(gepa.cli, ["plan"])

    assert result.exit_code == gepa.UNWIRED_NODE
    assert "degenerate optimum" in result.stderr
    assert "outcome labelling" not in result.stderr


@pytest.fixture
def a_config_run(monkeypatch, tmp_path, well_behaved):
    """A finished effort search, faked from the gate backwards. The seed is the
    block in force; the winner moved one node."""
    seed = targets.CONFIG.seed()
    better = {"efforts": seed["efforts"].replace("explore:\n  effort: ", "explore:\n  effort: low  # was ", 1)}
    monkeypatch.setattr(gepa, "OUT", tmp_path)
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            [seed, better],
            [0.60, 0.80],
            val_subscores=[
                {"customers_active": 1.0, "customers_west": 0.0},
                {"customers_active": 1.0, "customers_west": 0.6},
            ],
        ),
    )
    return seed, better


def test_the_config_winner_is_a_yaml_block_that_pastes_into_the_file(a_config_run):
    """`make gepa-config > new.yaml`: six nodes, one key each, normalised —
    the comment the fake winner carried is gone, because it went through the
    same parser a candidate is scored through."""
    result = CliRunner().invoke(gepa.cli, ["config", "--yes"])

    assert result.exit_code == 0, result.stderr
    assert result.stdout.startswith("plan:\n  effort: ")
    assert "explore:\n  effort: low\n" in result.stdout
    assert "# was" not in result.stdout
    assert result.stdout.count("effort:") == 6
    # The diff names where a person would go, and the spend was said first.
    assert "config/config.yaml" in result.stderr
    assert "components  1" in result.stderr


def test_the_config_target_is_searchable_and_the_rest_still_say_why_not(monkeypatch):
    """`config` used to sit in UNWIRED with a reason that asked for exactly what
    the metric now has. A reason that has become false is worse than none.

    The check is stubbed: the real one is nineteen cold turns."""
    monkeypatch.setitem(
        targets.TARGETS,
        "config",
        dataclasses.replace(targets.CONFIG, check=lambda loop, yes: 0),
    )

    result = CliRunner().invoke(gepa.cli, ["config", "--probe-only", "--yes"])

    assert result.exit_code == 0, result.stderr
    assert "not searchable" not in result.stderr


def test_a_budget_smaller_than_the_valset_is_flagged_before_it_is_spent(
    a_tools_run, monkeypatch
):
    """GEPA scores the seed over the whole valset before proposing anything, so
    a budget that small buys a baseline and stops. The run then says GEPA
    proposed nothing, which reads as a saturated metric rather than as
    arithmetic. Observed on a real run at `--budget 5` against nine val cases.
    """
    result = CliRunner().invoke(
        gepa.cli, ["tools", "--budget", "3", "--val-fraction", "0.5", "--yes"]
    )

    assert "the baseline evaluation and nothing else" in result.stderr


# ------------------------------------------------------------------ overfit
#
# `--overfit N` is the run with the guards off, built to be shown failing:
# train and validate on N cases, let the gate speak and ignore it, then score
# the pool on the cases the search never saw. These say each guard really is
# off, and that the number the talk needs — the held-out score — is produced
# and written down.


def search_kwargs(monkeypatch, argv: list[str]) -> tuple[dict, object]:
    seen = {}

    def record(gepa_module, **kwargs):
        seen.update(kwargs)
        return FakeResult([kwargs["seed_candidate"]], [1.0])

    monkeypatch.setattr(gepa, "_search", record)
    return seen, CliRunner().invoke(gepa.cli, argv)


def test_overfit_trains_and_validates_on_the_same_n_cases(a_run, monkeypatch):
    """GEPA ranks parents by the validation set, so a validation set is not
    held out. The only honest shape is train == val == the N cases, with the
    rest never shown to the search. And the thin warning fires on those N,
    not on the corpus they were cut from."""
    seen, result = search_kwargs(monkeypatch, ["extract", "--overfit", "3"])

    assert len(seen["trainset"]) == 3
    assert seen["valset"] is seen["trainset"]
    assert seen["node"] == "extract-overfit", "its own run dir, not section 3's"
    assert "the seed over all 8 cases, to find the 3 it does worst on" in result.stderr
    assert "3 train (the seed's worst:" in result.stderr
    assert "5 held out" in result.stderr
    assert "3 cases is thin" in result.stderr


def test_overfit_trains_on_the_cases_the_seed_does_worst_on(a_run, monkeypatch):
    """A random three were the first attempt, and the seed scored 0.996 on
    them: nothing could beat its parent, so nothing was searched. The worst
    are where a search has room, and what a person would pick by hand."""
    from tests.test_gepa_adapter import CENSUS_OUTPUT

    monkeypatch.setitem(
        targets.TARGETS,
        "extract",
        dataclasses.replace(
            targets.EXTRACT,
            corpus=lambda **kwargs: [
                ExtractCase.authored(
                    name=f"case-{i}",
                    question="what was revenue last quarter?",
                    sql=SQL,
                    findings=f"[(1234.50,)]  -- case-{i}",
                )
                for i in range(8)
            ],
        ),
    )

    async def worse_on_odd_cases(**kwargs):
        # The findings reach the model in the user message, and carry the
        # case's name; the seed is made to record a census on the odd ones.
        sent = json.dumps(kwargs.get("messages", []))
        odd = any(f"case-{i}" in sent for i in (1, 3, 5))
        return llm.Result(
            text=json.dumps(CENSUS_OUTPUT if odd else GOOD_OUTPUT),
            stop_reason="end_turn", tokens_in=400, tokens_out=120,
        )

    monkeypatch.setattr(llm, "complete", worse_on_odd_cases)
    seen, _ = search_kwargs(monkeypatch, ["extract", "--overfit", "3"])

    assert sorted(c.name for c in seen["trainset"]) == ["case-1", "case-3", "case-5"]


def test_the_held_out_cases_are_scored_for_every_candidate_and_written(
    a_front, tmp_path
):
    """The number the section exists for. Every candidate in the pool, the
    seed included, on the cases the search never saw; on stderr as a table and
    in the front's file as a block, at the overfit path — the real run's front
    is not touched."""
    result = CliRunner().invoke(gepa.cli, ["extract", "--overfit", "3"])

    assert result.exit_code == 0, result.stderr
    assert "5 cases the search never saw" in result.stderr
    assert "worse than the seed on" in result.stderr
    assert "decided nothing" in result.stderr

    written = json.loads((tmp_path / "extract-overfit.pareto.json").read_text())
    assert written["holdout"]["cases"] == 5
    assert set(written["holdout"]["candidates"]) == {"0", "1", "2", "3"}
    assert written["holdout"]["candidates"]["0"]["train"] == 0.8
    assert written["holdout"]["candidates"]["1"]["holdout"] is not None
    assert not (tmp_path / "extract.pareto.json").exists()


def test_a_candidate_the_gate_discards_still_reaches_stdout(a_run, monkeypatch):
    """The gate says what it would have done, in red, and the run keeps the
    candidate anyway: that is the demonstration. The model here honours the
    invariants under the seed and records a census under the candidate, so
    the gate really does fail it — and the held-out sweep, scored the same
    way, says it lost every unseen case to the seed."""
    from tests.test_gepa_adapter import CENSUS_OUTPUT

    async def cheats_under_the_candidate(**kwargs):
        cheating = kwargs.get("system", "").startswith(BETTER)
        return llm.Result(
            text=json.dumps(CENSUS_OUTPUT if cheating else GOOD_OUTPUT),
            stop_reason="end_turn", tokens_in=400, tokens_out=120,
        )

    monkeypatch.setattr(llm, "complete", cheats_under_the_candidate)

    result = CliRunner().invoke(gepa.cli, ["extract", "--overfit", "3"])

    assert result.exit_code == 0, result.stderr
    assert "DISCARDED" in result.stderr, "the gate spoke"
    assert "decided nothing" in result.stderr, "and was ignored"
    assert result.stdout == BETTER + "\n", "kept whatever it produced"
    assert "5 of 5" in result.stderr, "and the unseen cases say it is worse"


@pytest.mark.parametrize(
    "n, reason",
    [("0", "at least one case"), ("8", "leaves nothing held out")],
)
def test_overfit_refuses_a_split_with_nothing_on_one_side(a_run, monkeypatch, n, reason):
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: pytest.fail("a search was started")
    )

    result = CliRunner().invoke(gepa.cli, ["extract", "--overfit", n])

    assert result.exit_code == 1
    assert reason in result.output


def test_the_thin_warning_is_about_the_training_set(a_run, monkeypatch):
    """It used to look at the corpus, which would stay silent on a 37-case
    corpus cut to three. What GEPA fits is the training set."""
    monkeypatch.setitem(
        targets.TARGETS,
        "extract",
        dataclasses.replace(
            targets.EXTRACT, corpus=lambda **kwargs: [_case(i) for i in range(20)]
        ),
    )

    _, normal = search_kwargs(monkeypatch, ["extract"])
    assert "is thin" not in normal.stderr, "14 train cases is not thin"

    _, thin = search_kwargs(monkeypatch, ["extract", "--val-fraction", "0.9"])
    assert "2 cases is thin" in thin.stderr


# --------------------------------------------------------------- transcript
#
# The front was always written to a file. The gate's verdicts and the diff
# lived only on the terminal, so "here is what last night's run left" was the
# front and nothing else.


def test_a_run_leaves_its_whole_stderr_beside_the_front(a_run, tmp_path):
    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == 0, result.stderr
    kept = (tmp_path / "extract.run.txt").read_text()
    assert "split     " in kept, "the run's own narration"
    assert "candidate 1" in kept, "the gate's verdict"
    assert "chars ->" in kept, "the diff's summary line"
    assert kept.strip() == result.stderr.strip(), "the terminal's text, nothing more"
    assert "\x1b[" not in kept, "and without the colour codes"


def test_a_probe_does_not_touch_the_transcript(well_behaved, monkeypatch, tmp_path):
    """The probe runs on stage, seconds before the transcript of last night's
    search is shown. It must not overwrite it."""
    monkeypatch.setattr(gepa, "OUT", tmp_path)
    (tmp_path / "extract.run.txt").write_text("last night")

    result = CliRunner().invoke(gepa.cli, ["extract", "--probe-only"])

    assert result.exit_code == 0, result.stderr
    assert (tmp_path / "extract.run.txt").read_text() == "last night"


def test_the_overfit_run_keeps_its_own_transcript(a_run, tmp_path):
    CliRunner().invoke(gepa.cli, ["extract", "--overfit", "3"])

    assert (tmp_path / "extract-overfit.run.txt").exists()
    assert not (tmp_path / "extract.run.txt").exists(), "section 3's is untouched"
