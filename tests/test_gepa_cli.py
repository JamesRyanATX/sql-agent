"""The command's contract: stdout is the prompt, and nothing else ever is.

A stray `click.echo` without `err=True` or a GEPA log line both produce the same
failure — a prompt file with commentary in it, which the next run reads back as
the seed. Invisible until someone pastes a progress bar into `extract.md`.

Scripted model, no database, no search: `_search` is replaced by a pool.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field

import pytest

pytest.importorskip("gepa", reason="uv run --group gepa")

from app import llm, prompts, tracing  # noqa: E402
from click.testing import CliRunner  # noqa: E402
from tools.gepa import cli as gepa  # noqa: E402
from tools.gepa import golden, targets  # noqa: E402
from tools.gepa.adapter import COMPONENT  # noqa: E402
from tools.gepa.cases import ExtractCase  # noqa: E402
from tests.test_gepa_adapter import GOOD_OUTPUT, SQL  # noqa: E402

BETTER = "a better instruction, which also honours every invariant"


@dataclass
class FakeResult:
    """Enough of `GEPAResult` for the command. An empty `val_subscores` is how
    a run with no per-example scores reaches `_pareto`."""

    candidates: list[dict[str, str]]
    val_aggregate_scores: list[float] = field(default_factory=list)
    total_metric_calls: int = 42
    # Per candidate, per validation case, keyed by position in the valset.
    # The turn gate's whole evidence base, and the report's table.
    val_subscores: list[dict] = field(default_factory=list)
    # Per validation case, the candidates with the top score on it.
    per_val_instance_best_candidates: dict = field(default_factory=dict)


# The real question, kept so the tests about it can put it back.
ASK = gepa.ask_before_spending


@pytest.fixture(autouse=True)
def somebody_agreed(monkeypatch):
    """Every search asks "Start?" before it starts, and a test has nobody at a
    terminal to answer. So the question is answered for every test here, and
    the tests that are about the question ask for `the_real_question`."""
    monkeypatch.setattr(gepa, "ask_before_spending", lambda yes=False: None)


@pytest.fixture
def the_real_question(somebody_agreed, monkeypatch):
    monkeypatch.setattr(gepa, "ask_before_spending", ASK)


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
    assert "nothing beat the starting text" in result.stderr


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


# ---------------------------------------------------------------- the report
#
# GEPA selects the next candidate to mutate from a table of candidates against
# validation examples. These say that table lands in a file and on stderr, is
# labelled with the cases it was scored on, and never reaches stdout.


@pytest.fixture
def a_pool(a_run, monkeypatch):
    """Four candidates over a two-case valset. 0 is the seed and 1 ties it on
    the first example, so both are best there; 3 is the only one best on the
    second; 2 is best on neither, so GEPA would never pick it to mutate.

    Returns the valset GEPA was handed, so a test can check the labels."""
    pool = [
        {COMPONENT: a_run},
        {COMPONENT: BETTER},
        {COMPONENT: "worse on every example"},
        {COMPONENT: "good on the second example only"},
    ]
    handed: dict = {}

    def search(*a, **k):
        handed["valset"] = k["valset"]
        return FakeResult(
            pool,
            [0.80, 0.95, 0.40, 0.72],
            # Keyed by position in the valset, as GEPA keys them.
            val_subscores=[
                {0: 1.0, 1: 0.6},
                {0: 1.0, 1: 0.9},
                {0: 0.5, 1: 0.3},
                {0: 0.44, 1: 1.0},
            ],
            per_val_instance_best_candidates={0: {1, 0}, 1: {3}},
        )

    monkeypatch.setattr(gepa, "_search", search)
    return handed


def test_the_report_is_written_without_touching_stdout(a_pool, tmp_path):
    """The whole command's contract, applied to a second artifact: a file is
    written, a table is printed, and the prompt on stdout is untouched."""
    path = tmp_path / "artifacts" / "extract.pareto.json"

    result = CliRunner().invoke(gepa.cli, ["extract", "--pareto", str(path)])

    assert result.exit_code == 0, result.stderr
    assert result.stdout == BETTER + "\n", "the report must not reach stdout"

    document = json.loads(path.read_text())
    assert list(document) == ["target", "score", "examples", "pool", "candidates"]
    assert document["target"] == "extract"
    assert document["score"] == targets.EXTRACT.score()
    assert document["pool"] == {
        "candidates": 4, "eligible": 3, "metric_calls": 42, "seed_index": 0,
    }

    assert "4 candidates over 2 validation examples, 3 best on at least one" in result.stderr
    assert "E1    E2" in result.stderr, "the table's example columns"


def test_the_examples_are_the_valset_in_the_order_gepa_was_given_it(a_pool, tmp_path):
    """GEPA keys its scores by position, so E1 has to be `valset[0]`. An
    extract case has no question, so its name is its label."""
    path = tmp_path / "report.json"

    result = CliRunner().invoke(gepa.cli, ["extract", "--pareto", str(path)])

    names = [case.name for case in a_pool["valset"]]
    assert json.loads(path.read_text())["examples"] == names
    assert f"E1  {names[0]}" in result.stderr
    assert f"E2  {names[1]}" in result.stderr


def test_every_candidate_is_listed_with_the_examples_it_is_best_on(a_pool, tmp_path):
    """Candidate 2 is best on no example and is listed anyway: the file is
    the whole pool, and which candidates are eligible is a column in it."""
    path = tmp_path / "report.json"

    CliRunner().invoke(gepa.cli, ["extract", "--pareto", str(path)])

    rows = json.loads(path.read_text())["candidates"]
    assert [r["index"] for r in rows] == [0, 1, 2, 3], "discovery order"
    assert [r["seed"] for r in rows] == [True, False, False, False]
    assert [r["best_on"] for r in rows] == [[1], [1], [], [2]]
    assert rows[3]["scores"] == [0.44, 1.0]
    assert rows[1]["val"] == 0.95
    assert rows[1]["chars"] == len(BETTER)
    assert rows[1]["components"] == {COMPONENT: BETTER}
    assert list(rows[0]) == [
        "index", "seed", "val", "scores", "best_on", "chars", "components",
    ]


def test_the_report_is_byte_stable_across_runs(a_pool, tmp_path):
    """The file is committed and read in a diff. GEPA hands back sets, and one
    unsorted set reaching the JSON makes every rerun a spurious change."""
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    CliRunner().invoke(gepa.cli, ["extract", "--pareto", str(first)])
    CliRunner().invoke(gepa.cli, ["extract", "--pareto", str(second)])

    assert first.read_bytes() == second.read_bytes()


def test_the_report_is_always_written_to_the_run_directory(a_pool, tmp_path):
    """`--pareto` names a tracked copy. Not passing it must not mean losing the
    evidence — `out/` is where every other artifact of a run already lands."""
    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == 0
    assert json.loads(gepa.pareto("extract").read_text())["candidates"]


def test_a_result_with_no_per_example_scores_says_so_rather_than_crashing(
    a_run, monkeypatch
):
    """A result with no validation rows is a run with no table, not a run
    that failed. Says so in one line and writes nothing."""
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult([{COMPONENT: a_run}, {COMPONENT: BETTER}], [0.8, 0.9]),
    )

    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == 0
    assert result.stdout == BETTER + "\n"
    assert "no per-example scores" in result.stderr
    assert not gepa.pareto("extract").exists()


def test_a_mean_that_is_not_a_number_is_written_as_null(a_run, monkeypatch):
    """GEPA averages a candidate with no validation scores to -inf, and
    `json.dumps` would write that as `-Infinity`, which a JSON reader
    refuses."""
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            [{COMPONENT: a_run}, {COMPONENT: BETTER}],
            [0.9, float("-inf")],
            val_subscores=[{0: 0.9, 1: 0.9}, {}],
            per_val_instance_best_candidates={0: {0}, 1: {0}},
        ),
    )

    CliRunner().invoke(gepa.cli, ["extract"])

    text = gepa.pareto("extract").read_text()
    assert "Infinity" not in text
    rows = json.loads(text)["candidates"]
    assert rows[1]["val"] is None
    assert rows[0]["val"] == 0.9


def test_a_candidate_missing_scores_shows_null_rather_than_crashing(a_run, monkeypatch):
    """GEPA can hold a candidate it scored on only some of the validation
    examples. Its missing cells are null in the file and `-` in the table.

    The run also ends NO_IMPROVEMENT, and the report is on disk anyway. That
    is the reason `_pareto` runs before the pool is judged: a run that
    produced nothing promotable is the one whose evidence is most worth
    keeping.
    """
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            [{COMPONENT: a_run}, {COMPONENT: BETTER}],
            [0.9, 0.0],
            val_subscores=[{0: 0.9, 1: 0.9}, {1: 0.0}],
            per_val_instance_best_candidates={0: {0}, 1: {0}},
        ),
    )

    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert result.exit_code == gepa.NO_IMPROVEMENT
    assert result.stdout == ""
    rows = json.loads(gepa.pareto("extract").read_text())["candidates"]
    assert rows[1]["scores"] == [None, 0.0]
    assert rows[1]["best_on"] == []
    assert rows[0]["best_on"] == [1, 2]


# --------------------------------------------------------- more than one thing
#
# The command is target-generic now. These are the properties that only show up
# once a second target exists, and the one about money.

TOOLS_SEED = None  # filled by the fixture; the live descriptions


@pytest.fixture
def a_tools_run(monkeypatch, tmp_path, well_behaved):
    """A finished tools search, faked from the gate backwards.

    The corpus is the real `demo/golden/`, handed over in place of a harvest:
    it is tracked, it costs nothing to read, and it is the shape a harvest
    produces, so the split and the gate see what they will see for real.
    """
    seed = targets.TOOLS.seed()
    better = {**seed, "list_tables": "Every table, with its column count."}
    monkeypatch.setattr(gepa, "OUT", tmp_path)
    monkeypatch.setitem(
        targets.TARGETS,
        "tools",
        dataclasses.replace(targets.TOOLS, corpus=lambda **kwargs: golden.load()),
    )
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


def handed_to_search(monkeypatch, seed, argv: list[str]) -> dict:
    """Every keyword the search was handed."""
    seen: dict = {}

    def record(gepa_module, **kwargs):
        seen.update(kwargs)
        return FakeResult([seed], [1.0])

    monkeypatch.setattr(gepa, "_search", record)
    CliRunner().invoke(gepa.cli, argv)
    return seen


def test_extract_proposes_from_one_case_and_the_turn_targets_from_gepas_three(
    a_run, monkeypatch
):
    """Shown three failing cases, the reflection model rewrites the whole
    prompt in one step. Shown one, it corrects one mistake at a time."""
    assert handed_to_search(monkeypatch, {COMPONENT: a_run}, ["extract"])["minibatch"] == 1
    assert targets.TOOLS.minibatch is None and targets.CONFIG.minibatch is None


def test_the_batch_size_reaches_gepa_only_when_a_target_names_one(monkeypatch, tmp_path):
    seen: list[dict] = []

    class Gepa:
        @staticmethod
        def optimize(**kwargs):
            seen.append(kwargs)

    monkeypatch.setattr(gepa, "OUT", tmp_path)
    common = dict(
        adapter=None, reflection=None, seed_candidate={}, trainset=[], valset=[],
        node="extract", budget=1, seed=0, verbose=False,
    )
    gepa._search(Gepa, **common, minibatch=1)
    gepa._search(Gepa, **common)

    assert seen[0]["reflection_minibatch_size"] == 1
    assert "reflection_minibatch_size" not in seen[1]


def test_a_whole_turn_target_brings_its_own_budget(a_tools_run, monkeypatch):
    """A whole-turn rollout is about 11,500 tokens, so a whole-turn target
    names its own budget rather than taking the command's 150. It is 180,
    about two million tokens, which the run states before it asks: `make
    gepa-tools` is the whole command, with nothing to pass."""
    assert budget_of(monkeypatch, a_tools_run[0], ["tools", "--yes"]) == 180


def test_the_cheap_target_keeps_the_budget_it_always_had(a_run, monkeypatch):
    assert budget_of(monkeypatch, {COMPONENT: a_run}, ["extract"]) == 150


def test_an_explicit_budget_still_wins(a_tools_run, monkeypatch):
    assert budget_of(monkeypatch, a_tools_run[0], ["tools", "--budget", "4", "--yes"]) == 4


def test_a_run_prints_the_harvest_and_the_split_and_nothing_else_before_it_asks(
    a_tools_run,
):
    """What is on the screen before "Start?": where the cases came from and how
    they were split, which names the budget. No summary of the target and no
    paragraph about what the run will not touch."""
    result = CliRunner().invoke(gepa.cli, ["tools", "--yes"])

    assert result.exit_code == 0
    assert "split: " in result.stderr and "budget 180 calls" in result.stderr
    for gone in ("rollouts at roughly", "components ", "proposals ", "memory off"):
        assert gone not in result.stderr, gone


@pytest.mark.parametrize("target", ["tools", "extract"])
def test_every_run_refuses_a_pipe_rather_than_spending_into_it(
    target, request, the_real_question, monkeypatch
):
    """Every CI job, and any run with nobody at a terminal: nobody is there to
    agree, and the run would spend anyway. The cheap target asks as well, so
    every target reads the same."""
    request.getfixturevalue("a_tools_run" if target == "tools" else "a_run")
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: pytest.fail("it started without asking")
    )

    result = CliRunner().invoke(gepa.cli, [target])

    assert result.exit_code == 1
    assert "nobody at a terminal" in result.output
    assert result.stdout == ""


def test_yes_starts_a_run_without_asking(a_run, the_real_question):
    result = CliRunner().invoke(gepa.cli, ["extract", "--yes"])

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
    # The answer key in place of a harvest, as `a_tools_run` does.
    monkeypatch.setitem(
        targets.TARGETS,
        "config",
        dataclasses.replace(targets.CONFIG, corpus=lambda **kwargs: golden.load()),
    )
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
    # The diff names where a person would go.
    assert "config/config.yaml" in result.stderr


def test_the_config_target_is_searchable_and_the_rest_still_say_why_not(monkeypatch):
    """`config` used to sit in UNWIRED with a reason that asked for exactly what
    the metric now has. A reason that has become false is worse than none.

    The check is stubbed: the real one is nineteen cold turns."""
    monkeypatch.setitem(
        targets.TARGETS,
        "config",
        dataclasses.replace(targets.CONFIG, check=lambda loop, yes, **kw: 0),
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
    a_pool, tmp_path
):
    """The number the section exists for. Every candidate in the pool, the
    seed included, on the cases the search never saw; on stderr as a table and
    in the report's file as a block, at the overfit path — the real run's
    report is not touched."""
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
    assert "split: " in kept, "the run's own narration"
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


# --------------------------------------------------------------- iterations
#
# One iteration and a read of what it did is how a search is understood.
# `--budget` approximates that through call arithmetic; GEPA has a stopper
# that counts proposals, and this passes it through.


def test_iterations_reaches_the_search(a_run, monkeypatch):
    seen, result = search_kwargs(monkeypatch, ["extract", "--iterations", "2"])

    assert seen["iterations"] == 2
    assert "budget 150 calls, 2 iterations" in result.stderr


def test_iterations_means_that_many_more_whatever_the_state_says():
    """GEPA's own proposals stopper counts from the start of the state, and
    `--resume` reloads the state: observed, a resumed run woke at iteration
    20 and stopped before proposing. Counted from the first check instead,
    one more is one more on a fresh run and on a resumed one alike."""

    class State:
        def __init__(self, i):
            self.i = i

    fresh = gepa._More(1, lambda s: s.i)
    assert not fresh(State(-1)), "about to run the first proposal"
    assert fresh(State(0)), "one done"

    resumed = gepa._More(1, lambda s: s.i)
    assert not resumed(State(19))
    assert resumed(State(20))

    two = gepa._More(2, lambda s: s.i)
    assert not two(State(-1)) and not two(State(0)) and two(State(1))


def test_iterations_becomes_a_stopper_and_the_budget_counts_from_there_too(monkeypatch, tmp_path):
    """Two relative stoppers and no absolute call cap: a resumed run has
    already spent the budget, and an absolute cap would stop it at once."""
    seen = {}

    class FakeGepa:
        @staticmethod
        def optimize(**kwargs):
            seen.update(kwargs)
            return FakeResult([{COMPONENT: "seed"}], [1.0])

    monkeypatch.setattr(gepa, "OUT", tmp_path)
    common = dict(
        adapter=None, reflection=None, seed_candidate={COMPONENT: "seed"},
        trainset=[], valset=[], node="extract", budget=17, seed=0, verbose=False,
    )

    from gepa.utils.stop_condition import ScoreThresholdStopper

    gepa._search(FakeGepa, iterations=2, **common)
    perfect, steps, calls = seen["stop_callbacks"]
    assert isinstance(perfect, ScoreThresholdStopper) and perfect.threshold == 1.0
    assert isinstance(steps, gepa._More) and steps.n == 2
    assert isinstance(calls, gepa._More) and calls.n == 17
    assert seen["max_metric_calls"] is None

    seen.clear()
    gepa._search(FakeGepa, **common)
    (perfect,) = seen["stop_callbacks"]
    assert isinstance(perfect, ScoreThresholdStopper) and perfect.threshold == 1.0
    assert seen["max_metric_calls"] == 17


def test_the_search_stops_when_a_candidate_scores_one_on_every_example():
    """GEPA's own stopper reads the best mean validation score in the state.
    A mean of 1.0 means 1.0 on every example, since no score is above 1.0."""
    from types import SimpleNamespace

    seen = {}

    class FakeGepa:
        @staticmethod
        def optimize(**kwargs):
            seen.update(kwargs)
            return FakeResult([{COMPONENT: "seed"}], [1.0])

    gepa._search(
        FakeGepa, adapter=None, reflection=None, seed_candidate={COMPONENT: "seed"},
        trainset=[], valset=[], node="extract", budget=17, seed=0, verbose=False,
    )
    (perfect,) = seen["stop_callbacks"]

    assert perfect(SimpleNamespace(program_full_scores_val_set=[0.4, 1.0])) is True
    assert perfect(SimpleNamespace(program_full_scores_val_set=[0.4, 0.8])) is False


def test_a_pool_of_one_after_the_asked_iterations_is_reported_as_that(a_run, monkeypatch):
    """Not "GEPA proposed nothing", which points at the metric. The run
    stopped where it was told to."""
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: FakeResult([{COMPONENT: a_run}], [1.0])
    )

    result = CliRunner().invoke(gepa.cli, ["extract", "--iterations", "1"])

    assert result.exit_code == gepa.NO_IMPROVEMENT
    assert "Stopped after 1 iteration, as asked" in result.stderr
    assert "GEPA proposed nothing" not in result.stderr


def test_zero_iterations_is_a_usage_error(a_run, monkeypatch):
    monkeypatch.setattr(
        gepa, "_search", lambda *a, **k: pytest.fail("a search was started")
    )

    result = CliRunner().invoke(gepa.cli, ["extract", "--iterations", "0"])

    assert result.exit_code == 2
    assert "at least one" in result.output


# -------------------------------------------------------------------- --from
#
# A search can start from the text in a file rather than the working text.
# The demo starts `extract` from a deliberately short prompt.

SHORT = "Record what this query established, as short notes the next question can use."


@pytest.fixture
def short_file(tmp_path):
    path = tmp_path / "extract.short.md"
    path.write_text(SHORT + "\n\n", encoding="utf-8")
    return path


def from_the_seed(monkeypatch) -> dict:
    """A search that returns its seed and one better candidate, and keeps
    what it was handed."""
    seen: dict = {}

    def search(*a, **k):
        seen.update(k)
        return FakeResult([k["seed_candidate"], {COMPONENT: BETTER}], [0.5, 0.9])

    monkeypatch.setattr(gepa, "_search", search)
    return seen


def test_from_makes_the_files_text_the_seed_the_search_receives(
    a_run, monkeypatch, short_file
):
    seen = from_the_seed(monkeypatch)

    result = CliRunner().invoke(gepa.cli, ["extract", "--from", str(short_file)])

    assert result.exit_code == 0, result.stderr
    assert seen["seed_candidate"] == {COMPONENT: SHORT}, "read and stripped"
    assert result.stdout == BETTER + "\n"


def test_from_says_which_text_the_search_started_from(a_run, monkeypatch, short_file):
    from_the_seed(monkeypatch)

    result = CliRunner().invoke(gepa.cli, ["extract", "--from", str(short_file)])

    assert (
        f"seed: {short_file}, {len(SHORT):,} chars, in place of the working "
        f"text's {len(a_run):,}"
    ) in result.stderr
    assert f"seed: {short_file}" in (gepa.OUT / "extract.run.txt").read_text()


def test_the_diff_header_names_the_file_the_search_started_from(
    a_run, monkeypatch, short_file
):
    from_the_seed(monkeypatch)

    result = CliRunner().invoke(gepa.cli, ["extract", "--from", str(short_file)])

    assert f"--- {short_file}" in result.stderr
    assert "--- " + str(prompts.directory() / "extract.md") not in result.stderr


def test_an_empty_from_file_is_refused(a_run, monkeypatch, tmp_path):
    monkeypatch.setattr(gepa, "_search", lambda *a, **k: pytest.fail("a search was started"))
    empty = tmp_path / "empty.md"
    empty.write_text("  \n\n", encoding="utf-8")

    result = CliRunner().invoke(gepa.cli, ["extract", "--from", str(empty)])

    assert result.exit_code == 2
    assert f"--from {empty} is empty" in result.stderr


def test_from_is_refused_for_a_target_with_several_components(
    monkeypatch, tmp_path, short_file
):
    """The tools target has one component per tool, and a file holds one
    text, so there is no saying which description it replaces."""
    monkeypatch.setattr(gepa, "_search", lambda *a, **k: pytest.fail("a search was started"))

    result = CliRunner().invoke(gepa.cli, ["tools", "--from", str(short_file)])

    assert result.exit_code == 2
    assert (
        f"tools has {len(targets.TOOLS.seed())}" in result.stderr
    ), "the message says how many components"


def test_the_size_limit_stays_tied_to_the_working_prompt(a_run, monkeypatch, short_file):
    """A search that starts from a short prompt needs room to grow to 1.5
    times the working prompt, not 1.5 times its own start."""
    seen = from_the_seed(monkeypatch)

    CliRunner().invoke(gepa.cli, ["extract", "--from", str(short_file)])

    assert seen["adapter"].max_chars == int(targets.MAX_GROWTH * len(a_run))
    assert seen["adapter"].max_chars > int(targets.MAX_GROWTH * len(SHORT))


def test_probe_only_with_from_checks_the_working_text_and_says_so(
    monkeypatch, tmp_path, short_file
):
    """The real check, under a scripted model: the probes are replayed
    against the working prompt, and the file is named only to say it is not
    used."""
    monkeypatch.setattr(gepa, "OUT", tmp_path)
    sent: list[str] = []

    async def fake_complete(**kwargs):
        sent.append(kwargs["system"])
        return llm.Result(
            text=json.dumps(GOOD_OUTPUT),
            stop_reason="end_turn", tokens_in=400, tokens_out=120,
        )

    monkeypatch.setattr(llm, "complete", fake_complete)

    result = CliRunner().invoke(
        gepa.cli, ["extract", "--probe-only", "--from", str(short_file)]
    )

    assert result.exit_code == 0, result.stderr
    assert (
        f"--probe-only checks the working text; the --from file {short_file} "
        f"is not used"
    ) in result.stderr
    assert sent and set(sent) == {prompts.get("extract")}, "the working prompt, every probe"
    assert "all 4 probes pass" in result.stderr


def test_with_from_the_gate_and_the_seed_check_use_the_files_text(
    a_run, monkeypatch, tmp_path
):
    """The file is the seed in every comparison the search makes. Its text
    fails two probes under this model and the working prompt fails none, so
    the gate's baseline shows which one it read. The seed's score is found
    by matching the file's text, so a candidate below it is not offered."""
    from tests.test_gepa_adapter import CENSUS_OUTPUT

    async def census_under_degenerate(**kwargs):
        payload = CENSUS_OUTPUT if "DEGENERATE" in kwargs["system"] else GOOD_OUTPUT
        return llm.Result(
            text=json.dumps(payload), stop_reason="end_turn", tokens_in=400, tokens_out=120,
        )

    monkeypatch.setattr(llm, "complete", census_under_degenerate)
    start = tmp_path / "start.md"
    start.write_text("DEGENERATE: a short prompt that records counts", encoding="utf-8")
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            [k["seed_candidate"], {COMPONENT: BETTER}], [0.95, 0.90]
        ),
    )

    result = CliRunner().invoke(gepa.cli, ["extract", "--from", str(start)])

    assert "seed passes 2/4" in result.stderr, "the file's text, not the working prompt's"
    assert result.exit_code == gepa.NO_IMPROVEMENT
    assert "The text the search started from scored 0.950" in result.stderr


def with_rows(monkeypatch, scores: list[float]) -> None:
    """A search over two validation examples that returns its seed and BETTER
    with these averages, and a full row of per-example scores for each."""
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult(
            [k["seed_candidate"], {COMPONENT: BETTER}][: len(scores)],
            scores,
            val_subscores=[{0: s, 1: s} for s in scores],
            per_val_instance_best_candidates={0: {len(scores) - 1}, 1: {len(scores) - 1}},
        ),
    )


def test_with_from_the_report_has_no_working_block(a_run, monkeypatch, short_file):
    """The working text is scored for the `compare:` line and nothing else.
    It is not in the report file or in the table the run prints."""
    with_rows(monkeypatch, [0.4, 1.0])

    result = CliRunner().invoke(gepa.cli, ["extract", "--from", str(short_file)])

    assert result.exit_code == 0, result.stderr
    document = json.loads(gepa.pareto("extract").read_text())
    assert list(document) == ["target", "score", "examples", "pool", "candidates"]
    assert "working " not in result.stderr.replace("working text", "")
    assert "compare: " in result.stderr


def test_without_from_the_working_text_is_not_scored_again(a_run, monkeypatch):
    """Without `--from` the seed is the working text, already in the pool."""
    with_rows(monkeypatch, [0.4, 1.0])

    result = CliRunner().invoke(gepa.cli, ["extract"])

    assert "working" not in json.loads(gepa.pareto("extract").read_text())
    assert "compare: " not in result.stderr


def test_with_from_a_pool_of_one_does_not_score_the_working_text(
    a_run, monkeypatch, short_file
):
    """No winner is printed, so there is no `compare:` line, and scoring the
    working text would be rollouts spent on nothing."""
    with_rows(monkeypatch, [0.4])
    sent: list[str] = []

    async def counting(**kwargs):
        sent.append(kwargs["system"])
        return llm.Result(
            text=json.dumps(GOOD_OUTPUT), stop_reason="end_turn", tokens_in=400, tokens_out=120,
        )

    monkeypatch.setattr(llm, "complete", counting)

    result = CliRunner().invoke(gepa.cli, ["extract", "--from", str(short_file)])

    assert result.exit_code == gepa.NO_IMPROVEMENT
    assert a_run not in sent
    assert "compare: " not in result.stderr


def test_with_from_the_winner_is_compared_with_the_working_text(
    a_run, monkeypatch, short_file
):
    with_rows(monkeypatch, [0.4, 1.0])

    result = CliRunner().invoke(gepa.cli, ["extract", "--from", str(short_file)])

    label = prompts.directory() / "extract.md"
    assert (
        f"compare: started from {short_file} at 0.400, the winner scored 1.000, "
        f"and the working text {label} scored 1.000"
    ) in result.stderr
    assert "is not a replacement" not in result.stderr, "not below, so no warning"
    assert result.stdout == BETTER + "\n"


def test_a_winner_below_the_working_text_is_printed_with_a_warning(
    a_run, monkeypatch, short_file
):
    """A warning and not a refusal: the search beat the file it started
    from, and that is what it was asked to do."""
    with_rows(monkeypatch, [0.4, 0.8])

    result = CliRunner().invoke(gepa.cli, ["extract", "--from", str(short_file)])

    label = prompts.directory() / "extract.md"
    assert (
        f"The text on stdout scored 0.800, lower than the 1.000 of the text the "
        f"agent runs, {label}. It is not a replacement for it."
    ) in result.stderr
    assert result.stdout == BETTER + "\n"


# --------------------------------------------------------------- --resume
#
# GEPA resumes from its saved state and ignores the seed it is handed, so the
# command keeps the seed beside the state and checks it.


def test_a_fresh_run_records_the_text_it_started_from(a_run):
    CliRunner().invoke(gepa.cli, ["extract"])

    saved = json.loads((gepa.run_dir("extract") / "seed.json").read_text())
    assert saved == {COMPONENT: a_run}


def test_resume_is_refused_when_the_saved_search_started_from_other_text(
    a_run, monkeypatch, short_file
):
    """The saved search started from the short file; this run's seed is the
    working prompt."""
    gepa.run_dir("extract").mkdir(parents=True)
    (gepa.run_dir("extract") / "seed.json").write_text(json.dumps({COMPONENT: SHORT}))
    monkeypatch.setattr(gepa, "_search", lambda *a, **k: pytest.fail("a search was started"))

    result = CliRunner().invoke(gepa.cli, ["extract", "--resume"])

    assert result.exit_code == 1
    assert "started from different text" in result.stderr
    assert "make gepa-extract-reset" in result.stderr


def test_resume_continues_when_the_saved_search_started_from_the_same_text(a_run):
    CliRunner().invoke(gepa.cli, ["extract"])

    result = CliRunner().invoke(gepa.cli, ["extract", "--resume"])

    assert result.exit_code == 0, result.stderr
    assert "could not be checked" not in result.stderr


def test_resume_with_a_saved_search_and_no_record_checks_it_after_loading(a_run):
    """A run directory written before the seed was recorded. GEPA's saved
    search keeps its seed as candidate 0, which here is this run's seed, so
    the run continues and the record is written for the next resume."""
    gepa.run_dir("extract").mkdir(parents=True)
    (gepa.run_dir("extract") / "gepa_state.bin").write_bytes(b"")

    result = CliRunner().invoke(gepa.cli, ["extract", "--resume"])

    assert result.exit_code == 0, result.stderr
    assert (
        "the saved search has no record of the text it started from, so that "
        "could not be checked before loading it"
    ) in result.stderr
    saved = json.loads((gepa.run_dir("extract") / "seed.json").read_text())
    assert saved == {COMPONENT: a_run}


def test_resume_with_a_saved_search_that_started_elsewhere_is_refused_after_loading(
    a_run, monkeypatch
):
    """Candidate 0 is the saved search's seed. It is not this run's, so the
    run stops before the report, the gate or stdout."""
    gepa.run_dir("extract").mkdir(parents=True)
    (gepa.run_dir("extract") / "gepa_state.bin").write_bytes(b"")
    monkeypatch.setattr(
        gepa,
        "_search",
        lambda *a, **k: FakeResult([{COMPONENT: SHORT}, {COMPONENT: BETTER}], [0.4, 0.9]),
    )

    result = CliRunner().invoke(gepa.cli, ["extract", "--resume"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "started from different text" in result.stderr
    assert "make gepa-extract-reset" in result.stderr
    assert not gepa.pareto("extract").exists(), "no report written"
    assert "checking the pool" not in result.stderr, "the gate did not run"
    assert not (gepa.run_dir("extract") / "seed.json").exists()


def test_resume_with_no_record_and_no_saved_search_writes_the_record_quietly(a_run):
    """GEPA will start fresh, so there is nothing to check and nothing to say."""
    gepa.run_dir("extract").mkdir(parents=True)

    result = CliRunner().invoke(gepa.cli, ["extract", "--resume"])

    assert result.exit_code == 0, result.stderr
    assert "no record of the text" not in result.stderr
    saved = json.loads((gepa.run_dir("extract") / "seed.json").read_text())
    assert saved == {COMPONENT: a_run}


def test_a_refused_resume_leaves_the_previous_transcript_as_it_was(
    a_run, monkeypatch
):
    gepa.run_dir("extract").mkdir(parents=True)
    (gepa.run_dir("extract") / "seed.json").write_text(json.dumps({COMPONENT: SHORT}))
    transcript = gepa.transcript("extract")
    transcript.write_text("last night's run\n", encoding="utf-8")
    monkeypatch.setattr(gepa, "_search", lambda *a, **k: pytest.fail("a search was started"))

    result = CliRunner().invoke(gepa.cli, ["extract", "--resume"])

    assert result.exit_code == 1
    assert "started from different text" in result.stderr
    assert transcript.read_text(encoding="utf-8") == "last night's run\n"
