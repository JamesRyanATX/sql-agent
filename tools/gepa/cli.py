"""`python -m tools.gepa <target>` — GEPA over one searchable thing.

One command, needing no other: it finds the corpus, searches, writes the front,
gates the pool and prints the winner. Fresh corpus and fresh run dir every time,
so a run is never partly made of the last one.

stdout is the promoted text and nothing else. Progress, the front, the diff and
the gate go to stderr, so `make gepa-extract > new.md` leaves prose with no
commentary.

What a target *is* — its seed, its corpus, its gate, how its winner is rendered
— lives in `targets.py`. Everything here is the same whichever one is running.

Nothing is written to config/prompts/ or app/, and nothing is committed.

`--overfit N` is the same run with the guards off: N cases as both train and
validation, the gate reporting and deciding nothing, and afterwards every
candidate in the pool scored on the cases the search never saw. It exists to
be shown failing.

Exit: 0 text is on stdout, 1 preflight, 2 the target has no metric, 3 nothing
scored better than what is already there. Through `make` these all arrive as
make's own 2 — an empty stdout is the portable signal.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import click

from tools.gepa import targets
from tools.gepa.gates import Survivor, say
from tools.gepa.targets import Target

# The tests' one monkeypatch point, hence the three paths below deriving from it.
OUT = Path(__file__).resolve().parent / "out"

UNWIRED_NODE = 2
NO_IMPROVEMENT = 3

# Under this, GEPA is fitting noise.
THIN_CORPUS = 12


def corpus(name: str) -> Path:
    return OUT / f"{name}.jsonl"


def run_dir(name: str) -> Path:
    return OUT / "run" / name


def pareto(name: str) -> Path:
    return OUT / f"{name}.pareto.json"


def reflections(name: str) -> Path:
    """What every reflection read, one JSON line each. Inside the run dir, so
    it is wiped with the run and never outlives the candidates it explains."""
    return run_dir(name) / "reflections.jsonl"


def transcript(name: str) -> Path:
    """Everything the run said, as the terminal saw it minus the colour.

    The front is written to a file; until this, the gate's verdicts and the
    diff lived only on the terminal, and "here is what last night's run
    left" was the front and nothing else. Beside the front rather than in the
    run dir, so `--probe-only` and a resume cannot touch it.
    """
    return OUT / f"{name}.run.txt"


class _Tee:
    """stderr and a file. Everything a run says goes through stderr — `say`
    is `click.secho(err=True)`, and `_search` redirects GEPA's own log there
    — so one tee is the whole transcript. The file gets the text without the
    colour codes; the terminal decides about colour for itself."""

    def __init__(self, terminal, file) -> None:
        self._terminal, self._file = terminal, file

    def write(self, text: str) -> int:
        self._terminal.write(text)
        self._file.write(click.unstyle(text))
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._file.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()

    def __getattr__(self, name):
        return getattr(self._terminal, name)


@contextlib.contextmanager
def _recording(name: str):
    """Keep the run's stderr in `transcript(name)` for as long as it runs."""
    path = transcript(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        real = sys.stderr
        sys.stderr = _Tee(real, file)
        try:
            yield
        finally:
            sys.stderr = real


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("target")
@click.option("-v", "--verbose", is_flag=True, help="Per-probe and per-candidate detail.")
@click.option("--budget", type=int, default=None, help="max metric calls [per target]")
@click.option("--val-fraction", default=0.3, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option(
    "--resume",
    is_flag=True,
    help="continue the last run: its corpus and its GEPA state, no harvest",
)
@click.option("--probe-only", is_flag=True, help="check the invariants and stop")
@click.option("--days", default=30, show_default=True, help="how far back to harvest")
@click.option("--yes", is_flag=True, help="do not ask before spending")
@click.option(
    "--pareto",
    "pareto_to",
    type=click.Path(dir_okay=False, path_type=Path),
    help="also write the front here, somewhere tracked",
)
@click.option(
    "--overfit",
    type=int,
    default=None,
    metavar="N",
    help="guards off: train and validate on N cases, gate reports only, "
    "then score the pool on the rest",
)
@click.option(
    "--iterations",
    type=int,
    default=None,
    metavar="N",
    help="stop after N proposals; the budget still caps the calls",
)
def cli(
    target: str,
    verbose: bool,
    budget: int | None,
    val_fraction: float,
    seed: int,
    resume: bool,
    probe_only: bool,
    days: int,
    yes: bool,
    pareto_to: Path | None,
    overfit: int | None,
    iterations: int | None,
) -> None:
    """GEPA over one searchable thing. The new text goes to stdout.

    \b
      make gepa-extract              read it
      make gepa-extract > new.md     keep it

    Nothing is written to config/prompts/ and nothing is committed.
    """
    chosen = _resolve(target)
    seed_candidate = chosen.seed()
    if iterations is not None and iterations < 1:
        raise click.UsageError("--iterations needs at least one")

    from tools.gepa.adapter import Loop, reflection_lm

    if probe_only:
        if chosen.check is None:
            raise click.ClickException(f"{chosen.name} has no cheap pre-check")
        with Loop() as loop:
            raise SystemExit(chosen.check(loop, yes))

    # An overfit run keeps its own run dir, front and transcript, so the real
    # run's are not wiped by the demonstration of what the real run refuses
    # to do. It is always fresh: `--resume` there means the last corpus and
    # no more, which is what it wants — the same cases the real run searched.
    run = f"{chosen.name}-overfit" if overfit is not None else chosen.name
    with _recording(run):
        _run(
            chosen, seed_candidate, run=run, verbose=verbose, budget=budget,
            val_fraction=val_fraction, seed=seed, resume=resume, days=days,
            yes=yes, pareto_to=pareto_to, overfit=overfit, iterations=iterations,
        )


def _run(
    chosen: Target,
    seed_candidate: dict[str, str],
    *,
    run: str,
    verbose: bool,
    budget: int | None,
    val_fraction: float,
    seed: int,
    resume: bool,
    days: int,
    yes: bool,
    pareto_to: Path | None,
    overfit: int | None,
    iterations: int | None = None,
) -> None:
    """The search, from corpus to stdout. Split from the command so the
    transcript can be held open around every way out of it."""
    from tools.gepa.adapter import Loop, reflection_lm

    _fresh_run_dir(run, resume and overfit is None)
    cases = chosen.corpus(days=days, resume=resume, verbose=verbose)
    budget = chosen.budget if budget is None else budget
    holdout: list[Any] = []
    seed_scores: list[float] | None = None
    if overfit is not None:
        _overfit_fits(cases, overfit)
    else:
        trainset, valset = _split(cases, val_fraction, seed)
        say(
            f"split     {len(trainset)} train / {len(valset)} val, budget "
            f"{budget} calls{_steps(iterations)}"
        )
        _not_too_thin(trainset)
        _budget_covers(budget, valset)

    _confirm(
        chosen, seed_candidate, cases, budget=budget, yes=yes,
        holdout=len(cases) - overfit if overfit is not None else 0,
    )

    import gepa

    with Loop() as loop:
        adapter = chosen.adapter(loop)
        # Set here rather than by the target: where a run keeps its files is
        # the command's business, and every adapter has the attribute.
        adapter.reflections = reflections(run)

        if overfit is not None:
            # Needs the adapter, so it happens here: the seed is scored over
            # the corpus and trained on the cases it does worst on.
            trainset, holdout, seed_scores = _overfit_split(
                adapter, seed_candidate, cases, overfit
            )
            valset = trainset
            if iterations is not None:
                say(f"          budget {budget} calls{_steps(iterations)}")
            _not_too_thin(trainset)
            _budget_covers(budget, valset)

        result = _search(
            gepa,
            adapter=adapter,
            reflection=reflection_lm(loop),
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            node=run,
            budget=budget,
            seed=seed,
            verbose=verbose,
            templates=chosen.templates() if chosen.templates else None,
            iterations=iterations,
        )

        say(
            f"search    {result.total_metric_calls} metric calls, "
            f"{len(result.candidates)} candidates in the pool"
        )
        seed_index = _seed_index(result, seed_candidate)

        # The cases the search never saw, scored for every candidate. On a
        # pool of one there is only the seed, and the seed's own held-out
        # score says nothing about overfitting.
        unseen = None
        if holdout and len(result.candidates) > 1:
            unseen = _holdout(adapter, result, holdout, seed_index, seed_scores or [])

        # Before the empty-pool check below, not after. A run that ends with
        # NO_IMPROVEMENT is the one whose evidence is most worth keeping, and
        # on a pool of one the seed's own terms are what say whether the metric
        # is saturated — which is exactly what that message asks you to go and
        # find out.
        _pareto(
            result,
            target=chosen,
            seed_index=seed_index,
            path=pareto_to,
            name=run,
            holdout=unseen,
        )

        # `skip_perfect_score` means a seed at the top of the metric is never
        # mutated. A result, but an empty pool reads as a broken run.
        if len(result.candidates) <= 1:
            if iterations is not None:
                say(
                    f"\nStopped after {_plural(iterations, 'iteration')}, as "
                    f"asked, and the pool is the seed alone: nothing proposed "
                    f"in that many beat its parent. The transcript has what "
                    f"was tried.",
                    fg="yellow",
                )
            else:
                say(
                    "\nGEPA proposed nothing. Either the seed already scores at the "
                    "top of this metric on this corpus — in which case the corpus "
                    "or the metric is what needs work — or the budget ran out "
                    "before a mutation was accepted.",
                    fg="yellow",
                )
            raise SystemExit(NO_IMPROVEMENT)

        # The valset, not the whole corpus: GEPA keys its per-case scores by
        # position in what it was given to validate on.
        survivors = chosen.gate(loop, result, seed_candidate, valset)

        if overfit is not None:
            # The gate has said what it would have done. Now the run keeps
            # whatever scored best on the cases it trained on, which is the
            # demonstration: this is what a search does when nothing outside
            # the objective is allowed to say no.
            say("\noverfit   the gate reported and decided nothing", fg="yellow")
            scores = result.val_aggregate_scores or []
            survivors = sorted(
                (
                    Survivor(index=i, candidate=c, score=scores[i] if i < len(scores) else 0.0)
                    for i, c in enumerate(result.candidates)
                    if i != seed_index
                ),
                key=lambda s: -s.score,
            )

    if not survivors:
        say(
            "\nEvery candidate was disqualified by the gate. That is the gate "
            "working: the trainset metric cannot see these failures, because "
            "they surface turns later.",
            fg="red",
        )
        raise SystemExit(NO_IMPROVEMENT)

    best = survivors[0]
    seed_score = _seed_score(result, seed_candidate)

    # Clearing every probe does not make a candidate better than what it would
    # replace. Observed: a run scored the seed 0.959 and its best survivor 0.928.
    if seed_score is not None and best.score < seed_score:
        say(
            f"\nThe seed scored {seed_score:.3f} and the best survivor "
            f"{best.score:.3f} — nothing better than what is already there, so "
            f"stdout is empty.",
            fg="yellow",
        )
        raise SystemExit(NO_IMPROVEMENT)

    _diff(seed_candidate, best.candidate, chosen, score=best.score, seed_score=seed_score)

    # The one write to stdout.
    sys.stdout.write(chosen.render(best.candidate))


# -------------------------------------------------------------------- preflight
#
# Everything that can refuse the run refuses it here, before a token is spent:
# `apply` used to check its preconditions after a ten-minute search.


def _resolve(name: str) -> Target:
    """The target, or the reason there is not one."""
    if name in targets.TARGETS:
        return targets.TARGETS[name]

    if name not in targets.UNWIRED:
        raise click.ClickException(
            f"no target named {name!r} — this command knows "
            f"{', '.join(sorted(targets.TARGETS))} and can tell you why not for "
            f"{', '.join(sorted(targets.UNWIRED))}"
        )

    say(f"`{name}` is not searchable, so there is nothing to run.\n", fg="yellow")
    say(targets.UNWIRED[name])
    say(
        "\nWiring one up is a `Target` in tools/gepa/targets.py. Where a whole\n"
        "turn can score it, most of the work is already here — `TurnAdapter`\n"
        "and demo/golden/ — and what is left is the seed, a reflection template\n"
        "and a budget. Where it cannot, the metric comes first, and the metric\n"
        "is the hard part."
    )
    raise SystemExit(UNWIRED_NODE)


def _confirm(
    target: Target,
    seed_candidate: dict[str, str],
    cases: list[Any],
    *,
    budget: int,
    yes: bool,
    holdout: int = 0,
) -> None:
    """Say what the run is about to spend, and ask.

    Only where a rollout is expensive. `extract` is one model call a rollout and
    asking about it would train people to type `--yes`, which is how a
    confirmation stops being read.

    Everything here goes to stderr, the confirmation included: a prompt on
    stdout would land in the middle of the artifact.
    """
    if not target.rollout_tokens:
        return

    components = len(seed_candidate)
    tokens = budget * target.rollout_tokens
    say(f"\n{target.name}    {target.blurb}")
    say(f"  budget      {budget} rollouts at roughly "
        f"{target.rollout_tokens:,} tokens each — about {tokens:,} tokens")
    say(f"  components  {components}, mutated round-robin: about "
        f"{budget // max(components, 1)} proposals each at this budget")
    say(f"  corpus      {len(cases)} cases")
    if holdout:
        say(
            f"  held out    {holdout} cases, scored for every candidate in the "
            f"pool once the search ends — about {holdout} rollouts per candidate"
        )
    say(
        "\nEvery turn runs with the memory off: it reads nothing the agent has "
        "learned and saves nothing it learns. Nothing is written to app/ or "
        "config/, and nothing is committed."
    )

    ask_before_spending(yes)


def ask_before_spending(yes: bool = False) -> None:
    """The last chance not to spend the money.

    On stderr, confirmation included: a prompt on stdout would land in the
    middle of the artifact. Refuses a pipe rather than spending into one, which
    is `make gepa-tools > new.md` and every CI job.
    """
    if yes:
        return
    if not sys.stdin.isatty():
        raise click.ClickException(
            "this run costs real money and there is nobody at a terminal to "
            "agree to it — pass --yes if that is deliberate"
        )
    click.confirm("Start?", default=True, abort=True, err=True)


def _fresh_run_dir(name: str, resume: bool) -> None:
    """Wiped rather than refused. GEPA resumes from a run dir silently and its
    state is keyed to the seed and trainset it started from, so the leftovers of
    one run are never what the next one wants — and telling the operator to
    `rm -rf` it themselves makes a one-command target a two-command one."""
    directory = run_dir(name)
    if resume or not directory.exists():
        return
    shutil.rmtree(directory)


def _not_too_thin(trainset: list[Any]) -> list[Any]:
    """The training set, not the corpus: it is what GEPA fits. A 37-case
    corpus cut to three for an overfit run is thin, and the warning is the
    line the talk quotes."""
    if len(trainset) < THIN_CORPUS:
        say(
            f"          {len(trainset)} cases is thin — GEPA will fit whichever "
            f"questions happen to be in here",
            fg="yellow",
        )
    return trainset


def _budget_covers(budget: int, valset: list[Any]) -> None:
    """GEPA scores the seed over the whole valset before it proposes anything,
    so a budget that small buys a baseline and stops. The run then reports that
    GEPA proposed nothing, which reads as a saturated metric rather than as
    arithmetic. Said here, where it still costs nothing to change."""
    if budget <= len(valset):
        say(
            f"          {budget} calls against a {len(valset)}-case valset is "
            f"the baseline evaluation and nothing else — raise --budget or "
            f"lower --val-fraction",
            fg="yellow",
        )


def _overfit_fits(cases: list[Any], n: int) -> None:
    if n < 1:
        raise click.ClickException("--overfit needs at least one case to train on")
    if n >= len(cases):
        raise click.ClickException(
            f"--overfit {n} leaves nothing held out of {len(cases)} cases"
        )


def _overfit_split(
    adapter, seed_candidate: dict[str, str], cases: list[Any], n: int
) -> tuple[list[Any], list[Any], list[float]]:
    """The N cases the seed does worst on, to train and validate on; the rest
    never shown to the search; and the seed's score on every case.

    Worst, not random. A random three were the first attempt, and the seed
    scored 0.996 on them: nothing could beat its parent on the minibatch, so
    nothing entered the pool and there was nothing to overfit. The cases it
    does worst on are where a search has room, and they are what anyone
    tuning a prompt by hand would reach for, which is the point.

    GEPA admits a candidate on a train-minibatch improvement and ranks parents
    by the validation set, so a validation set is not held out from selection.
    The held-out cases are scored after the search, outside it, and the seed's
    scores from this sweep are its held-out baseline: measured once, the same
    way every other candidate will be.
    """
    say(f"\noverfit   the seed over all {len(cases)} cases, to find the {n} it does worst on")
    scores = adapter.evaluate(cases, seed_candidate).scores
    order = sorted(range(len(cases)), key=lambda i: scores[i])
    train = [cases[i] for i in order[:n]]
    rest = [cases[i] for i in order[n:]]
    worst = ", ".join(f"{scores[i]:.3f}" for i in order[:n])
    say(
        f"split     {n} train (the seed's worst: {worst}), validated on the "
        f"same {n}, {len(rest)} held out"
    )
    return train, rest, [scores[i] for i in order]


def _split(
    cases: list[Any], fraction: float, seed: int
) -> tuple[list[Any], list[Any]]:
    if len(cases) < 4:
        say(
            "too few cases to hold anything out — training and validating on "
            "the same rows, so the val score is not evidence of generalisation",
            fg="yellow",
        )
        return cases, cases
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * fraction))
    return shuffled[cut:], shuffled[:cut]


# --------------------------------------------------------------------- the search


class _More:
    """Stop once a counter has advanced `n` past where it stood at the first
    check. GEPA calls a stopper before each iteration, so the first check sees
    the state as this run found it: -1 on a fresh run, or wherever `--resume`
    reloaded it to. Either way, `n` more."""

    def __init__(self, n: int, read) -> None:
        self.n, self._read, self.start = n, read, None

    def __call__(self, state) -> bool:
        now = self._read(state)
        if self.start is None:
            self.start = now
        return now >= self.start + self.n


def _steps(iterations: int | None) -> str:
    return f", {_plural(iterations, 'iteration')}" if iterations is not None else ""


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _search(
    gepa,
    *,
    adapter,
    reflection,
    seed_candidate: dict[str, str],
    trainset: list[Any],
    valset: list[Any],
    node: str,
    budget: int,
    seed: int,
    verbose: bool,
    templates: dict[str, str] | None = None,
    iterations: int | None = None,
):
    """`gepa.optimize`, with its own stdout pointed at stderr: its engine logs
    to stdout, and one line of that is one line in the middle of the prompt.

    `reflection_prompt_template` is a dict keyed by component name, validated
    when the run is constructed — so a template missing its placeholders fails
    here rather than forty minutes in.

    `iterations` means that many *more*. GEPA's own proposals stopper counts
    from the start of the state, and `--resume` reloads the state, so on a
    resumed run "one iteration" would stop before proposing anything —
    observed: a resumed run woke at iteration 20 and stopped. `_More` counts
    from wherever the state is when this run begins, and with `--iterations`
    the budget is counted the same way, or a resumed run's spent budget would
    stop it just as surely. GEPA also installs a file stopper, `gepa.stop` in
    the run dir, which ends any run cleanly after the current iteration.
    """
    extra: dict[str, Any] = {"reflection_prompt_template": templates} if templates else {}
    if iterations is not None:
        extra["stop_callbacks"] = [
            _More(iterations, lambda state: state.i),
            _More(budget, lambda state: state.total_num_evals),
        ]
        budget = None
    with contextlib.redirect_stdout(sys.stderr):
        return gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection,
            max_metric_calls=budget,  # None with --iterations: the stoppers above
            run_dir=str(run_dir(node)),
            seed=seed,
            display_progress_bar=verbose,
            **extra,
        )


def _seed_index(result, seed_candidate: dict[str, str]) -> int | None:
    """Where the unmutated seed sits in the pool, if GEPA kept it.

    The whole candidate, not one component: with four of them, three matching
    and one mutated is not the seed.
    """
    for i, candidate in enumerate(result.candidates):
        if candidate == seed_candidate:
            return i
    return None


def _seed_score(result, seed_candidate: dict[str, str]) -> float | None:
    """What the current text scored on the valset. The gate skips the seed, so
    it cannot see that GEPA's best program is frequently that seed."""
    scores = result.val_aggregate_scores or []
    i = _seed_index(result, seed_candidate)
    return scores[i] if i is not None and i < len(scores) else None


# ----------------------------------------------------------------- the front
#
# The metric is weighted terms collapsed into one number, and the collapse
# settles a trade-off — grounding against cost — on the reader's behalf. GEPA
# keeps the per-term scores; `gepa.optimize` never writes them anywhere. These
# forty lines are the difference between "a Pareto front of prompts" as a
# sentence and as a file somebody can read.


def _front(subscores: list[dict[str, float]], objectives: tuple[str, ...]) -> list[int]:
    """The candidates no other candidate beats on every term.

    Not `result.per_objective_best_candidates`, which is the strictly smaller
    set of candidates topping at least one term. The non-dominated set is what
    "no other candidate is better on everything" means, and showing a file that
    disagrees with the sentence said over it is the failure this exists to
    prevent.

    A candidate missing a term is ineligible rather than an error: a rollout
    that errored on every validation case comes back with no terms at all.
    """
    scored = [i for i, s in enumerate(subscores) if all(o in s for o in objectives)]

    def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
        return all(a[o] >= b[o] for o in objectives) and any(
            a[o] > b[o] for o in objectives
        )

    return [
        i
        for i in scored
        if not any(dominates(subscores[j], subscores[i]) for j in scored if j != i)
    ]


def _pareto(
    result,
    *,
    target: Target,
    seed_index: int | None,
    path: Path | None,
    name: str | None = None,
    holdout: dict | None = None,
) -> None:
    """Write the front, and print it. Always to `out/`, and to `path` as well.

    `out/` is gitignored because a harvested case holds a recorded prompt. The
    talk needs a file that survives a clone, so `--pareto` makes that write
    something asked for by name rather than a copy out of the ignored
    directory. What lands there is candidate prose, which a reflection model
    wrote after reading harvested cases — the same exposure the promoted text
    already has, and handled the same way: read the diff.
    """
    weights = target.weights()

    # Direct attribute access, not getattr: these are real fields on
    # `GEPAResult`, and a default here would mean the tests' fake could drift
    # away from the type without anything noticing.
    subscores = result.val_aggregate_subscores
    if not subscores:
        say("pareto    no per-objective scores on this result — nothing to write")
        return

    # Weight order, not alphabetical, and the same order in every row and every
    # run: the file is committed and diffed.
    objectives = tuple(weights)
    # Sorted, because these arrive as sets and an unsorted set would make the
    # file diff against itself on nothing.
    tops = {
        objective: sorted(indices)
        for objective, indices in (result.per_objective_best_candidates or {}).items()
    }
    front = _front(subscores, objectives)
    scores = result.val_aggregate_scores or []

    document = {
        "target": target.name,
        "objectives": list(objectives),
        "weights": dict(weights),
        "pool": {
            "candidates": len(result.candidates),
            "on_front": len(front),
            "metric_calls": result.total_metric_calls,
            "seed_index": seed_index,
        },
        # The best reached on each term anywhere in the pool. One line of code,
        # and it is the row that proves no single candidate has all of it.
        "best_per_objective": {
            o: round(v, 4) for o, v in (result.objective_pareto_front or {}).items()
        },
        "front": [
            {
                "index": i,
                "seed": i == seed_index,
                "val": round(scores[i], 4) if i < len(scores) else None,
                "scores": {o: round(subscores[i][o], 4) for o in objectives},
                "tops": [o for o in objectives if i in tops.get(o, ())],
                "chars": sum(len(text) for text in result.candidates[i].values()),
                "components": dict(result.candidates[i]),
            }
            # Discovery order, so the file diffs positionally between runs.
            for i in front
        ],
    }

    if holdout:
        document["holdout"] = holdout

    written = [pareto(name or target.name)] + ([path] if path else [])
    for destination in written:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    _front_table(document, objectives, target.legend())
    for destination in written:
        say(f"          written {destination}")


def _holdout(
    adapter,
    result,
    cases: list[Any],
    seed_index: int | None,
    seed_all: list[float],
) -> dict:
    """Every candidate in the pool on the cases the search never saw.

    Two numbers per candidate beside its training score: the mean over the
    held-out cases, and on how many of them it scored below the seed. The mean
    is the number a run would report about itself; the count is the one the
    gate would have read, because a mean is exactly what lets a candidate buy
    one bad case with several small wins.

    `seed_all` is the seed's score on every case from the split's sweep, in
    the split's order, so its tail is its score on exactly these cases. The
    seed is not scored again: one measurement, made the way every other
    candidate's is.
    """
    say(
        f"\nholdout   {len(cases)} cases the search never saw, scoring "
        f"{len(result.candidates)} candidates on them"
    )
    seed_scores = seed_all[len(seed_all) - len(cases):] if seed_all else []
    per_case = [
        seed_scores if i == seed_index else adapter.evaluate(cases, c).scores
        for i, c in enumerate(result.candidates)
    ]
    train = result.val_aggregate_scores or []

    rows: dict[int, dict[str, Any]] = {}
    for i, scores in enumerate(per_case):
        rows[i] = {
            "train": round(train[i], 4) if i < len(train) else None,
            "holdout": round(sum(scores) / len(scores), 4) if scores else None,
            "worse_than_seed_on": (
                sum(1 for s, t in zip(scores, seed_scores) if s < t)
                if seed_scores and i != seed_index
                else None
            ),
        }

    say(f"\n          {'cand':>4}  {'train':>6}  {'holdout':>7}  worse than the seed on")
    for i, row in rows.items():
        train_cell = f"{row['train']:.3f}" if row["train"] is not None else "-"
        held_cell = f"{row['holdout']:.3f}" if row["holdout"] is not None else "-"
        if i == seed_index:
            note = "seed"
        elif row["worse_than_seed_on"] is None:
            note = "no seed to compare with"
        else:
            note = f"{row['worse_than_seed_on']} of {len(cases)}"
        say(f"          {i:>4}  {train_cell:>6}  {held_cell:>7}  {note}")

    return {"cases": len(cases), "candidates": rows}


def _front_table(
    document: dict, objectives: tuple[str, ...], legend: dict[str, str] | None = None
) -> None:
    """The front on stderr, so it is read without opening anything.

    The rendering lives in `front.py`, which is also what reads a committed
    front back off disk: one table, whether it is printed at the end of a run
    or a month later from the file.
    """
    from tools.gepa import front

    say(f"\npareto    {front.summary(document)}")
    for line in front.render(document, legend=legend):
        # Under the run's labelled lines, so it sits in their gutter.
        say(f"          {line}" if line else line)


# ------------------------------------------------------------------------- diff


def _diff(
    seed: dict[str, str],
    proposed: dict[str, str],
    target: Target,
    *,
    score: float,
    seed_score: float | None,
) -> None:
    """What changed, component by component, beside what it has to keep.

    Only components that moved. Round-robin mutates one per iteration, so a
    four-component winner is usually three unchanged documents and one edit,
    and printing all four buries the edit.
    """
    say()
    for component, before in seed.items():
        after = proposed[component]
        if before == after:
            continue
        for line in difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=target.label(component),
            tofile=f"candidate (val {score:.3f})",
            lineterm="",
        ):
            colour = {"+": "green", "-": "red", "@": "cyan"}.get(
                line[:1] if line[:2] not in ("++", "--") else ""
            )
            say(line, fg=colour)

    before_chars = sum(len(t) for t in seed.values())
    after_chars = sum(len(t) for t in proposed.values())
    against = f" against the seed's {seed_score:.3f}" if seed_score is not None else ""
    say(
        f"\n{before_chars} chars -> {after_chars} "
        f"({(after_chars - before_chars) / before_chars:+.0%}), "
        f"val {score:.3f}{against}\n"
    )

    notes = target.notes()
    if notes:
        say(notes)


if __name__ == "__main__":
    cli()
