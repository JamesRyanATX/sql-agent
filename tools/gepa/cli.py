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
from tools.gepa.gates import say
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
) -> None:
    """GEPA over one searchable thing. The new text goes to stdout.

    \b
      make gepa-extract              read it
      make gepa-extract > new.md     keep it

    Nothing is written to config/prompts/ and nothing is committed.
    """
    chosen = _resolve(target)
    seed_candidate = chosen.seed()

    from tools.gepa.adapter import Loop, reflection_lm

    if probe_only:
        if chosen.check is None:
            raise click.ClickException(f"{chosen.name} has no cheap pre-check")
        with Loop() as loop:
            raise SystemExit(chosen.check(loop))

    _fresh_run_dir(chosen.name, resume)
    cases = _not_too_thin(chosen.corpus(days=days, resume=resume, verbose=verbose))
    trainset, valset = _split(cases, val_fraction, seed)
    budget = chosen.budget if budget is None else budget
    say(f"split     {len(trainset)} train / {len(valset)} val, budget {budget} calls")
    _confirm(chosen, seed_candidate, cases, budget=budget, yes=yes)

    import gepa

    with Loop() as loop:
        result = _search(
            gepa,
            adapter=chosen.adapter(loop),
            reflection=reflection_lm(loop),
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            node=chosen.name,
            budget=budget,
            seed=seed,
            verbose=verbose,
            templates=chosen.templates() if chosen.templates else None,
        )

        say(
            f"search    {result.total_metric_calls} metric calls, "
            f"{len(result.candidates)} candidates in the pool"
        )

        # Before the empty-pool check below, not after. A run that ends with
        # NO_IMPROVEMENT is the one whose evidence is most worth keeping, and
        # on a pool of one the seed's own terms are what say whether the metric
        # is saturated — which is exactly what that message asks you to go and
        # find out.
        _pareto(
            result,
            target=chosen,
            seed_index=_seed_index(result, seed_candidate),
            path=pareto_to,
        )

        # `skip_perfect_score` means a seed at the top of the metric is never
        # mutated. A result, but an empty pool reads as a broken run.
        if len(result.candidates) <= 1:
            say(
                "\nGEPA proposed nothing. Either the seed already scores at the "
                "top of this metric on this corpus — in which case the corpus "
                "or the metric is what needs work — or the budget ran out "
                "before a mutation was accepted.",
                fg="yellow",
            )
            raise SystemExit(NO_IMPROVEMENT)

        survivors = chosen.gate(loop, result, seed_candidate, cases)

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
    say(
        "\nEvery turn runs with the memory off: it reads nothing the agent has "
        "learned and saves nothing it learns. Nothing is written to app/ or "
        "config/, and nothing is committed."
    )

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


def _not_too_thin(cases: list[Any]) -> list[Any]:
    if len(cases) < THIN_CORPUS:
        say(
            f"          {len(cases)} cases is thin — GEPA will fit whichever "
            f"questions happen to be in here",
            fg="yellow",
        )
    return cases


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
):
    """`gepa.optimize`, with its own stdout pointed at stderr: its engine logs
    to stdout, and one line of that is one line in the middle of the prompt.

    `reflection_prompt_template` is a dict keyed by component name, validated
    when the run is constructed — so a template missing its placeholders fails
    here rather than forty minutes in.
    """
    extra = {"reflection_prompt_template": templates} if templates else {}
    with contextlib.redirect_stdout(sys.stderr):
        return gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection,
            max_metric_calls=budget,
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


def _pareto(result, *, target: Target, seed_index: int | None, path: Path | None) -> None:
    """Write the front, and print it. Always to `out/`, and to `path` as well.

    `out/` is gitignored because a harvested case holds a recorded prompt. The
    talk needs a file that survives a clone, so `--pareto` makes that write
    something asked for by name rather than a copy out of the ignored
    directory. What lands there is candidate prose, which a reflection model
    wrote after reading harvested cases — the same exposure the promoted text
    already has, and handled the same way: read the diff.
    """
    from tools.gepa.metric_extract import WEIGHTS

    # Direct attribute access, not getattr: these are real fields on
    # `GEPAResult`, and a default here would mean the tests' fake could drift
    # away from the type without anything noticing.
    subscores = result.val_aggregate_subscores
    if not subscores:
        say("pareto    no per-objective scores on this result — nothing to write")
        return

    # Weight order, not alphabetical, and the same order in every row and every
    # run: the file is committed and diffed.
    objectives = tuple(WEIGHTS)
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
        "weights": dict(WEIGHTS),
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

    written = [pareto(target.name)] + ([path] if path else [])
    for destination in written:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    _front_table(document, objectives)
    for destination in written:
        say(f"          written {destination}")


def _front_table(document: dict, objectives: tuple[str, ...]) -> None:
    """The front on stderr, so it is read without opening anything.

    No colour: the gate spends green and red on pass and fail, and a table where
    every row is one colour is not saying anything.
    """
    pool = document["pool"]
    say(
        f"\npareto    {len(objectives)} objectives over {pool['candidates']} "
        f"candidates, {pool['on_front']} on the front\n"
    )

    width = {o: max(len(o), 6) for o in objectives}
    header = "  ".join(f"{o:>{width[o]}}" for o in objectives)
    say(f"          {'cand':>4}  {'val':>6}  {header}")

    for entry in document["front"]:
        cells = "  ".join(f"{entry['scores'][o]:>{width[o]}.2f}" for o in objectives)
        note = "seed" if entry["seed"] else ""
        if entry["tops"]:
            note = (note + "  " if note else "") + "tops " + ", ".join(entry["tops"])
        val = f"{entry['val']:.3f}" if entry["val"] is not None else "-"
        say(f"          {entry['index']:>4}  {val:>6}  {cells}  {note}".rstrip())

    best = document["best_per_objective"]
    if best:
        cells = "  ".join(
            f"{best.get(o, float('nan')):>{width[o]}.2f}" for o in objectives
        )
        say(f"          {'best':>4}  {'-':>6}  {cells}")


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
