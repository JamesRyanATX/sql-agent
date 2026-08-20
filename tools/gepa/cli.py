"""`python -m tools.gepa <node>` — GEPA over one node's prompt.

One command, needing no other: it harvests the telemetry, searches, gates the
pool on the probes and prints the winner. Fresh corpus and fresh run dir every
time, so a run is never partly made of the last one.

stdout is the new prompt and nothing else. Progress, the diff and the gate go to
stderr, so `make gepa-extract > new.md` leaves prose with no commentary.

Nothing is written to config/prompts/ and nothing is committed.

Exit: 0 a prompt is on stdout, 1 preflight, 2 the node has no metric,
3 nothing scored better than what is already there. Through `make` these all
arrive as make's own 2 — an empty stdout is the portable signal.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from app import prompts
from tools.gepa import probes
from tools.gepa.cases import ExtractCase, read_jsonl, write_jsonl

# The tests' one monkeypatch point, hence the two paths below deriving from it.
OUT = Path(__file__).resolve().parent / "out"

UNWIRED_NODE = 2
NO_IMPROVEMENT = 3

# Under this, GEPA is fitting noise.
THIN_CORPUS = 12

WIRED = frozenset({"extract"})

# Why the others are not. Two are arguments against building the thing.
UNWIRED = {
    "plan": (
        "`plan` needs outcome labelling — whether the SQL ran, how many fix\n"
        "attempts it took — so the metric is not a function of the recorded\n"
        "call. Its degenerate optimum is the worst in the graph: always say\n"
        "sufficient."
    ),
    "explore": (
        "One metric call for `explore` is a whole turn: ~11.5k tokens against\n"
        "the ~150 rollouts a search wants."
    ),
    "answer": (
        "`answer`'s only cheap proxy is length, and that trade was measured on\n"
        "the quarters question — 353 output tokens before, 278 after — and\n"
        "decided the other way. See config/prompts/README.md."
    ),
    "generate_sql": (
        "Scoring one call means running the SQL it wrote against a warehouse\n"
        "whose answers are known."
    ),
    "fix": (
        "As `generate_sql`, plus a corpus problem: a fix case only exists where\n"
        "a turn errored, and the recorded ones are too few and too alike."
    ),
}


def corpus(node: str) -> Path:
    return OUT / f"{node}.jsonl"


def run_dir(node: str) -> Path:
    return OUT / "run" / node


def say(message: str = "", **kwargs) -> None:
    """Narration. Every print in this module goes through here, so stdout stays
    the artifact — the promise breaks on one `click.echo` written in a hurry."""
    click.secho(message, err=True, **kwargs)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("node")
@click.option("-v", "--verbose", is_flag=True, help="Per-probe and per-candidate detail.")
@click.option("--budget", default=150, show_default=True, help="max metric calls")
@click.option("--val-fraction", default=0.3, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option(
    "--resume",
    is_flag=True,
    help="continue the last run: its corpus and its GEPA state, no harvest",
)
@click.option("--probe-only", is_flag=True, help="check the invariants and stop")
@click.option("-c", "--conn", default="default", help="which connection to harvest")
@click.option("--days", default=30, show_default=True, help="how far back to harvest")
def cli(
    node: str,
    verbose: bool,
    budget: int,
    val_fraction: float,
    seed: int,
    resume: bool,
    probe_only: bool,
    conn: str,
    days: int,
) -> None:
    """GEPA over one node's prompt. The new prose goes to stdout.

    \b
      make gepa-extract              read it
      make gepa-extract > new.md     keep it

    Nothing is written to config/prompts/ and nothing is committed.
    """
    _check_node(node)
    seed_prompt = prompts.get(node)

    from tools.gepa.adapter import COMPONENT, ExtractAdapter, Loop, reflection_lm

    if probe_only:
        with Loop() as loop:
            outcomes = _run_probes(loop, seed_prompt, probes.load(node))
        raise SystemExit(1 if _report(outcomes, _seed_label(node)) else 0)

    _fresh_run_dir(node, resume)
    cases = _corpus(node, conn=conn, days=days, resume=resume, verbose=verbose)
    trainset, valset = _split(cases, val_fraction, seed)
    say(f"split     {len(trainset)} train / {len(valset)} val, budget {budget} calls")

    import gepa

    with Loop() as loop:
        result = _search(
            gepa,
            adapter=ExtractAdapter(loop),
            reflection=reflection_lm(loop),
            seed_candidate={COMPONENT: seed_prompt},
            trainset=trainset,
            valset=valset,
            node=node,
            budget=budget,
            seed=seed,
            verbose=verbose,
        )

        say(
            f"search    {result.total_metric_calls} metric calls, "
            f"{len(result.candidates)} candidates in the pool"
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

        survivors = _gate(loop, result, seed_prompt, node)

    if not survivors:
        say(
            "\nEvery candidate regressed a probe the seed passed. That is the "
            "gate working: the trainset metric cannot see these failures, "
            "because they surface turns later.",
            fg="red",
        )
        raise SystemExit(NO_IMPROVEMENT)

    best = survivors[0]
    seed_score = _seed_score(result, seed_prompt, COMPONENT)

    # Clearing every probe does not make a candidate better than what it would
    # replace. Observed: a run scored the seed 0.959 and its best survivor 0.928.
    if seed_score is not None and best["score"] < seed_score:
        say(
            f"\nThe seed scored {seed_score:.3f} and the best survivor "
            f"{best['score']:.3f} — nothing better than "
            f"{prompts.directory() / f'{node}.md'}, so stdout is empty.",
            fg="yellow",
        )
        raise SystemExit(NO_IMPROVEMENT)

    _diff(seed_prompt, best["text"], node, score=best["score"], seed_score=seed_score)

    # The one write to stdout.
    sys.stdout.write(best["text"].strip() + "\n")


# -------------------------------------------------------------------- preflight
#
# Everything that can refuse the run refuses it here, before a token is spent:
# `apply` used to check its preconditions after a ten-minute search.


def _check_node(node: str) -> None:
    if node not in prompts.NODES:
        raise click.ClickException(
            f"no prompt named {node!r} — config/prompts/ holds "
            f"{', '.join(sorted(prompts.NODES))}"
        )
    if node in WIRED:
        return

    say(f"`{node}` has no metric, so there is nothing to search.\n", fg="yellow")
    say(UNWIRED[node])
    say("\nWiring a node up is four things, and the first is the hard one:")
    say(f"  a metric scoring one recorded call    tools/gepa/metric_{node}.py")
    say("  a replay that makes that call         tools/gepa/replay.py")
    say(f"  probes for its prose-only invariants  tests/probes/{node}/*.json")
    say("  a harvest that builds its corpus      tools/gepa/harvest.py")
    raise SystemExit(UNWIRED_NODE)


def _fresh_run_dir(node: str, resume: bool) -> None:
    """Wiped rather than refused. GEPA resumes from a run dir silently and its
    state is keyed to the seed and trainset it started from, so the leftovers of
    one run are never what the next one wants — and telling the operator to
    `rm -rf` it themselves makes a one-command target a two-command one."""
    directory = run_dir(node)
    if resume or not directory.exists():
        return
    shutil.rmtree(directory)


def _corpus(
    node: str, *, conn: str, days: int, resume: bool, verbose: bool
) -> list[ExtractCase]:
    """Harvested every run, so a search always scores the prompt against what
    the agent has actually been doing. Only `--resume` reuses what is on disk:
    GEPA's state references its trainset, so resuming onto a re-harvested corpus
    would be a run made of two different populations."""
    path = corpus(node)
    if resume and path.exists():
        cases = read_jsonl(path)
        say(f"corpus    {len(cases)} cases reused from {path}")
        return _not_too_thin(cases)

    from app import tracing
    from tools.gepa.harvest import extract_cases

    if not tracing.enabled():
        raise click.ClickException(
            "tracing is off, so there is nothing to harvest — set both "
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (make langfuse-up), "
            "run some turns, then try again."
        )

    harvested = extract_cases(connection_id=conn, days=days)
    if verbose:
        say(harvested.report())
    else:
        say(f"harvest   {len(harvested.cases)} cases from {harvested.seen} "
            f"recorded {node} calls (-v for what was dropped)")
    if not harvested.cases:
        raise click.ClickException(
            f"no cases — has connection {conn!r} run any turns since tracing "
            "was switched on?"
        )

    write_jsonl(path, harvested.cases)
    return _not_too_thin(harvested.cases)


def _not_too_thin(cases: list[ExtractCase]) -> list[ExtractCase]:
    if len(cases) < THIN_CORPUS:
        say(
            f"          {len(cases)} cases is thin — GEPA will fit whichever "
            f"questions happen to be in here",
            fg="yellow",
        )
    return cases


def _split(
    cases: list[ExtractCase], fraction: float, seed: int
) -> tuple[list[ExtractCase], list[ExtractCase]]:
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
    trainset: list[ExtractCase],
    valset: list[ExtractCase],
    node: str,
    budget: int,
    seed: int,
    verbose: bool,
):
    """`gepa.optimize`, with its own stdout pointed at stderr: its engine logs
    to stdout, and one line of that is one line in the middle of the prompt."""
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
        )


# ------------------------------------------------------------------- the gate


@dataclass
class Outcome:
    probe: probes.Probe
    ok: bool
    reason: str


def _run_probes(loop, text: str, all_probes: list[probes.Probe]) -> list[Outcome]:
    from tools.gepa.replay import replay

    async def one(p: probes.Probe) -> Outcome:
        ok, reason = probes.check(p, await replay(text, p.case))
        return Outcome(p, ok, reason)

    # `gather` is built inside the coroutine: it needs a running loop to attach
    # its future to, and the caller is on the main thread where there is none.
    async def all_of_them() -> list[Outcome]:
        return list(await asyncio.gather(*(one(p) for p in all_probes)))

    return loop.run(all_of_them())


def _seed_score(result, seed_prompt: str, component: str) -> float | None:
    """What the current prompt scored on the valset. `_gate` skips the seed, so
    it cannot see that GEPA's best program is frequently that seed."""
    scores = result.val_aggregate_scores or []
    for i, candidate in enumerate(result.candidates):
        if candidate[component] == seed_prompt and i < len(scores):
            return scores[i]
    return None


def _gate(loop, result, seed_prompt: str, node: str) -> list[dict]:
    """Re-check every candidate. Regressions are disqualifying.

    Outside GEPA's objective, because weights cannot express "never": a
    mean-maximising search trades a rare catastrophic failure for a broad small
    gain whenever the arithmetic allows.
    """
    from tools.gepa.adapter import COMPONENT

    all_probes = probes.load(node)
    say("\nchecking the pool against the invariant probes")

    baseline = _run_probes(loop, seed_prompt, all_probes)
    seed_passing = {o.probe.name for o in baseline if o.ok}
    say(f"  seed passes {len(seed_passing)}/{len(all_probes)}: "
        f"{sorted(seed_passing) or '(none)'}")

    scores = result.val_aggregate_scores or []
    survivors = []
    for i, candidate in enumerate(result.candidates):
        text = candidate[COMPONENT]
        if text == seed_prompt:
            continue
        outcomes = _run_probes(loop, text, all_probes)
        regressions = [o for o in outcomes if not o.ok and o.probe.name in seed_passing]
        score = scores[i] if i < len(scores) else 0.0

        if regressions:
            say(
                f"  candidate {i} (val {score:.3f}) DISCARDED — regressed "
                f"{[o.probe.name for o in regressions]}",
                fg="red",
            )
            for o in regressions:
                say(f"      {o.probe.invariant}")
                say(f"      {o.reason.strip()}")
            continue

        # Not rejection — a shorter prompt can be right. But one that won by
        # deleting most of the instruction has to be seen.
        if len(text) < 0.6 * len(seed_prompt):
            say(
                f"  candidate {i} (val {score:.3f}) is {len(text)} chars against "
                f"the seed's {len(seed_prompt)} — read the diff closely",
                fg="yellow",
            )
        else:
            say(f"  candidate {i} (val {score:.3f}) survives", fg="green")
        survivors.append({"index": i, "text": text, "score": score})

    return sorted(survivors, key=lambda s: -s["score"])


def _seed_label(node: str) -> str:
    return str(prompts.directory() / f"{node}.md")


def _report(outcomes: list[Outcome], label: str) -> list[Outcome]:
    failures = [o for o in outcomes if not o.ok]
    say(f"{label}\n")
    for o in outcomes:
        mark, colour = ("PASS", "green") if o.ok else ("FAIL", "red")
        say(f"  {mark}  {o.probe.name}", fg=colour)
        say(f"        {o.probe.invariant}")
        if not o.ok:
            say("".join(f"      {line}\n" for line in o.reason.splitlines()))
    say()
    if failures:
        say(f"{len(failures)} of {len(outcomes)} probes failed", fg="red")
    else:
        say(f"all {len(outcomes)} probes pass", fg="green")
    return failures


# ------------------------------------------------------------------------- diff


def _diff(
    seed: str, proposed: str, node: str, *, score: float, seed_score: float | None
) -> None:
    """What changed, beside the invariants it has to keep."""
    say()
    for line in difflib.unified_diff(
        seed.splitlines(), proposed.splitlines(),
        fromfile=_seed_label(node), tofile=f"candidate (val {score:.3f})",
        lineterm="",
    ):
        colour = {"+": "green", "-": "red", "@": "cyan"}.get(
            line[:1] if line[:2] not in ("++", "--") else ""
        )
        say(line, fg=colour)

    against = f" against the seed's {seed_score:.3f}" if seed_score is not None else ""
    say(f"\n{len(seed)} chars -> {len(proposed)} "
        f"({(len(proposed) - len(seed)) / len(seed):+.0%}), val {score:.3f}{against}\n")
    say("the invariants this prompt is the only home for:")
    for p in probes.load(node):
        say(f"  - {p.invariant}")
        say(f"    {p.cites}")
    say(
        "\nA dropped sentence passing the probes does not mean the sentence was "
        "doing nothing: cache rot surfaces turns later, outside anything "
        "measured here.\n"
    )


if __name__ == "__main__":
    cli()
