"""`python -m optim.run <command>`. Four verbs, and deliberately not a fifth.

    harvest    pull recorded extract calls out of Langfuse into a corpus
    probe      do the current prompts still honour their invariants?
    optimize   GEPA over one node's prompt, gated on the probes
    diff       what the winner changed, beside the invariant checklist

There is no `apply`. Writing a prompt into `app/prompts.py` is a git-visible
edit a human makes after reading the diff — a machine-applied prompt arrives
without the comment explaining why it says what it says, in a repo whose stated
convention is that comments cite the failure that motivated the code. The last
mile is a person writing that comment.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import click

from app import prompts
from optim import probes
from optim.cases import ExtractCase, read_jsonl, write_jsonl

OUT = Path(__file__).resolve().parent / "out"
CORPUS = OUT / "extract.jsonl"
WINNER = OUT / "candidate-extract.txt"
RUN_DIR = OUT / "run"

# Under this, GEPA is fitting noise. Not a hard stop — a warning, because the
# right response is usually to go and generate more traces rather than to
# abandon the run.
THIN_CORPUS = 12


@click.group()
def cli() -> None:
    """Prompt evaluation and search."""


# ---------------------------------------------------------------------- harvest


@cli.command()
@click.option("--conn", "-c", default="default", help="which registered connection")
@click.option("--days", default=30, show_default=True, help="how far back to look")
def harvest(conn: str, days: int) -> None:
    """Pull recorded `extract` calls out of Langfuse into a corpus."""
    from app import db, tracing
    from optim.harvest import extract_cases

    if not tracing.enabled():
        raise click.ClickException(
            "tracing is off, so there is nothing to harvest — set both "
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, then run some turns"
        )

    async def go():
        # The API opens these in its lifespan and the tests in a fixture; a
        # command-line tool is the third caller and has to do it itself.
        await db.open_pools()
        try:
            return await extract_cases(connection_id=conn, days=days)
        finally:
            await db.close_pools()

    result = asyncio.run(go())
    click.echo(result.report())
    if not result.cases:
        raise click.ClickException("no cases — has this connection run any turns?")

    write_jsonl(CORPUS, result.cases)
    click.echo(f"wrote {CORPUS}")
    if len(result.cases) < THIN_CORPUS:
        click.secho(
            f"\n{len(result.cases)} cases is thin. GEPA will fit whichever "
            f"questions happen to be in here — ask more, or accept that the "
            f"result is about this handful.",
            fg="yellow",
        )


# ------------------------------------------------------------------------ probe


@cli.command()
@click.option("--node", default="extract", show_default=True)
@click.option(
    "--candidate",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="a prompt file to check instead of the one in app/prompts.py",
)
def probe(node: str, candidate: Path | None) -> None:
    """Do the prompts still honour the invariants written into them?"""
    from optim.adapter import Loop

    text = candidate.read_text(encoding="utf-8").strip() if candidate else prompts.get(node)
    label = str(candidate) if candidate else "app/prompts.py"

    with Loop() as loop:
        outcomes = _run_probes(loop, text, probes.load(node))

    failures = _report(outcomes, label)
    if failures:
        raise SystemExit(1)


# --------------------------------------------------------------------- optimize


@cli.command()
@click.option("--node", default="extract", show_default=True)
@click.option("--budget", default=150, show_default=True, help="max metric calls")
@click.option("--val-fraction", default=0.3, show_default=True)
@click.option("--seed", default=0, show_default=True)
def optimize(node: str, budget: int, val_fraction: float, seed: int) -> None:
    """GEPA over one node's prompt, then the probe gate."""
    import gepa

    from optim.adapter import COMPONENT, ExtractAdapter, Loop, reflection_lm

    if node != COMPONENT:
        raise click.ClickException(
            f"only {COMPONENT!r} is wired up. `plan` needs outcome labelling and "
            "SQL execution against a deterministic warehouse; `explore` costs a "
            "whole turn per metric call; `answer` has no cheap proxy but length."
        )
    if not CORPUS.exists():
        raise click.ClickException(f"no corpus at {CORPUS} — run `harvest` first")

    cases = read_jsonl(CORPUS)
    if len(cases) < THIN_CORPUS:
        click.secho(f"warning: {len(cases)} cases is a thin corpus", fg="yellow")

    trainset, valset = _split(cases, val_fraction, seed)
    click.echo(f"{len(trainset)} train / {len(valset)} val, budget {budget} calls")

    # GEPA sees only harvested data. The probes are deliberately NOT the valset:
    # GEPA would score them 0..1 with the metric, and a probe is a predicate —
    # partial credit on "did it record a census" is not a thing. They are a hard
    # gate afterwards instead, which is also the only way to express "never".
    seed_prompt = prompts.get(node)

    with Loop() as loop:
        adapter = ExtractAdapter(loop)
        result = gepa.optimize(
            seed_candidate={COMPONENT: seed_prompt},
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm(loop),
            max_metric_calls=budget,
            run_dir=str(RUN_DIR),
            seed=seed,
            display_progress_bar=True,
        )

        click.echo(f"\n{result.total_metric_calls} metric calls, "
                   f"{len(result.candidates)} candidates in the pool")

        # `skip_perfect_score` is on by default, so a seed that already scores
        # 1.0 on every sampled minibatch means GEPA declines to mutate at all.
        # That is a result, not a broken run — but an empty pool reads as one,
        # so say which happened.
        if len(result.candidates) <= 1:
            click.secho(
                "\nGEPA proposed nothing. Either the seed already scores at the "
                "top of this metric on this corpus — in which case the corpus "
                "or the metric is what needs work, not the prompt — or the "
                "budget ran out before a mutation was accepted.",
                fg="yellow",
            )
            return

        survivors = _gate(loop, result, seed_prompt, node)

    if not survivors:
        raise click.ClickException(
            "every candidate regressed a probe the seed passed. That is the "
            "gate working: the trainset metric cannot see the failures these "
            "defend, because they surface turns later."
        )

    best = survivors[0]
    WINNER.parent.mkdir(parents=True, exist_ok=True)
    WINNER.write_text(best["text"], encoding="utf-8")
    click.echo(f"\nwrote {WINNER} (val score {best['score']:.3f})")
    click.echo("read it, then `python -m optim.run diff`. Nothing is applied.")


def _split(
    cases: list[ExtractCase], fraction: float, seed: int
) -> tuple[list[ExtractCase], list[ExtractCase]]:
    if len(cases) < 4:
        click.secho(
            "too few cases to hold anything out — training and validating on "
            "the same rows, so the val score is not evidence of generalisation",
            fg="yellow",
        )
        return cases, cases
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * fraction))
    return shuffled[cut:], shuffled[:cut]


# ------------------------------------------------------------------- the gate


@dataclass
class Outcome:
    probe: probes.Probe
    ok: bool
    reason: str


def _run_probes(loop, text: str, all_probes: list[probes.Probe]) -> list[Outcome]:
    from optim.replay import replay

    async def one(p: probes.Probe) -> Outcome:
        ok, reason = probes.check(p, await replay(text, p.case))
        return Outcome(p, ok, reason)

    # `gather` is built *inside* the coroutine, not passed into `loop.run`
    # already constructed: it needs a running loop to attach its future to, and
    # the caller is on the main thread where there is none. Same family as the
    # bug `Loop` exists to prevent, and it fails immediately rather than
    # intermittently only because the probes run before any other loop work.
    async def all_of_them() -> list[Outcome]:
        return list(await asyncio.gather(*(one(p) for p in all_probes)))

    return loop.run(all_of_them())


def _gate(loop, result, seed_prompt: str, node: str) -> list[dict]:
    """Re-check every candidate in the pool. Regressions are disqualifying.

    Outside GEPA's objective on purpose. Weights cannot express "never": a
    mean-maximising search will trade a rare catastrophic failure for a broad
    small gain whenever the arithmetic allows, and the failures these probes
    defend are exactly the ones the trainset metric cannot see — a deleted
    census paragraph costs nothing today and poisons the cache next week.
    """
    from optim.adapter import COMPONENT

    all_probes = probes.load(node)
    click.echo("\nchecking the pool against the invariant probes")

    baseline = _run_probes(loop, seed_prompt, all_probes)
    seed_passing = {o.probe.name for o in baseline if o.ok}
    click.echo(f"  seed passes {len(seed_passing)}/{len(all_probes)}: "
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
            click.secho(
                f"  candidate {i} (val {score:.3f}) DISCARDED — regressed "
                f"{[o.probe.name for o in regressions]}",
                fg="red",
            )
            for o in regressions:
                click.echo(f"      {o.probe.invariant}")
                click.echo(f"      {o.reason.strip()}")
            continue

        # Not rejection. ANSWER_SYSTEM's comment shows this team has shortened a
        # prompt on purpose and measured it. But a candidate that won by
        # deleting most of the instruction has to be *seen*.
        if len(text) < 0.6 * len(seed_prompt):
            click.secho(
                f"  candidate {i} (val {score:.3f}) is {len(text)} chars against "
                f"the seed's {len(seed_prompt)} — read the diff closely",
                fg="yellow",
            )
        else:
            click.secho(f"  candidate {i} (val {score:.3f}) survives", fg="green")
        survivors.append({"index": i, "text": text, "score": score})

    return sorted(survivors, key=lambda s: -s["score"])


def _report(outcomes: list[Outcome], label: str) -> list[Outcome]:
    failures = [o for o in outcomes if not o.ok]
    click.echo(f"{label}\n")
    for o in outcomes:
        mark, colour = ("PASS", "green") if o.ok else ("FAIL", "red")
        click.secho(f"  {mark}  {o.probe.name}", fg=colour)
        click.echo(f"        {o.probe.invariant}")
        if not o.ok:
            click.echo("".join(f"      {line}\n" for line in o.reason.splitlines()))
    click.echo()
    if failures:
        click.secho(f"{len(failures)} of {len(outcomes)} probes failed", fg="red")
    else:
        click.secho(f"all {len(outcomes)} probes pass", fg="green")
    return failures


# ------------------------------------------------------------------------- diff


@cli.command()
@click.option("--node", default="extract", show_default=True)
def diff(node: str) -> None:
    """What the winner changed, beside the invariants it has to keep."""
    if not WINNER.exists():
        raise click.ClickException(f"no candidate at {WINNER} — run `optimize` first")

    seed = prompts.get(node)
    winner = WINNER.read_text(encoding="utf-8").strip()

    for line in difflib.unified_diff(
        seed.splitlines(), winner.splitlines(),
        fromfile=f"app/prompts.py::{node}", tofile=str(WINNER), lineterm="",
    ):
        colour = {"+": "green", "-": "red", "@": "cyan"}.get(line[:1] if line[:2] not in ("++", "--") else "")
        click.secho(line, fg=colour)

    click.echo(f"\n{len(seed)} chars -> {len(winner)} "
               f"({(len(winner) - len(seed)) / len(seed):+.0%})\n")
    click.echo("the invariants this prompt is the only home for:")
    for p in probes.load(node):
        click.echo(f"  - {p.invariant}")
        click.echo(f"    {p.cites}")
    click.echo(
        "\nRead the diff against that list. If the winner dropped a sentence, "
        "the probe suite says it still behaves — it does not say the sentence "
        "was doing nothing. Cache rot surfaces turns later, outside anything "
        "measured here."
    )


if __name__ == "__main__":
    cli()
