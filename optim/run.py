"""`python -m optim.run <command>`. Five verbs.

    harvest    pull recorded extract calls out of Langfuse into a corpus
    probe      do the current prompts still honour their invariants?
    optimize   GEPA over one node's prompt, gated on the probes
    diff       what the winner changed, beside the invariant checklist
    apply      write the winner into config/prompts/<node>.md

`apply` writes a file and stops: nothing staged, nothing committed, and what it
leaves behind is a working-tree diff a human reads. The reason for a change
belongs in the commit message, where `git log -p config/prompts/` finds it.
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
RUN_DIR = OUT / "run"


def winner(node: str) -> Path:
    """Where `optimize` leaves a gated candidate, for `diff` and `apply`."""
    return OUT / f"candidate-{node}.txt"


def _verdict(node: str) -> Path:
    """What `optimize` recorded about the run that produced `winner(node)`.

    Beside the winner rather than in it: `apply` copies that file verbatim, so a
    header would end up in the prose.
    """
    return OUT / f"candidate-{node}.json"


# Under this, GEPA is fitting noise. A warning rather than a hard stop: the
# right response is usually to go and generate more traces.
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
    from app import tracing
    from optim.harvest import extract_cases

    if not tracing.enabled():
        raise click.ClickException(
            "tracing is off, so there is nothing to harvest — set both "
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, then run some turns"
        )

    # No database: the corpus is entirely what Langfuse recorded, which is what
    # makes it survive `make reset`. See optim/harvest.py.
    result = extract_cases(connection_id=conn, days=days)
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
    help="a prompt file to check instead of the tracked one",
)
def probe(node: str, candidate: Path | None) -> None:
    """Do the prompts still honour the invariants written into them?"""
    from optim.adapter import Loop

    text = candidate.read_text(encoding="utf-8").strip() if candidate else prompts.get(node)
    label = str(candidate or prompts.directory() / f"{node}.md")

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
@click.option("--resume", is_flag=True, help=f"continue the run in {RUN_DIR.name}/")
def optimize(node: str, budget: int, val_fraction: float, seed: int, resume: bool) -> None:
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

    # GEPA resumes from `run_dir` silently if there is anything in it. That is
    # right when continuing an exhausted budget and wrong otherwise: the state is
    # keyed to the seed it started from, so a resume after editing
    # config/prompts/ reports a candidate as descended from a seed that never
    # produced it.
    if not resume and RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise click.ClickException(
            f"{RUN_DIR} holds a previous run and GEPA would resume from it.\n"
            f"  --resume            continue that run\n"
            f"  rm -rf {RUN_DIR}    start fresh (this is what you want after "
            f"editing a prompt or re-harvesting)"
        )

    cases = read_jsonl(CORPUS)
    if len(cases) < THIN_CORPUS:
        click.secho(f"warning: {len(cases)} cases is a thin corpus", fg="yellow")

    trainset, valset = _split(cases, val_fraction, seed)
    click.echo(f"{len(trainset)} train / {len(valset)} val, budget {budget} calls")

    # GEPA sees only harvested data. The probes are NOT the valset: GEPA would
    # score them 0..1 with the metric, and a probe is a predicate. They are a
    # hard gate afterwards, which is the only way to express "never".
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

        # `skip_perfect_score` is on by default, so a seed scoring 1.0 on every
        # sampled minibatch means GEPA declines to mutate. That is a result, but
        # an empty pool reads as a broken run, so say which happened.
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
    seed_score = _seed_score(result, seed_prompt)
    target = winner(node)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(best["text"], encoding="utf-8")
    # The gate's verdict, where `apply` can read it — `apply` refuses a winner
    # it cannot see a passing gate for, so a file hand-dropped into optim/out/
    # cannot be promoted as though a run had cleared it.
    _verdict(node).write_text(
        json.dumps(
            {
                "node": node,
                "score": best["score"],
                "seed_score": seed_score,
                "candidate": best["index"],
                "gated": True,
                "probes": sorted(p.name for p in probes.load(node)),
                "seed_fingerprint": prompts.fingerprint().get(node),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    click.echo(f"\nwrote {target} (val score {best['score']:.3f})")

    # `_gate` ranks survivors against each other and never against the prompt
    # they came from, so when GEPA's best program *is* the seed it still hands
    # back the best mutation and this still writes it. The comparison has to be
    # made here and carried in the verdict, because `apply` is one command.
    if seed_score is not None and best["score"] < seed_score:
        click.secho(
            f"\nBut the seed scored {seed_score:.3f} and this scored "
            f"{best['score']:.3f} — the run found nothing better than what is "
            f"already in config/prompts/{node}.md. `apply` will refuse it "
            f"without --force, and forcing it promotes a measured regression.",
            fg="yellow",
        )
        return
    click.echo(
        "read it with `python -m optim.run diff`, then `python -m optim.run "
        f"apply --node {node}` to write it into config/prompts/{node}.md. "
        "Nothing is applied and nothing is committed."
    )


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

    # `gather` is built *inside* the coroutine rather than passed into
    # `loop.run` already constructed: it needs a running loop to attach its
    # future to, and the caller is on the main thread where there is none.
    async def all_of_them() -> list[Outcome]:
        return list(await asyncio.gather(*(one(p) for p in all_probes)))

    return loop.run(all_of_them())


def _seed_score(result, seed_prompt: str) -> float | None:
    """What the *current* prompt scored on the valset, for comparison.

    GEPA's best program is frequently the seed, and `_gate` cannot see that
    because it skips the seed. None when there is no score to compare.
    """
    from optim.adapter import COMPONENT

    scores = result.val_aggregate_scores or []
    for i, candidate in enumerate(result.candidates):
        if candidate[COMPONENT] == seed_prompt and i < len(scores):
            return scores[i]
    return None


def _gate(loop, result, seed_prompt: str, node: str) -> list[dict]:
    """Re-check every candidate in the pool. Regressions are disqualifying.

    Outside GEPA's objective, because weights cannot express "never": a
    mean-maximising search trades a rare catastrophic failure for a broad small
    gain whenever the arithmetic allows.
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

        # Not rejection — a shorter prompt can be the right answer. But a
        # candidate that won by deleting most of the instruction has to be seen.
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
    source = winner(node)
    if not source.exists():
        raise click.ClickException(f"no candidate at {source} — run `optimize` first")

    seed = prompts.get(node)
    proposed = source.read_text(encoding="utf-8").strip()

    for line in difflib.unified_diff(
        seed.splitlines(), proposed.splitlines(),
        fromfile=str(prompts.directory() / f"{node}.md"), tofile=str(source),
        lineterm="",
    ):
        colour = {"+": "green", "-": "red", "@": "cyan"}.get(line[:1] if line[:2] not in ("++", "--") else "")
        click.secho(line, fg=colour)

    click.echo(f"\n{len(seed)} chars -> {len(proposed)} "
               f"({(len(proposed) - len(seed)) / len(seed):+.0%})\n")
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
    click.echo(f"\n`apply --node {node}` writes it into {prompts.directory() / f'{node}.md'}.")


# ------------------------------------------------------------------------ apply


def _git(*args: str) -> str:
    """git, or "" if this is not a checkout. Never raises."""
    import subprocess

    try:
        done = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return done.stdout if done.returncode == 0 else ""


@cli.command()
@click.option("--node", default="extract", show_default=True)
@click.option(
    "--force",
    is_flag=True,
    help="write over a dirty target, or promote a candidate that scored below the seed",
)
def apply(node: str, force: bool) -> None:
    """Write the gated winner into config/prompts/<node>.md, and stop: nothing
    staged, nothing committed, a working-tree diff left behind.

    Three refusals. A promoted prompt must be traceable to a run that cleared
    the probes (`gated: true`); it must not have measured worse than the prompt
    it replaces (`seed_score`); and the target must be clean, so the diff a
    human reads is the one this wrote.
    """
    source = winner(node)
    if not source.exists():
        raise click.ClickException(f"no candidate at {source} — run `optimize` first")

    verdict_path = _verdict(node)
    if not verdict_path.exists():
        raise click.ClickException(
            f"no gate verdict at {verdict_path}. {source} exists, but nothing "
            f"records that it survived the probes — and a candidate that did "
            f"not clear the gate is exactly what the gate is for. Re-run "
            f"`optimize --node {node}`."
        )
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    if not verdict.get("gated"):
        raise click.ClickException(f"{verdict_path} does not record a passing gate")

    # A candidate can clear every probe and still be worse than what it was
    # mutated from: the probes are predicates about failures, not a measure of
    # quality. Promoting a measured regression takes --force.
    seed_score, score = verdict.get("seed_score"), verdict.get("score")
    if not force and seed_score is not None and score is not None and score < seed_score:
        raise click.ClickException(
            f"the winner scored {score:.3f} and the prompt already in "
            f"config/prompts/{node}.md scored {seed_score:.3f}. The run found "
            f"nothing better. Applying this promotes a measured regression — "
            f"pass --force only if you have read the diff and want it anyway."
        )

    target = prompts.directory() / f"{node}.md"
    if not target.is_file():
        raise click.ClickException(f"no prompt at {target} — is CONFIG_DIR right?")

    # `git status --porcelain <path>` is empty for a clean tracked file, and
    # also for "not a git checkout" — hence the outer check, since refusing to
    # work outside a repository would be a rule about tooling, not the prompt.
    if not force and _git("rev-parse", "--git-dir"):
        if _git("status", "--porcelain", "--", str(target)).strip():
            raise click.ClickException(
                f"{target} has uncommitted changes. Commit or discard them "
                f"first, so the diff this leaves is the one it wrote — or pass "
                f"--force if you meant to write over them."
            )

    proposed = source.read_text(encoding="utf-8").strip()
    before = target.read_text(encoding="utf-8").strip()
    if proposed == before:
        click.echo(f"{target} already says exactly this. Nothing to do.")
        return

    target.write_text(proposed + "\n", encoding="utf-8")

    # The provenance the file body cannot carry, since the whole file is the
    # prompt. A line here is a machine's claim about a score; the commit beside
    # it is the human's claim about why.
    notes = prompts.directory() / prompts.NOTES
    if notes.is_file():
        stamp = (_git("log", "-1", "--format=%cs") or "").strip() or "undated"
        with notes.open("a", encoding="utf-8") as fh:
            fh.write(
                f"\n- `{node}.md` — GEPA candidate {verdict.get('candidate')}, "
                f"val {verdict.get('score', 0.0):.3f}, from prompt "
                f"`{verdict.get('seed_fingerprint')}`. Applied on or after {stamp}.\n"
            )

    click.secho(f"wrote {target}", fg="green")
    click.echo(
        f"\nNothing is staged and nothing is committed. Read it:\n"
        f"    git diff -- {target}\n"
        f"Then commit, and put in the message which failure the new wording "
        f"addresses — that reason is the one thing no run can produce, and the "
        f"file body can no longer hold a comment."
    )


if __name__ == "__main__":
    cli()
