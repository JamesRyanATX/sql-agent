"""What disqualifies a candidate, whatever it scored.

Outside GEPA's objective, and that is the whole point. Weights cannot express
"never": a mean-maximising search trades a rare catastrophic failure for a broad
small gain the moment the arithmetic allows it, and the failures worth refusing
here are exactly the rare catastrophic ones.

Two gates, because there are two kinds of thing to defend, and they parallel
rather than share. `probe_gate` checks invariants that exist only as prose in a
prompt, by replaying authored cases: a recorded count going stale, a recipe
grounded in a fragment too short to be wrong. Those have no ground truth, so an
authored predicate is the only witness there is. `turn_gate` checks questions
whose answers are known, which is the stronger thing to have and is available
only where a golden corpus exists.

Generalising the probe machinery to cover both was the alternative, and it is
not worth it: a probe's case is an `ExtractCase` and every predicate reads a
`Replayed`, so it is extract-shaped end to end. Both gates return the same type
and take the same arguments, which is all `cli.py` needs to not care.

Moved out of `cli.py` so `targets.py` can name a gate without importing the
command.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import click

from tools.gepa import probes


@dataclass(frozen=True)
class Survivor:
    """A candidate the gate did not throw out."""

    index: int
    candidate: dict[str, str]
    score: float


@dataclass
class Outcome:
    probe: probes.Probe
    ok: bool
    reason: str


def say(message: str = "", **kwargs) -> None:
    """Narration. Every print in this package goes through one of these, so
    stdout stays the artifact."""
    click.secho(message, err=True, **kwargs)


# ------------------------------------------------------------- authored probes


def run_probes(loop, text: str, all_probes: list[probes.Probe]) -> list[Outcome]:
    from tools.gepa.replay import replay

    async def one(p: probes.Probe) -> Outcome:
        ok, reason = probes.check(p, await replay(text, p.case))
        return Outcome(p, ok, reason)

    # `gather` is built inside the coroutine: it needs a running loop to attach
    # its future to, and the caller is on the main thread where there is none.
    async def all_of_them() -> list[Outcome]:
        return list(await asyncio.gather(*(one(p) for p in all_probes)))

    return loop.run(all_of_them())


def probe_gate(loop, result, seed: dict[str, str], *, node: str) -> list[Survivor]:
    """Re-check every candidate against the authored probes.

    A probe the seed already fails cannot disqualify anybody: the baseline is
    the seed's pass set, not the whole list. Otherwise a known-failing invariant
    would empty the pool every run and the gate would say nothing.
    """
    component = next(iter(seed))
    seed_text = seed[component]

    all_probes = probes.load(node)
    say("\nchecking the pool against the invariant probes")

    baseline = run_probes(loop, seed_text, all_probes)
    passing = {o.probe.name for o in baseline if o.ok}
    say(f"  seed passes {len(passing)}/{len(all_probes)}: {sorted(passing) or '(none)'}")

    scores = result.val_aggregate_scores or []
    survivors: list[Survivor] = []
    for i, candidate in enumerate(result.candidates):
        text = candidate[component]
        if text == seed_text:
            continue
        outcomes = run_probes(loop, text, all_probes)
        regressions = [o for o in outcomes if not o.ok and o.probe.name in passing]
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

        # Neither direction is rejection — a shorter prompt can be right, and
        # so can a longer one. Both have to be *seen*, and the growth half is
        # the one the metric is blind to: `metric_extract._cost` charges the
        # model's output tokens against a recorded baseline, and a prompt is
        # input. So a candidate can quadruple the instruction, pay nothing in
        # the score, and cost real money on every turn it runs afterwards.
        # Observed on the first real run: 2,076 chars to 7,890, +280%, and
        # every term said it was better.
        if len(text) < 0.6 * len(seed_text) or len(text) > 1.5 * len(seed_text):
            grew = (len(text) - len(seed_text)) / len(seed_text)
            say(
                f"  candidate {i} (val {score:.3f}) is {len(text)} chars against "
                f"the seed's {len(seed_text)} ({grew:+.0%}) — read the diff "
                f"closely; prompt length is not in the metric",
                fg="yellow",
            )
        else:
            say(f"  candidate {i} (val {score:.3f}) survives", fg="green")
        survivors.append(Survivor(index=i, candidate=candidate, score=score))

    return sorted(survivors, key=lambda s: -s.score)


def report(outcomes: list[Outcome], label: str) -> list[Outcome]:
    """Every probe, pass or fail, with the invariant it defends. Returns the
    failures so a caller can exit on them."""
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


# --------------------------------------------------------------- known answers


def _asked(valset, key) -> str:
    """The question behind one of GEPA's validation scores.

    `val_subscores` is keyed by *position in the valset*, not by anything this
    repo names. The first version was handed the whole corpus and looked the key
    up by case name, so every lookup missed and a discarded candidate was
    reported as having lost case `2` — an index, printed where a question
    belongs. The tests passed because their fixture used the keys I assumed
    rather than the ones GEPA uses.
    """
    if isinstance(key, int) and 0 <= key < len(valset):
        return getattr(valset[key], "question", str(key))
    return str(key)


def turn_gate(loop, result, seed: dict[str, str], *, cases) -> list[Survivor]:
    """A candidate that got a golden case wrong which the seed got right is
    discarded, whatever it scored.

    `cases` is the **validation** set, in the order GEPA was given it, because
    that order is what its per-case scores are keyed by.

    Read out of GEPA's own per-case validation scores rather than re-run.
    `metric_turn` scores a wrong answer zero regardless of cost, so
    `val_subscores[i][case] > 0` *is* "candidate i answered that case". The
    naive version — every pooled candidate re-run over every case — is about
    290,000 tokens on top of a search that cost 700,000, and a gate should not
    cost half of what it protects.

    **Nothing is averaged.** A mean is exactly what lets a candidate buy one
    catastrophic case with a broad small gain, which is the trade this refuses.

    `loop` is unused and stays in the signature so `cli.py` can call either gate
    the same way.
    """
    subscores = result.val_subscores or []
    scores = result.val_aggregate_scores or []

    seed_index = next(
        (i for i, candidate in enumerate(result.candidates) if candidate == seed),
        None,
    )
    if seed_index is None or seed_index >= len(subscores):
        # GEPA drops the seed from the pool only if it was never evaluated, and
        # then there is nothing to have regressed *against*. Saying so is better
        # than an empty gate that looks like it ran.
        say(
            "\nthe seed is not in the pool with a validation row, so there is "
            "nothing to have regressed against — the gate is not checking",
            fg="yellow",
        )
        answered: set = set()
    else:
        answered = {case for case, s in subscores[seed_index].items() if s > 0}
        say(
            f"\nthe seed answers {len(answered)}/{len(subscores[seed_index])} "
            f"golden cases correctly"
        )

    survivors: list[Survivor] = []
    for i, candidate in enumerate(result.candidates):
        if i == seed_index:
            continue
        score = scores[i] if i < len(scores) else 0.0

        if i >= len(subscores):
            say(
                f"  candidate {i} (val {score:.3f}) DISCARDED — no validation "
                f"row, so nothing says it did not regress",
                fg="red",
            )
            continue

        theirs = subscores[i]
        lost = sorted(case for case in answered if theirs.get(case, 0.0) <= 0)
        if lost:
            say(
                f"  candidate {i} (val {score:.3f}) DISCARDED — lost "
                f"{len(lost)} case(s) the seed answered",
                fg="red",
            )
            for case in lost:
                say(f"      {_asked(cases, case)}")
            continue

        say(f"  candidate {i} (val {score:.3f}) survives", fg="green")
        survivors.append(Survivor(index=i, candidate=candidate, score=score))

    return sorted(survivors, key=lambda s: -s.score)
