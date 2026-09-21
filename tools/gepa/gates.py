"""What disqualifies a candidate, whatever it scored.

Outside GEPA's objective, and that is the whole point. Weights cannot express
"never": a mean-maximising search trades a rare catastrophic failure for a broad
small gain the moment the arithmetic allows it, and the failures worth refusing
here are exactly the rare catastrophic ones.

`probe_gate` checks invariants that exist only as prose in a prompt, by
replaying authored cases: a recorded count going stale, a recipe grounded in a
fragment too short to be wrong. Those have no ground truth, so an authored
predicate is the only witness there is.

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

        # Not rejection — a shorter prompt can be right. But one that won by
        # deleting most of the instruction has to be seen.
        if len(text) < 0.6 * len(seed_text):
            say(
                f"  candidate {i} (val {score:.3f}) is {len(text)} chars against "
                f"the seed's {len(seed_text)} — read the diff closely",
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
