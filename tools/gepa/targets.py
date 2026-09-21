"""What `make gepa-<name>` can search, and what each one needs to be searched.

`cli.py` used to assume a candidate was one text component and that the
component was `config/prompts/<node>.md`. That assumption sat in eleven places:
how the seed was built, how the corpus was found, how the pool was gated, how
the winner reached stdout, how a diff was labelled. Every one of them has to
change the moment a candidate has four keys, which is what searching the tool
descriptions means.

So each of those eleven expressions became a field here, and `cli.py` became
target-generic. The alternative was the same conditional in six functions, which
is the shape that rots: the seventh place that needs it gets forgotten, and
nothing fails until a run produces a document with one section missing.

`TARGETS` and `UNWIRED` sit together because they are one statement — these are
searchable, these are not, and here is why not. There is deliberately no second
list of what is wired: two registries of one fact disagree eventually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import click

from app import prompts
from tools.gepa import gates, probes
from tools.gepa.gates import Survivor, say

# `extract` is one node with one prompt, so its candidate has one key. Named
# rather than inlined because three things have to agree on the string, and it
# is no longer called COMPONENT: that name was a lie the moment a candidate
# could hold four.
EXTRACT_COMPONENT = "extract"


@dataclass(frozen=True)
class Target:
    """One searchable thing.

    Every field replaces an expression that used to be inline in `cli.py`.
    Nothing here is speculative.
    """

    name: str

    # The candidate GEPA starts from. A dict even where there is one key, so
    # nothing downstream may assume otherwise.
    seed: Callable[[], dict[str, str]]

    # (days, resume, verbose) -> the cases to score against. Narrates its own
    # progress, because what is worth saying differs: one target harvests and
    # reports what it dropped, another reads a tracked directory.
    corpus: Callable[..., list[Any]]

    # Loop -> a GEPA adapter. Typed loosely so this module does not import gepa.
    adapter: Callable[[Any], Any]

    # (loop, result, seed) -> survivors, best first. Outside the objective; see
    # `gates.py` for why.
    gate: Callable[..., list[Survivor]]

    # The winner as the document stdout gets. Ends in exactly one newline.
    render: Callable[[dict[str, str]], str]

    # component -> where a person would go to change it. Diff headers.
    label: Callable[[str], str]

    # What to say after the diff: what this text is the only home for, and what
    # a passing gate still does not prove. Empty is allowed.
    notes: Callable[[], str]

    # The cheap pre-check `--probe-only` runs, returning an exit status. None
    # where a target has nothing cheap to offer.
    check: Callable[[Any], int] | None = None


# ------------------------------------------------------------------- extract


def _extract_seed() -> dict[str, str]:
    return {EXTRACT_COMPONENT: prompts.get(EXTRACT_COMPONENT)}


def _extract_corpus(*, days: int, resume: bool, verbose: bool) -> list[Any]:
    """Harvested every run, so a search always scores the prompt against what
    the agent has actually been doing. Only `--resume` reuses what is on disk:
    GEPA's state references its trainset, so resuming onto a re-harvested corpus
    would be a run made of two different populations."""
    from tools.gepa import cli
    from tools.gepa.cases import read_jsonl, write_jsonl

    path = cli.corpus(EXTRACT_COMPONENT)
    if resume and path.exists():
        cases = read_jsonl(path)
        say(f"corpus    {len(cases)} cases reused from {path}")
        return cases

    from app import tracing
    from tools.gepa.harvest import extract_cases

    if not tracing.enabled():
        raise click.ClickException(
            "tracing is off, so there is nothing to harvest — set both "
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (make langfuse-up), "
            "run some turns, then try again."
        )

    harvested = extract_cases(days=days)
    if verbose:
        say(harvested.report())
    else:
        say(
            f"harvest   {len(harvested.cases)} cases from {harvested.seen} "
            f"recorded extract calls (-v for what was dropped)"
        )
    if not harvested.cases:
        raise click.ClickException(
            "no cases — has the agent answered any questions since tracing was "
            "switched on?"
        )

    write_jsonl(path, harvested.cases)
    return harvested.cases


def _extract_adapter(loop) -> Any:
    from tools.gepa.adapter import ExtractAdapter

    return ExtractAdapter(loop)


def _extract_render(candidate: dict[str, str]) -> str:
    return candidate[EXTRACT_COMPONENT].strip() + "\n"


def _extract_label(component: str) -> str:
    return str(prompts.directory() / f"{component}.md")


def _extract_notes() -> str:
    lines = ["the invariants this prompt is the only home for:"]
    for p in probes.load(EXTRACT_COMPONENT):
        lines.append(f"  - {p.invariant}")
        lines.append(f"    {p.cites}")
    lines.append(
        "\nA dropped sentence passing the probes does not mean the sentence was "
        "doing nothing: cache rot surfaces turns later, outside anything "
        "measured here.\n"
    )
    return "\n".join(lines)


def _extract_check(loop) -> int:
    outcomes = gates.run_probes(
        loop, _extract_seed()[EXTRACT_COMPONENT], probes.load(EXTRACT_COMPONENT)
    )
    return 1 if gates.report(outcomes, _extract_label(EXTRACT_COMPONENT)) else 0


EXTRACT = Target(
    name=EXTRACT_COMPONENT,
    seed=_extract_seed,
    corpus=_extract_corpus,
    adapter=_extract_adapter,
    gate=lambda loop, result, seed: gates.probe_gate(
        loop, result, seed, node=EXTRACT_COMPONENT
    ),
    render=_extract_render,
    label=_extract_label,
    notes=_extract_notes,
    check=_extract_check,
)


# -------------------------------------------------------------------- registry

TARGETS: dict[str, Target] = {EXTRACT.name: EXTRACT}

# Why the rest are not searchable. Two of these are arguments against building
# the thing rather than a to-do list.
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
