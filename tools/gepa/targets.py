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

# What one cold turn costs, measured — the README's T1, and the same constant
# `cli/sql_agent/corpus.py` quotes before a corpus run. Spelled again rather
# than imported: `tools/` does not depend on the client package.
COLD_TURN_TOKENS = 11_500


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

    # (loop, result, seed, cases) -> survivors, best first. Outside the
    # objective; see `gates.py` for why. Takes the cases whether or not it uses
    # them, so the command can call either gate the same way.
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

    # component -> the reflection prompt GEPA renders for it. None uses GEPA's
    # own generic one, which is right for a node instruction and wrong for a
    # tool description.
    templates: Callable[[], dict[str, str]] | None = None

    # Metric calls, when `--budget` does not say. Per target because a rollout
    # is one model call for one target and a whole turn for another.
    budget: int = 150

    # Roughly what one rollout costs, for the sentence before the spend. Zero
    # means cheap enough not to ask.
    rollout_tokens: int = 0

    # One line naming what a run of this actually touches.
    blurb: str = ""


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
    # `cases` is accepted and ignored: the probe gate replays authored cases of
    # its own, which is what it means for an invariant to have no ground truth.
    gate=lambda loop, result, seed, cases: gates.probe_gate(
        loop, result, seed, node=EXTRACT_COMPONENT
    ),
    render=_extract_render,
    label=_extract_label,
    notes=_extract_notes,
    check=_extract_check,
    blurb="config/prompts/extract.md",
)


# --------------------------------------------------------------------- tools
#
# The four `description` strings in `app/tools.py`, as four components of one
# candidate. Prose somebody wrote once, sent on every cold turn, never measured.
#
# They stay in `app/tools.py` rather than moving to `config/tools/*.md`. The
# symmetry with `config/prompts/` is real and so is the promotion ergonomics,
# but it costs a loader, a directory contract and a new production failure mode
# for four strings. The trigger is written down beside `SCHEMAS`: if these get
# promoted from a search more than twice, move them.


def _tools_seed() -> dict[str, str]:
    from app import tools

    return {s["name"]: s["description"] for s in tools.SCHEMAS}


def _tools_corpus(*, days: int, resume: bool, verbose: bool) -> list[Any]:
    """Tracked on disk, not harvested. `days` and `resume` mean nothing to a
    directory in git, and are ignored rather than the command knowing which
    targets read them."""
    from tools.gepa import golden

    cases = golden.load()
    say(f"corpus    {len(cases)} golden cases from {golden.GOLDEN}")
    if verbose:
        with_expect = sum(1 for c in cases if c.expect is not None)
        say(f"          {with_expect} carry a literal answer; the rest are "
            f"checked against their reference query alone")
    return cases


def _tools_overrides(candidate: dict[str, str]):
    """A candidate reaches the graph as tool descriptions and nothing else.

    `prompts` and `efforts` stay empty deliberately: a search over descriptions
    that quietly also changed a prompt would be measuring something nobody
    asked about.
    """
    from app import overrides

    return overrides.Overrides(tools=dict(candidate))


def _tools_focus(component: str, replayed) -> dict[str, str]:
    """One component's own slice of the turn, for the reflection step.

    "It called `sample_column` three times on one column" is a sentence about
    `sample_column`'s description. The undifferentiated turn summary is not, and
    a reflection that cannot see the difference proposes generic prose.
    """
    calls = [t for t in replayed.tools if t.name == component]
    if not calls:
        return {
            f"How `{component}` was used": (
                "It was never called on this turn. That is as much a fact about "
                "its description as a redundant call would be."
            )
        }
    rendered = "; ".join(
        f"{t.name}({', '.join(f'{k}={v!r}' for k, v in t.args.items())})"
        + (" [errored]" if t.error else "")
        for t in calls
    )
    return {
        f"How `{component}` was used": f"called {len(calls)} time(s): {rendered}"
    }


def _tools_adapter(loop) -> Any:
    from tools.gepa.adapter import TurnAdapter

    return TurnAdapter(loop, to_overrides=_tools_overrides, focus=_tools_focus)


def _tools_render(candidate: dict[str, str]) -> str:
    """One Markdown document, a section per tool, every tool present.

    All four, including the ones GEPA never mutated: the document is the
    complete set, and a promotion that updates three and leaves one stale is
    exactly the failure a per-component format would invite.

    A description beginning with `#` would forge a section boundary, so leading
    hashes are stripped.
    """
    seed = _tools_seed()
    sections = []
    for name in seed:
        text = candidate.get(name, seed[name]).strip()
        cleaned = "\n".join(line.lstrip("#").lstrip() if line.lstrip().startswith("#")
                            else line for line in text.splitlines())
        if cleaned != text:
            say(f"          stripped a leading # from {name}'s description",
                fg="yellow")
        sections.append(f"## {name}\n\n{cleaned}")
    return "\n\n".join(sections) + "\n"


def _tools_label(component: str) -> str:
    return f"app/tools.py SCHEMAS[{component}].description"


def _tools_notes() -> str:
    return (
        "to promote: app/tools.py, the SCHEMAS list — replace each tool's\n"
        "`description` with the matching section above. `name` and\n"
        "`input_schema` do not change, and a candidate could not have changed\n"
        "them.\n\n"
        "A description that reads well is not the same as one that made the\n"
        "explore loop shorter. What was measured is whole turns against\n"
        "demo/golden/, and nothing here saw a database other than the demo.\n"
    )


TOOL_TEMPLATE = """\
You are rewriting the `description` of one tool in an API tool definition.
The tool is `{name}`.
{args}

The assistant sees all four tool descriptions on every cold turn, before it has
seen anything about the database it is pointed at. The description is the only
thing telling it when to reach for this tool and when to reach for another one.

What you may not do, because each of these breaks the call rather than merely
scoring badly:
  - Do not rename the tool. Dispatch keys on the name.
  - Do not add, remove or rename an argument. Refer only to the ones listed.
  - Do not describe one particular database. This description ships with the
    code; the database it points at is whatever somebody configured.

What a good one does:
  - Says when to call it and, just as usefully, when not to. The feedback below
    is mostly about a tool called three times where once would do, or one that
    should have been called and was not.
  - Two or three sentences. It is sent on every cold turn, so every word is a
    bill that recurs.

The current description:
```
<curr_param>
```

Turns run under it, worst first, with feedback:
```
<side_info>
```

Reply with the replacement description inside a ``` block, and nothing else.
"""


def _tools_templates() -> dict[str, str]:
    """A reflection prompt per tool, through GEPA's own per-component mechanism.

    `gepa.optimize(reflection_prompt_template=...)` takes a dict keyed by
    component name and validates it when the run is constructed, so a template
    missing its placeholders fails before a token is spent. The generic prompt
    would rewrite a tool description as though it were a node instruction.

    The argument names come out of each tool's own `input_schema`, so they
    cannot drift from what the tool actually takes.
    """
    from app import tools

    return {
        s["name"]: TOOL_TEMPLATE.format(name=s["name"], args=_argument_line(s))
        for s in tools.SCHEMAS
    }


def _argument_line(schema: dict) -> str:
    """What the tool takes, as a sentence.

    `list_tables` takes nothing, and "its arguments are: ." is broken prose in a
    prompt a model is about to read carefully.
    """
    names = list(schema["input_schema"].get("properties") or {})
    if not names:
        return "It takes no arguments."
    return "It takes these arguments, and they are fixed: " + ", ".join(
        f"`{n}`" for n in names
    ) + "."


def _tools_check(loop) -> int:
    """The seed over the whole golden corpus, once, before any search.

    Two things nothing else will tell you. Whether the corpus is answerable at
    all — if the seed gets most of it wrong, `correct` never reaches 1.0, the
    cost and tool-call terms are gated off on every case, and a search has
    nothing to optimise but a term it cannot move, which looks exactly like GEPA
    finding nothing. And the before half of a before-and-after table, which is
    the only form the talk can show a tool-description change in.

    Nineteen cold turns, so it asks first like a search does.
    """
    from tools.gepa import cli, golden, metric_turn, reference
    from tools.gepa.adapter import TurnAdapter

    cases = golden.load()
    seed = _tools_seed()
    say(f"\ntools     the seed over all {len(cases)} golden cases")
    say(f"  budget      {len(cases)} rollouts at roughly "
        f"{COLD_TURN_TOKENS:,} tokens each — about "
        f"{len(cases) * COLD_TURN_TOKENS:,} tokens")
    cli.ask_before_spending()

    adapter = TurnAdapter(loop, to_overrides=_tools_overrides, focus=_tools_focus)
    batch = adapter.evaluate(cases, seed, capture_traces=True)

    say(f"\n  {'case':<32} {'right':>5} {'tokens':>7} {'tools':>5}")
    right = 0
    for t in batch.trajectories or []:
        ok = t.score.terms["correct"] >= 1.0
        right += ok
        say(
            f"  {t.case.name:<32} {'yes' if ok else 'NO':>5} "
            f"{t.replayed.tokens:>7,} {len(t.replayed.tools):>5}",
            fg=None if ok else "yellow",
        )

    say(f"\n  the seed answers {right} of {len(cases)} correctly")
    if right * 2 < len(cases):
        say(
            "\nUnder half. Two of three terms are gated off on a wrong answer, "
            "so a search here would have nothing to optimise but correctness — "
            "which reads as GEPA finding nothing. The fix is easier cases, not "
            "more rollouts.",
            fg="yellow",
        )
        return 1
    return 0


TOOLS = Target(
    name="tools",
    seed=_tools_seed,
    corpus=_tools_corpus,
    adapter=_tools_adapter,
    gate=lambda loop, result, seed, cases: gates.turn_gate(
        loop, result, seed, cases=cases
    ),
    render=_tools_render,
    label=_tools_label,
    notes=_tools_notes,
    check=_tools_check,
    templates=_tools_templates,
    # A whole cold turn a rollout, against `extract`'s single model call.
    # CHALLENGE's own figure: 60 with a 10-case train split is under 700k
    # tokens. The default of 150 would be 1.7 million.
    budget=60,
    rollout_tokens=COLD_TURN_TOKENS,
    blurb="the four `description` strings in app/tools.py",
)


# -------------------------------------------------------------------- registry

TARGETS: dict[str, Target] = {EXTRACT.name: EXTRACT, TOOLS.name: TOOLS}

# Why the rest are not searchable. Not a to-do list: two of these are arguments
# against ever building the thing, and three are budget rather than machinery.
#
# Four of these five said something different before `TurnAdapter` and
# `demo/golden/` existed. "Scoring one call means running the SQL against a
# warehouse whose answers are known" was a reason not to; it is now a
# description of the corpus. A reason that has quietly become false is worse
# than no reason, because it still reads like one.
UNWIRED = {
    "plan": (
        "`plan`'s degenerate optimum is the worst in the graph: always say\n"
        "sufficient, skip the exploration, and every cheap term in the turn\n"
        "metric improves at once. Nothing in demo/golden/ gates that — a\n"
        "question the memory can already answer is right either way — so the\n"
        "metric would pay for the collapse."
    ),
    "explore": (
        "Wirable today: it is a prompt, and `TurnAdapter` scores a whole turn.\n"
        "Unwired on budget. Round-robin gives each component one proposal in\n"
        "every N iterations, and four tool descriptions already spend a 60\n"
        "rollout run. Add it as a fifth component when there is budget for one."
    ),
    "answer": (
        "`answer`'s only cheap proxy is length, and that trade was measured on\n"
        "the quarters question — 353 output tokens before, 278 after — and\n"
        "decided the other way. The turn metric does not help: it scores the\n"
        "result set, and `answer` writes the sentence above it."
    ),
    "generate_sql": (
        "Wirable today — demo/golden/ is exactly the warehouse-with-known-\n"
        "answers this used to need. Unwired on budget, as `explore` is."
    ),
    "fix": (
        "A corpus problem that the golden set does not solve. Under the turn\n"
        "metric `fix` runs only on rollouts where the candidate had already\n"
        "written broken SQL, so most rollouts give its component no signal at\n"
        "all, and the ones that do are the ones where something else went wrong."
    ),
    "config": (
        "The per-node `effort` block, as a text component. `overrides.Overrides`\n"
        "already carries efforts, so the injection exists. What is missing is a\n"
        "reflection template that will not rewrite YAML as prose, and an\n"
        "argument: the search space is six values per node, so GEPA earns its\n"
        "place here through the feedback rather than the search. If the\n"
        "reflection is not visibly reading the tool-call trace, this is not\n"
        "worth wiring."
    ),
}
