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
import yaml

from app import prompts
from app.config import Config, config
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

    # The metric's weighted terms, which are the axes of this target's Pareto
    # front and the column order of the table under it. Per target because
    # each metric has its own terms: reading one metric's weights for every
    # target puts `nan` in four of five columns and nobody on the front.
    weights: Callable[[], dict[str, float]]

    # The cheap pre-check `--probe-only` runs: (loop, yes) -> exit status. It
    # takes `yes` because a check can itself be expensive enough to ask about.
    # None where a target has nothing cheap to offer.
    check: Callable[[Any, bool], int] | None = None

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


def _extract_weights() -> dict[str, float]:
    from tools.gepa.metric_extract import WEIGHTS

    return WEIGHTS


def _extract_check(loop, yes: bool = False) -> int:
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
    weights=_extract_weights,
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


def _tools_weights() -> dict[str, float]:
    from tools.gepa.metric_turn import WEIGHTS

    return WEIGHTS


def _tools_check(loop, yes: bool = False) -> int:
    return _turn_check(
        loop, yes, name="tools", seed=_tools_seed(),
        to_overrides=_tools_overrides, focus=_tools_focus,
    )


def _turn_check(
    loop, yes: bool, *, name: str, seed: dict[str, str], to_overrides, focus
) -> int:
    """The seed over the whole golden corpus, once, before any search.

    Two things nothing else will tell you. Whether the corpus is answerable at
    all — if the seed gets most of it wrong, `correct` never reaches 1.0, the
    cost and tool-call terms are gated off on every case, and a search has
    nothing to optimise but a term it cannot move, which looks exactly like GEPA
    finding nothing. And the before half of a before-and-after table, which is
    the only form the talk can show a tool-description or an effort change in.

    Shared by every whole-turn target, because the seed rollouts are the same
    rollouts whichever component is about to be searched: the same nineteen
    questions, under the text and the efforts on disk.

    Nineteen cold turns, so it asks first like a search does.
    """
    from tools.gepa import cli, golden
    from tools.gepa.adapter import TurnAdapter

    cases = golden.load()
    say(f"\n{name:<9} the seed over all {len(cases)} golden cases")
    say(f"  budget      {len(cases)} rollouts at roughly "
        f"{COLD_TURN_TOKENS:,} tokens each — about "
        f"{len(cases) * COLD_TURN_TOKENS:,} tokens")
    cli.ask_before_spending(yes)

    adapter = TurnAdapter(loop, to_overrides=to_overrides, focus=focus)
    batch = adapter.evaluate(cases, seed, capture_traces=True)

    say(f"\n  {'case':<32} {'right':>6} {'tokens':>7} {'tools':>5}")
    right = broke = 0
    for t in batch.trajectories or []:
        ok = t.score.terms["correct"] >= 1.0
        # A rollout that fell over is not a wrong answer, and a table that shows
        # them the same way is how nineteen zeroes read as a hard corpus.
        failed = t.replayed.error is not None
        right += ok
        broke += failed
        say(
            f"  {t.case.name:<32} {'broke' if failed else 'yes' if ok else 'NO':>6} "
            f"{t.replayed.tokens:>7,} {len(t.replayed.tools):>5}",
            fg="red" if failed else None if ok else "yellow",
        )

    if broke:
        say(
            f"\n{broke} of {len(cases)} rollouts did not complete. That is the "
            f"harness, not the corpus — the first one said:\n",
            fg="red",
        )
        first = next(t for t in batch.trajectories if t.replayed.error)
        say(f"  {first.replayed.error}")
        return 1

    say(f"\n  the seed answers {right} of {len(cases)} correctly")
    _spend_by_node(batch.trajectories or [])

    # Both ends are the problem, and the middle is the point. A corpus the seed
    # already answers cannot be won — `demo/questions.txt` says exactly that
    # about itself — and one it never answers has two of three terms gated off
    # on every case, which reads as GEPA finding nothing.
    if right == 0:
        say(
            "\nNone. Every case is gated at `correct`, so cost and tool calls "
            "are never scored and there is nothing for a search to move. Read "
            "the feedback on one case before spending anything.",
            fg="red",
        )
        return 1
    if right == len(cases):
        say(
            "\nAll of them. A candidate cannot win where the seed already "
            "wins, so this corpus can confirm the status quo and nothing else. "
            "It needs harder questions.",
            fg="yellow",
        )
        return 1
    say(
        f"  the {len(cases) - right} it gets wrong are where a candidate has "
        f"room to win",
    )
    return 0


def _spend_by_node(trajectories) -> None:
    """Where the seed's tokens went, summed over the corpus, dearest first.

    The before half for an effort search: it says which node's effort is worth
    lowering, and it says so before a search is paid for. The effort shown is
    the one in force on the first rollout, which is the seed's on a check.
    """
    totals: dict[str, int] = {}
    efforts: dict[str, str] = {}
    for t in trajectories:
        for node, (tin, tout) in t.replayed.per_node.items():
            totals[node] = totals.get(node, 0) + tin + tout
        efforts.update(t.replayed.efforts)
    grand = sum(totals.values())
    if not grand:
        return
    say("\n  where the seed's tokens went, over the whole corpus:")
    width = max(len(f"{n} ({efforts.get(n, '?')})") for n in totals)
    for node, spent in sorted(totals.items(), key=lambda kv: -kv[1]):
        label = f"{node} ({efforts[node]})" if node in efforts else node
        say(f"    {label:<{width}}  {spent / grand:>4.0%}  {spent:>9,}")


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
    weights=_tools_weights,
    check=_tools_check,
    templates=_tools_templates,
    # A whole cold turn a rollout, against `extract`'s single model call.
    # CHALLENGE's own figure: 60 with a 10-case train split is under 700k
    # tokens. The default of 150 would be 1.7 million.
    budget=60,
    rollout_tokens=COLD_TURN_TOKENS,
    blurb="the four `description` strings in app/tools.py",
)


# -------------------------------------------------------------------- config
#
# The per-node `effort` block in `config/config.yaml`, as one text component.
# Six enum values the model never reads. Not prose by any reading, and GEPA
# treats it as text anyway, which is the thesis in one line.
#
# One component holding six values rather than six components holding one:
# the reflection can then move `explore` down and `generate_sql` up in one
# proposal, having read which node spent the tokens and which node wrote the
# wrong SQL. A coordinate-wise loop cannot make that move, and it is the whole
# argument for a reflection over a loop when the space is this small per node.
#
# What the metric cannot see, and the template says so. `plan` makes no call
# on a cold turn (an empty memory skips it), so its effort is carried and not
# measured. `extract`'s output is not scored here (nothing it writes is read
# back) and neither is `answer`'s sentence (the result set is scored, not the
# prose above it): their tokens are charged, their quality is not. What is
# measured is `explore`, `generate_sql` and, on the turns that need it, `fix`
# — and `explore` is where most of a cold turn's tokens go, so that is where
# a search has room.

CONFIG_COMPONENT = "efforts"

# The graph nodes that call the model, in the order a cold turn runs them.
# `Config.NODES` also lists `gepa`, which is the reflection model and not part
# of a turn; `prompts.NODES` has the same six in a different order.
TURN_NODES = tuple(n for n in Config.NODES if n in prompts.NODES)

# Spelled here for the template. A test asserts it matches the `Literal` on
# `Node.effort`, so it cannot drift from what the config would accept.
EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")


def _is_claude(node: str) -> bool:
    return config().model_for(node).is_claude()


def _yaml(efforts: dict[str, str]) -> str:
    """The block as `config/config.yaml` writes it, one node per entry, in
    turn order, ending in one newline. Both the seed and the rendered winner go
    through this, so a promotion is a paste."""
    return yaml.safe_dump(
        {n: {"effort": efforts[n]} for n in TURN_NODES if n in efforts},
        sort_keys=False,
    )


def _config_seed() -> dict[str, str]:
    """What is in force today, resolved: the file's block where it has one and
    the node default where it does not, so the seed says `medium` rather than
    leaving a node out."""
    return {CONFIG_COMPONENT: _yaml({n: config().effort_for(n) for n in TURN_NODES})}


def _config_parse(text: str) -> dict[str, str]:
    """A candidate block into `Overrides.efforts`, or `Rejected` with the reason.

    The gate is the config schema. Each block goes through the same `Node`
    model the file goes through, so an illegal value fails with pydantic's own
    message, and `none` on a Claude model is refused with the file validator's
    reason. Nothing new was written to express "never" here.

    A node the candidate leaves out keeps what is on disk. A key that is not
    `effort` is refused outright: a search over effort that quietly also chose
    a model would be measuring something nobody asked about.
    """
    from pydantic import ValidationError

    from app.config import Node
    from tools.gepa.metric_turn import Rejected

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise Rejected(f"not YAML — {str(e).splitlines()[0]}") from None
    if not isinstance(loaded, dict) or not loaded:
        raise Rejected(
            f"expected a mapping of node name to `effort`, got "
            f"{type(loaded).__name__}"
        )
    unknown = sorted(set(loaded) - set(TURN_NODES))
    if unknown:
        raise Rejected(
            f"no node named {', '.join(unknown)} — the nodes are "
            f"{', '.join(TURN_NODES)}"
        )

    efforts: dict[str, str] = {}
    for node, block in loaded.items():
        if not isinstance(block, dict) or set(block) != {"effort"}:
            raise Rejected(
                f"{node}: one key, `effort`, and nothing else — this search is "
                f"over effort only, and the model is not on the table"
            )
        value = block["effort"]
        if value is None:
            raise Rejected(f"{node}: `effort` has no value")
        try:
            Node(effort=value)
        except ValidationError as e:
            raise Rejected(f"{node}: {e.errors()[0]['msg']}") from None
        if value == "none" and _is_claude(node):
            raise Rejected(
                f"{node}: effort none — Claude has no such level, and disabling "
                f"thinking breaks tool calls; the levels are "
                f"{', '.join(EFFORT_LEVELS[1:])}"
            )
        efforts[node] = str(value)
    return efforts


def _config_overrides(candidate: dict[str, str]):
    """A candidate reaches the graph as per-node efforts and nothing else."""
    from app import overrides

    return overrides.Overrides(efforts=_config_parse(candidate[CONFIG_COMPONENT]))


def _config_focus(component: str, replayed) -> dict[str, str]:
    """The component's own slice of the turn: which node ran at what and spent
    what. On every record, gated or not, because a turn that never wrote SQL
    still spent its tokens somewhere."""
    from tools.gepa import metric_turn

    return {
        "Effort in force per node, and what each spent": metric_turn.by_node(replayed)
    }


def _config_adapter(loop) -> Any:
    from tools.gepa.adapter import TurnAdapter

    return TurnAdapter(loop, to_overrides=_config_overrides, focus=_config_focus)


def _config_render(candidate: dict[str, str]) -> str:
    """The block, normalised through the parser, so stdout pastes into
    `config/config.yaml` whatever whitespace the reflection model used."""
    return _yaml(_config_parse(candidate[CONFIG_COMPONENT]))


def _config_label(component: str) -> str:
    return "config/config.yaml, the per-node effort blocks"


def _config_notes() -> str:
    return (
        "to promote: config/config.yaml — replace each node's `effort` with the\n"
        "value above. The `model:` block does not change, and a candidate could\n"
        "not have changed it.\n\n"
        "What was measured is whole cold turns against demo/golden/ on the\n"
        "model config.yaml names. Effort is a fraction of a model's own\n"
        "thinking, so a profile found on one model is not a profile for\n"
        "another. `plan` made no call on these cold turns, so its value above\n"
        "is the seed's, untested. Two nodes were charged and not judged:\n"
        "`extract`'s output is read by no turn here, and `answer`'s sentence\n"
        "sits above the result set that was scored. Lowering either is a bet\n"
        "this run did not test.\n"
    )


CONFIG_TEMPLATE = """\
You are choosing how hard the model thinks at each node of a text-to-SQL
agent. The candidate is a YAML block: one entry per node, each with exactly one
key, `effort`.

The nodes, in the order a cold turn runs them:
  - plan          reads what the agent already knows and decides whether it is
                  enough to write SQL. These turns run with the memory off, so
                  it has nothing to read and makes no call: its effort is
                  carried here and cannot be measured here. Leave it.
  - explore       the loop that calls the introspection tools. Most of a cold
                  turn's tokens are spent here, and it is where a trap — a
                  soft-delete column, three spellings of one region — is found
                  or missed.
  - generate_sql  writes the one SELECT from what explore found.
  - fix           runs only when that SQL failed, and corrects it from the
                  database's error.
  - extract       writes down what was learned for later turns. Its tokens are
                  charged here; its output is not scored, because these turns
                  run with the memory off.
  - answer        turns the result rows into a sentence. Charged, not scored:
                  the rows are what is compared.

The legal values, cheapest first: {levels}. {none_rule}

What you may not do, because each of these is rejected before a turn runs:
  - Do not add or remove a node, and do not add a key other than `effort`.
    Do not name a model: this search is over effort only.
  - Do not write prose. Reply with YAML in the shape shown.

What the feedback tells you, for every turn: which node ran at what effort and
spent what, whether the answer was right, and where it went wrong when it was
not. An effort is worth lowering where a node spent a lot and the turns stayed
right; it is worth raising where the wrong answers were written. Move more than
one node at a time when the evidence says so — that is the reason this is one
block rather than six.

The current block:
```
<curr_param>
```

Turns run under it, worst first, with feedback:
```
<side_info>
```

Reply with the replacement YAML block inside a ``` block, and nothing else.
"""


def _config_templates() -> dict[str, str]:
    """One reflection prompt, for one component. The legal values are read off
    the model in force: `none` is a level only where the model is not Claude,
    and telling the reflection otherwise buys a rejected candidate per round."""
    claude = any(_is_claude(n) for n in TURN_NODES)
    levels = EFFORT_LEVELS[1:] if claude else EFFORT_LEVELS
    none_rule = (
        "`none` is not one of them on this model: it disables thinking, which "
        "breaks tool calls, and a candidate proposing it is rejected."
        if claude
        else "`none` is legal on this model and is the cheapest setting."
    )
    return {
        CONFIG_COMPONENT: CONFIG_TEMPLATE.format(
            levels=", ".join(levels), none_rule=none_rule
        )
    }


def _config_check(loop, yes: bool = False) -> int:
    return _turn_check(
        loop, yes, name="config", seed=_config_seed(),
        to_overrides=_config_overrides, focus=_config_focus,
    )


CONFIG = Target(
    name="config",
    seed=_config_seed,
    corpus=_tools_corpus,
    adapter=_config_adapter,
    gate=lambda loop, result, seed, cases: gates.turn_gate(
        loop, result, seed, cases=cases
    ),
    render=_config_render,
    label=_config_label,
    notes=_config_notes,
    weights=_tools_weights,
    check=_config_check,
    templates=_config_templates,
    budget=60,
    rollout_tokens=COLD_TURN_TOKENS,
    blurb="the per-node `effort` blocks in config/config.yaml",
)


# -------------------------------------------------------------------- registry

TARGETS: dict[str, Target] = {
    EXTRACT.name: EXTRACT, TOOLS.name: TOOLS, CONFIG.name: CONFIG,
}

# Why the rest are not searchable. Not a to-do list: two of these are arguments
# against ever building the thing, and three are budget rather than machinery.
#
# Four of these five said something different before `TurnAdapter` and
# `demo/golden/` existed, and a sixth — `config` — sat here until the metric's
# feedback could name a node, which is what its reason asked for. "Scoring one call means running the SQL against a
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
}
