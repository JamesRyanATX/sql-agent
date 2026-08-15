"""The valset: authored, adversarial, hard-labelled.

**The trainset is harvested — real, messy, weakly labelled. The valset is
authored.** Conflating them is how you get a prompt that scores eight percent
better and deletes the census paragraph.

Every probe defends an invariant that exists *only as prose*, and that is the
whole scoping rule: an invariant enforced by code belongs in a test instead. A
line inside `config/prompts/extract.md` is something a candidate can delete for
a token saving, and its failure surfaces turns later.

A predicate is a predicate, not a rubric: 0 or 1, no partial credit. The probe
files are tracked in `tests/probes/` because they are contract; the harvested
corpus is gitignored, because it holds whatever the warehouse holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from optim import detect
from optim.cases import ExtractCase
from optim.replay import Replayed

ROOT = Path(__file__).resolve().parent.parent / "tests" / "probes"


@dataclass(frozen=True)
class Probe:
    name: str
    invariant: str  # the sentence in the prompt this defends
    cites: str  # where the reason is written down
    why: str  # the failure that motivated it
    predicate: str
    args: dict[str, Any]
    case: ExtractCase


def load(node: str = "extract") -> list[Probe]:
    """Every probe for a node, in filename order."""
    directory = ROOT / node
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no probes in {directory}")
    return [_probe(json.loads(p.read_text(encoding="utf-8"))) for p in files]


def _probe(raw: dict[str, Any]) -> Probe:
    case = raw["case"]
    return Probe(
        name=raw["name"],
        invariant=raw["invariant"],
        cites=raw["cites"],
        why=raw["why"],
        predicate=raw["predicate"],
        args=raw.get("args", {}),
        case=ExtractCase.authored(
            name=raw["name"],
            question=case["question"],
            sql=case["sql"],
            findings=case["findings"],
            filed=case.get("filed", {}),
        ),
    )


def check(probe: Probe, replayed: Replayed) -> tuple[bool, str]:
    """Did the candidate honour the invariant? A reason either way."""
    if replayed.error:
        return False, f"the call failed: {replayed.error}"
    return PREDICATES[probe.predicate](replayed, probe.args)


# ------------------------------------------------------------------ predicates


def _no_census(r: Replayed, args: dict[str, Any]) -> tuple[bool, str]:
    offenders = [
        (e.name, e.claim, hits)
        for e in r.entries
        if (hits := detect.census_hits(e.claim))
    ]
    if not offenders:
        return True, f"{len(r.entries)} entries, none of them a count"
    return False, "\n".join(
        f'  name="{n}" reads as a census {h}: {c}' for n, c, h in offenders
    )


def _recipes_are_grounded(r: Replayed, args: dict[str, Any]) -> tuple[bool, str]:
    if not r.recipes:
        # Vacuous truth is the first thing an optimiser finds: a query this
        # substantial establishes at least one recipe.
        return False, "no recipe at all — the query establishes at least one"

    problems: list[str] = []
    for e in r.recipes:
        tokens = graph_tokens(e.sql_fragment)
        if not e.verified:
            problems.append(
                f'  name="{e.name}" fragment={e.sql_fragment!r}\n'
                f"    is not a subsequence of the SQL that ran — it claims "
                f"something the query never did"
            )
        elif not detect.informative(e.sql_fragment, tokens):
            problems.append(
                f'  name="{e.name}" fragment={e.sql_fragment!r}\n'
                f"    is grounded only because it is too short to say anything "
                f"({detect.MIN_FRAGMENT_TOKENS} distinct tokens required)"
            )
    if problems:
        return False, "\n".join(problems)
    return True, f"{len(r.recipes)} recipes, all copied from the query that ran"


def _no_scope_creep(r: Replayed, args: dict[str, Any]) -> tuple[bool, str]:
    """A general rule must not be overwritten by a special case.

    Not "the protected name is absent" — reusing a name to *refine* it is what
    the upsert is for. What is forbidden is the name acquiring a scope it did
    not have, because every later question that composed the general rule
    silently inherits the narrower one.
    """
    protected = args["name"]
    forbidden = [w.casefold() for w in args["must_not_mention"]]
    for e in r.entries:
        if e.name != protected:
            continue
        found = [w for w in forbidden if w in (e.claim or "").casefold()]
        if found:
            return False, (
                f'  "{protected}" was overwritten with a narrower claim\n'
                f"    was: {r.case.filed.get(protected, '(unfiled)')}\n"
                f"    now: {e.claim}\n"
                f"    it picked up {found}, and every later question that "
                f"composed the general rule inherits the special case"
            )
    return True, f'"{protected}" kept its general meaning'


def _no_near_collision(r: Replayed, args: dict[str, Any]) -> tuple[bool, str]:
    collisions = detect.near_collisions(
        r.names, list(r.case.filed), max_distance=args.get("max_distance", 2)
    )
    if not collisions:
        return True, f"{len(r.names)} names, none a variant of a filed one"
    return False, "\n".join(
        f'  "{new}" is a paraphrase of the filed "{old}" — the upsert keys on '
        f"the name, so these become two entries saying one thing"
        for new, old in collisions
    )


PREDICATES: dict[str, Callable[[Replayed, dict[str, Any]], tuple[bool, str]]] = {
    "no_census": _no_census,
    "recipes_are_grounded": _recipes_are_grounded,
    "no_scope_creep": _no_scope_creep,
    "no_near_collision": _no_near_collision,
}


def graph_tokens(fragment: str | None) -> list[str]:
    """The gate's own tokenisation, so `informative` counts what it counts."""
    from app.graph import _tokens

    return _tokens(fragment or "")
