"""Do the current prompts still honour the invariants written into them?

Marked `live` because it is the only honest way to ask: the invariants are
instructions to a model, so the model has to answer. Four calls at
`effort_extract=low`, no database of any kind — `extract` is a function of three
strings, which is what makes this cheap enough to keep.

This file deliberately does not know a prompt optimiser exists. The probes are
its valset, but they earn their place before that: hand-editing a prompt is what
actually happens in this repo, and until now nothing checked that an edit kept
the sentences that were put there because of an observed failure. If GEPA is
never wired up, this still runs on every `make test-live`.
"""

from __future__ import annotations

import pytest

from app import prompts
from optim import probes, replay

PROBES = probes.load("extract")


def test_there_are_probes_to_run():
    assert PROBES, "no probe files — did tests/probes/extract move?"


@pytest.mark.live
@pytest.mark.parametrize("probe", PROBES, ids=lambda p: p.name)
async def test_the_extract_prompt_honours_its_invariants(probe):
    replayed = await replay.replay(prompts.get("extract"), probe.case)
    ok, reason = probes.check(probe, replayed)

    assert ok, "\n".join(
        [
            "",
            f"probe {probe.name!r} failed — the prompt stopped honouring:",
            f'  "{probe.invariant}"',
            "",
            reason,
            "",
            f"why this is checked: {probe.why}",
            f"written down in: {probe.cites}",
            "",
            "what the model returned:",
            *(
                f"  [{e.kind}] {e.name}: {e.claim}"
                + (f"\n      SQL: {e.sql_fragment}" if e.sql_fragment else "")
                for e in replayed.entries
            ),
        ]
    )
