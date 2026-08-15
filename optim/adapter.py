"""The GEPA integration point. The only module that imports `gepa`.

GEPA wants three things: run a candidate on a batch and score it, build a
reflective dataset the teacher model can read, and a model to do the reflecting.
Everything else in `optim/` is written so those three are thin.

**No DSPy.** Its Signatures would route model calls through litellm and lose
what `app/llm.py` carries — Anthropic's `output_config` with effort and a
JSON-schema format, adaptive thinking, the server-side-fallback beta,
`cache_control: ephemeral` — and optimising a prompt under a different request
shape than production uses tunes it for a configuration you do not run. Its
field markers would also make harness tokens stop being production tokens, in a
repo whose claim is a token count.

GEPA's reflection model is `app.llm.complete` too, so there is still exactly one
module that talks to a model.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from app import llm
from app.config import config
from optim import metric_extract as metric
from optim.cases import ExtractCase
from optim.replay import Replayed, replay

# The component name GEPA mutates. One component, because this optimises one
# node; `candidate` is `{"extract": "<the prompt>"}` throughout.
COMPONENT = "extract"

REFLECT_SYSTEM = (
    "You are improving the instructions given to a component of a larger "
    "system. Reply with the improved instruction text and nothing else."
)


class Loop:
    """One event loop in a daemon thread, for the whole optimisation.

    GEPA's engine is synchronous, `llm.complete` is not, and the module-level
    Anthropic and httpx clients bind to the loop that first used them. Not
    `asyncio.run()` per metric call, which would tear down a loop around a
    client created on the first one and fail intermittently.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="optim-loop", daemon=True
        )
        self._thread.start()

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self) -> Loop:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass
class Trajectory:
    """What one case did, kept for the reflection step."""

    replayed: Replayed
    score: metric.Score


class ExtractAdapter(GEPAAdapter):
    """Score a candidate `extract` prompt over a batch of recorded cases."""

    def __init__(
        self, loop: Loop, *, effort: str | None = None, concurrency: int = 4
    ) -> None:
        self._loop = loop
        self._effort = effort
        self._semaphore = asyncio.Semaphore(concurrency)
        self.calls = 0

    def evaluate(
        self,
        batch: list[ExtractCase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, dict[str, Any]]:
        prompt = candidate[COMPONENT]
        replays: list[Replayed] = self._loop.run(self._batch(prompt, batch))
        self.calls += len(replays)
        scores = [metric.score(r) for r in replays]

        return EvaluationBatch(
            outputs=[
                {
                    "case": r.case.name,
                    "entries": [
                        {"kind": e.kind, "name": e.name, "claim": e.claim,
                         "sql_fragment": e.sql_fragment, "verified": e.verified}
                        for e in r.entries
                    ],
                    "tokens_out": r.tokens_out,
                }
                for r in replays
            ],
            scores=[s.value for s in scores],
            trajectories=(
                [Trajectory(r, s) for r, s in zip(replays, scores)]
                if capture_traces
                else None
            ),
            # Grounding and cost genuinely trade off, so the per-term breakdown
            # is worth carrying: it makes `frontier_type="objective"` available,
            # and a Pareto front is more honest than one scalar pretending the
            # trade-off was already settled.
            objective_scores=[s.terms for s in scores],
        )

    async def _batch(self, prompt: str, batch: list[ExtractCase]) -> list[Replayed]:
        async def one(case: ExtractCase) -> Replayed:
            async with self._semaphore:
                return await replay(prompt, case, effort=self._effort)

        return list(await asyncio.gather(*(one(c) for c in batch)))

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, Sequence[dict[str, Any]]]:
        """The prose the teacher model reads, worst cases first.

        GEPA's contribution over a scalar reward is that the reflection step
        reads *diagnostics*, so the Feedback field names the entry, quotes the
        offending text and says which invariant it broke — "0.62" gives a
        mutation nothing to aim at.
        """
        if COMPONENT not in components_to_update:
            return {}

        records = []
        for t in sorted(eval_batch.trajectories or [], key=lambda t: t.score.value):
            r = t.replayed
            records.append(
                {
                    "Inputs": {
                        "the message the node was sent": r.case.user_message,
                        "the SQL that had already run": r.case.sql,
                    },
                    "Generated Outputs": (
                        "\n".join(
                            f"[{e.kind}] {e.name}: {e.claim}"
                            + (f"\n    sql_fragment: {e.sql_fragment}" if e.sql_fragment else "")
                            for e in r.entries
                        )
                        or "(nothing was recorded)"
                    ),
                    "Feedback": t.score.text(),
                    "score": round(t.score.value, 3),
                    "score_breakdown": {k: round(v, 3) for k, v in t.score.terms.items()},
                }
            )
        return {COMPONENT: records}


def reflection_lm(loop: Loop) -> Any:
    """GEPA's teacher, over the same seam every other model call uses.

    Max effort unless `config/config.yaml`'s `gepa:` block says otherwise: this
    runs a few dozen times and decides what the next candidate says. Its own
    block because it is the one call that is not part of a turn.
    """

    def call(prompt: str | list[dict[str, Any]]) -> str:
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        result = loop.run(
            llm.complete(
                system=REFLECT_SYSTEM,
                messages=messages,
                effort=config().effort_for("gepa.reflect", "max"),
                node="gepa.reflect",
            )
        )
        return result.text

    return call
