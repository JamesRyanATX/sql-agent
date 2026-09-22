"""The GEPA integration point. The only module that imports `gepa`.

GEPA wants three things: run a candidate on a batch and score it, build a
reflective dataset the teacher model can read, and a model to do the reflecting.
Everything else in `tools/gepa/` is written so those three are thin.

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
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from app import llm, overrides
from app.config import config
from tools.gepa import metric_extract as metric
from tools.gepa import metric_turn, reference
from tools.gepa.cases import ExtractCase
from tools.gepa.replay import Replayed, replay
from tools.gepa.replay_turn import TurnReplayed, replay_turn
from tools.gepa.score import Score

# The key this adapter reads out of a candidate. One component, because
# `extract` is one node with one prompt; `candidate` is `{"extract": "..."}`
# throughout. Imported rather than spelled again, so the string lives once —
# `targets.py` builds the seed dict that has to agree with it.
from tools.gepa.targets import EXTRACT_COMPONENT as COMPONENT  # noqa: E402

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
            target=self._loop.run_forever, name="gepa-loop", daemon=True
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
        self,
        loop: Loop,
        *,
        effort: str | None = None,
        concurrency: int = 4,
        reflections: Path | None = None,
    ) -> None:
        self._loop = loop
        self._effort = effort
        self._semaphore = asyncio.Semaphore(concurrency)
        self.calls = 0
        # Where to keep what the reflection model was handed. See `keep`.
        self.reflections = reflections

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
        keep(self.reflections, candidate, {COMPONENT: records})
        return {COMPONENT: records}


@dataclass
class TurnTrajectory:
    """What one golden case did under one candidate, kept for reflection."""

    case: Any
    replayed: TurnReplayed
    score: Score


class TurnAdapter(GEPAAdapter):
    """Score a candidate by running whole cold turns under it.

    Knows nothing about what its components *are*. `to_overrides` is how a
    candidate reaches the running graph and `focus` is how one component's own
    slice of a turn reaches the reflection step; both come from the target. That
    is what lets tool descriptions, node prompts and the config block be this
    adapter rather than three.

    **Concurrency is three, not four.** `explore` holds one target-database
    connection for the whole of a turn, and `TARGET_POOL_MAX` is five. A fourth
    concurrent rollout plus the batch's reference queries sits at the pool's
    limit, and SQLAlchemy's pool *blocks* rather than erroring — so the symptom
    is a rollout that times out and scores zero, which the metric then reads as
    a bad candidate. A confound that looks like a measurement is the worst
    failure available here. Raising this means raising the pool in the same
    breath.
    """

    def __init__(
        self,
        loop: Loop,
        *,
        to_overrides: Callable[[dict[str, str]], overrides.Overrides],
        focus: Callable[[str, TurnReplayed], dict[str, Any]] | None = None,
        concurrency: int = 3,
        reflections: Path | None = None,
    ) -> None:
        self._loop = loop
        self._to_overrides = to_overrides
        self._focus = focus or (lambda component, replayed: {})
        self._semaphore = asyncio.Semaphore(concurrency)
        self.reflections = reflections
        # What the seed spent per case, filled by the first evaluation and then
        # left alone. The cost and tool-call terms are one-sided against it.
        self._baseline: dict[str, metric_turn.Baseline] = {}
        self.calls = 0

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TurnTrajectory, dict[str, Any]]:
        try:
            applied = self._to_overrides(candidate)
        except metric_turn.Rejected as e:
            return self._rejected(batch, str(e), capture_traces)

        rolled: list[TurnReplayed] = self._loop.run(self._batch(applied, batch))
        self.calls += len(rolled)

        # Once per batch, not once per rollout. One transaction is one `now()`,
        # which is the fairness the date-windowed questions need, and it is
        # nineteen executions instead of nineteen per rollout against a pool the
        # rollouts are already using. See `reference.py`.
        truth = self._loop.run(reference.resolve(batch))

        scores = [
            metric_turn.score(r, case, truth[case.name], baseline=self._baseline.get(case.name))
            for r, case in zip(rolled, batch)
        ]
        self._remember(batch, rolled)
        return self._result(batch, rolled, scores, capture_traces)

    def _rejected(
        self, batch: list[Any], reason: str, capture_traces: bool
    ) -> EvaluationBatch[TurnTrajectory, dict[str, Any]]:
        """Every case zero, no rollout run, the reason on every record.

        A candidate the graph cannot run is a fact about the candidate, not a
        broken harness: the run continues, nothing is spent, and the reflection
        reads why. `calls` is not advanced because nothing was called.
        """
        rolled = [
            TurnReplayed(question=case.question, error=f"rejected: {reason}")
            for case in batch
        ]
        scores = [metric_turn.rejected(reason) for _ in batch]
        return self._result(batch, rolled, scores, capture_traces)

    @staticmethod
    def _result(
        batch: list[Any],
        rolled: list[TurnReplayed],
        scores: list[Score],
        capture_traces: bool,
    ) -> EvaluationBatch[TurnTrajectory, dict[str, Any]]:
        return EvaluationBatch(
            outputs=[
                {
                    "case": case.name,
                    "answer": r.answer,
                    "sql": r.sql,
                    "rows": len(r.rows),
                    "tools": r.tool_names,
                    "fix_attempts": r.fix_attempts,
                    "tokens": r.tokens,
                    "per_node": {n: list(v) for n, v in r.per_node.items()},
                    "error": r.error,
                }
                for r, case in zip(rolled, batch)
            ],
            scores=[s.value for s in scores],
            trajectories=(
                [TurnTrajectory(c, r, s) for c, r, s in zip(batch, rolled, scores)]
                if capture_traces
                else None
            ),
            objective_scores=[s.terms for s in scores],
        )

    def _remember(self, batch: list[Any], rolled: list[TurnReplayed]) -> None:
        """The first turn to answer a case sets that case's baseline.

        Not stored on the case: what a turn costs is a fact about a model
        version, and `demo/golden/` is committed. First-seen rather than
        best-seen, so the target a candidate is measured against does not move
        underneath the run.
        """
        for case, r in zip(batch, rolled):
            if case.name not in self._baseline and not r.error and r.rows:
                self._baseline[case.name] = metric_turn.Baseline(
                    tokens=r.tokens, tool_calls=len(r.tools)
                )

    async def _batch(
        self, applied: overrides.Overrides, batch: list[Any]
    ) -> list[TurnReplayed]:
        async def one(case: Any) -> TurnReplayed:
            async with self._semaphore:
                return await replay_turn(case.question, candidate=applied)

        return list(await asyncio.gather(*(one(c) for c in batch)))

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[TurnTrajectory, dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, Sequence[dict[str, Any]]]:
        """Worst cases first, with the component's own slice foregrounded.

        Round-robin mutates one component per iteration, so this is almost
        always one key. `focus` is why it matters: "it called `sample_column`
        three times on one column" is a sentence about `sample_column`'s
        description, and an undifferentiated turn summary is not.
        """
        records: dict[str, Sequence[dict[str, Any]]] = {}
        ordered = sorted(eval_batch.trajectories or [], key=lambda t: t.score.value)

        for component in components_to_update:
            records[component] = [
                {
                    "Inputs": {"the question": t.case.question},
                    "Generated Outputs": _turn_summary(t.replayed),
                    **self._focus(component, t.replayed),
                    "Feedback": t.score.text(),
                    "score": round(t.score.value, 3),
                    "score_breakdown": {
                        k: round(v, 3) for k, v in t.score.terms.items()
                    },
                }
                for t in ordered
            ]
        keep(self.reflections, candidate, records)
        return records


def keep(
    path: Path | None,
    candidate: dict[str, str],
    records: dict[str, Sequence[dict[str, Any]]],
) -> None:
    """One JSON line per reflection: the candidate it was about and exactly the
    records the teacher model was handed.

    GEPA's own run log keeps what the reflection *proposed* and never what it
    *read*. The claim that a proposal came from the feedback can only be shown
    with the two side by side, and until this existed the input half was the
    return value of a function and nothing else. Under the run directory, so
    it is wiped with the run: for `extract` the records hold harvested prompts,
    which is why that directory is ignored in the first place.
    """
    if path is None or not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"candidate": candidate, "records": records}, ensure_ascii=False, default=str
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _turn_summary(r: TurnReplayed) -> str:
    """What the turn did, for a reader who has the feedback beside it."""
    lines = [f"tools, in order: {', '.join(r.tool_names) or '(none)'}"]
    if r.fix_attempts:
        lines.append(f"fix attempts: {r.fix_attempts}")
    lines.append(f"SQL: {r.sql or '(none written)'}")
    lines.append(f"answer: {r.answer or '(none)'}")
    lines.append(f"tokens: {r.tokens}")
    if r.error:
        lines.append(f"error: {r.error}")
    return "\n".join(lines)


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
