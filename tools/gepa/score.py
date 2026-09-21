"""What a metric returns: a number for selection, prose for reflection.

Shared by `metric_extract` and `metric_turn`, because the contract is the same
one. The scalar decides which candidates survive. The prose is the Actionable
Side Information — what the reflection model reads before proposing the next
mutation, and the only reason GEPA points a direction rather than sampling at
random.

The prose is the harder half of every metric in this directory, and the one
worth arguing about. A number says a candidate was worse. A sentence saying
`sample_column` was called three times on one column says what to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Score:
    """A scalar for the optimiser and prose for the reflection step.

    `terms` is also what reaches `EvaluationBatch.objective_scores`, which is
    what the Pareto front is drawn from. Every score from one metric must carry
    the same keys, or the front has holes in it that nothing reports.
    """

    value: float
    terms: dict[str, float] = field(default_factory=dict)
    feedback: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n\n".join(self.feedback) if self.feedback else "No problems found."
