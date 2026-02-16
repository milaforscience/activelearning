from typing import Any, Callable, Optional, Sequence

from activelearning.oracle.oracle import Oracle
from activelearning.utils.types import Candidate, Observation
from activelearning.budget.budget import Budget


class DummyOracle(Oracle):
    """Deterministic oracle with constant per-sample cost.

    Args:
        cost_per_sample: Cost incurred per queried candidate.
        score_fn: Function mapping candidate.x to a score value.
    """

    def __init__(
        self,
        cost_per_sample: float = 1.0,
        score_fn: Optional[Callable[[Any], float]] = None,
    ) -> None:
        self._cost_per_sample = float(cost_per_sample)
        self._score_fn = score_fn or (lambda s: float(s))

    def get_costs(self, candidates: Sequence[Candidate]) -> list[float]:
        """Calculate the cost of querying each candidate.

        Args:
            candidates: Sequence of candidates to query.

        Returns:
            List of costs, one per candidate (all equal to cost_per_sample).
        """
        return [self._cost_per_sample for _ in candidates]

    def query(
        self, candidates: Sequence[Candidate], budget: Budget
    ) -> Sequence[Observation]:
        """Query oracle for labels of the given candidates.

        Calculates total cost and consumes from budget before returning observations.

        Args:
            candidates: Sequence of candidates to label.
            budget: Budget object to consume costs from.

        Returns:
            List of scores computed via the scoring function.

        Raises:
            ValueError: If total cost exceeds available budget.
        """
        costs = self.get_costs(candidates)
        total_cost = sum(costs)
        budget.consume(total_cost)

        observations = []
        for sample in candidates:
            value = sample.x
            observation = Observation(
                x=value, y=self._score_fn(value), fidelity=sample.fidelity
            )
            observations.append(observation)
        return observations
