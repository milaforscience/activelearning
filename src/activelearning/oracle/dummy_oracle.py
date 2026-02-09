from typing import Any, Callable, Optional, Sequence

from activelearning.oracle.oracle import Oracle
from activelearning.utils.types import Candidate


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

    def get_cost(self, candidates: Sequence[Candidate]) -> float:
        """Calculate total cost for querying the given candidates.

        Args:
            candidates: Sequence of candidates to query.

        Returns:
            Total cost proportional to number of candidates.
        """
        return self._cost_per_sample * len(candidates)

    def query(self, candidates: Sequence[Candidate]) -> list[float]:
        """Query oracle for labels of the given candidates.

        Args:
            candidates: Sequence of candidates to label.

        Returns:
            List of scores computed via the scoring function.
        """
        scores = []
        for sample in candidates:
            value = sample.x if hasattr(sample, "x") else sample
            scores.append(self._score_fn(value))
        return scores
