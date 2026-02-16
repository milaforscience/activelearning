from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

from activelearning.utils.types import Candidate


class Selector(ABC):
    """Abstract selector interface used to choose candidates.

    Selectors implement strategies for choosing the final subset of
    candidates to query from a larger pool.
    """

    @abstractmethod
    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Any] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
        **kwargs: Any,
    ) -> Sequence[Candidate]:
        """Select candidates from a pool based on a specific strategy.

        Args:
            candidates: Pool of candidates to select from.
            acquisition: Acquisition function to score candidates (optional).
                Required by score-based selectors, unused by random selectors.
            cost_fn: Function to compute per-candidate costs (optional).
                Required by cost-aware selectors.
            round_budget: Budget limit for this round (optional).
                Required by cost-aware selectors.
            **kwargs: Additional strategy-specific arguments.

        Returns:
            Selected subset of candidates (order may be significant).
        """
        pass
