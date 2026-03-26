from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

from activelearning.utils.types import Candidate
from activelearning.runtime import ALRuntimeMixin


class Selector(ABC, ALRuntimeMixin):
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
    ) -> Sequence[Candidate]:
        """Select candidates from a pool based on a specific strategy.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Pool of candidates to select from.
        acquisition : Optional[Any]
            Acquisition function to score candidates (optional).
            Required by score-based selectors, unused by random selectors.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Function to compute per-candidate costs (optional).
            Required by cost-aware selectors.
        round_budget : Optional[float]
            Budget limit for this round (optional).
            Required by cost-aware selectors.
        Returns
        -------
        result : Sequence[Candidate]
            Selected subset of candidates (order may be significant).
        """
        pass
