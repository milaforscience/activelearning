from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

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
        **kwargs: Any,
    ) -> Sequence[Candidate]:
        """Select candidates from a pool based on a specific strategy.

        Args:
            candidates: Pool of candidates to select from.
            acquisition: Acquisition function to score candidates.
            **kwargs: Additional strategy-specific arguments.

        Returns:
            Selected subset of candidates.
        """
        pass
