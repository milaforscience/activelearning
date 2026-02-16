from abc import ABC, abstractmethod
from typing import Sequence

from activelearning.utils.types import Candidate, Observation
from activelearning.budget.budget import Budget


class Oracle(ABC):
    """Abstract oracle interface used to obtain labels and costs.

    Oracles represent the ground truth evaluation mechanism, which may
    be expensive or time-consuming to query.
    """

    @abstractmethod
    def get_costs(self, candidates: Sequence[Candidate]) -> list[float]:
        """Calculate the cost of querying each candidate.

        Args:
            candidates: Sequence of candidates to query.

        Returns:
            List of costs, one per candidate, in the same order as input.
        """
        pass

    @abstractmethod
    def query(
        self, candidates: Sequence[Candidate], budget: Budget
    ) -> Sequence[Observation]:
        """Query the oracle for labels of the given candidates.

        Consumes budget by calculating total cost and calling budget.consume().

        Args:
            candidates: Sequence of candidates to label.
            budget: Budget object to consume costs from.

        Returns:
            Sequence of observations with same length and order as input candidates.

        Raises:
            ValueError: If total cost exceeds available budget (via budget.consume()).
        """
        pass
