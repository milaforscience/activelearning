from abc import ABC, abstractmethod
from typing import Any, Sequence

from activelearning.utils.types import Candidate


class Oracle(ABC):
    """Abstract oracle interface used to obtain labels and costs.

    Oracles represent the ground truth evaluation mechanism, which may
    be expensive or time-consuming to query.
    """

    @abstractmethod
    def get_cost(self, candidates: Sequence[Candidate]) -> float:
        """Calculate the cost of querying the given candidates.

        Args:
            candidates: Sequence of candidates to query.

        Returns:
            Total cost for querying all candidates.
        """
        pass

    @abstractmethod
    def query(self, candidates: Sequence[Candidate]) -> Sequence[Any]:
        """Query the oracle for labels of the given candidates.

        Args:
            candidates: Sequence of candidates to label.

        Returns:
            Sequence of labels/scores for each candidate.
        """
        pass
