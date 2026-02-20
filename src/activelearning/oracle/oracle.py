from abc import ABC, abstractmethod
from typing import Sequence

from activelearning.utils.types import Candidate, Observation


class Oracle(ABC):
    """Abstract oracle interface used to obtain labels and costs.

    Oracles represent the ground truth evaluation mechanism, which may
    be expensive or time-consuming to query. Oracles can handle multiple
    fidelity levels and route candidates internally.
    """

    def get_supported_fidelities(self) -> list[int]:
        """Return list of fidelity levels this oracle can handle.

        Returns:
            List of integer fidelity levels supported by this oracle.
            For single-fidelity oracles, returns a single-element list.
        """
        return sorted(self.get_fidelity_confidences().keys())

    @abstractmethod
    def get_fidelity_confidences(self) -> dict[int, float]:
        """Return confidence for each supported fidelity.

        Returns:
            Dictionary mapping each fidelity level to a confidence score in [0, 1].
        """
        pass

    @abstractmethod
    def get_costs(self, candidates: Sequence[Candidate]) -> list[float]:
        """Calculate the cost of querying each candidate.

        Costs are determined based on each candidate's fidelity level.

        Args:
            candidates: Sequence of candidates to query.

        Returns:
            List of costs, one per candidate, in the same order as input.

        Raises:
            ValueError: If a candidate has an unsupported fidelity level.
        """
        pass

    @abstractmethod
    def query(self, candidates: Sequence[Candidate]) -> Sequence[Observation]:
        """Query the oracle for labels of the given candidates.

        Handles candidates with mixed fidelity levels internally.
        Budget consumption is the caller's responsibility - calculate costs
        with get_costs() and consume budget before calling this method.

        Args:
            candidates: Sequence of candidates to label.

        Returns:
            Sequence of observations with same length and order as input candidates.

        Raises:
            ValueError: If a candidate has an unsupported fidelity level.
        """
        pass
