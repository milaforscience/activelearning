from abc import ABC, abstractmethod
from typing import Sequence, Optional

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


class Acquisition(ABC):
    """Abstract acquisition interface used to score candidates.

    Acquisition functions evaluate the utility of querying candidates
    based on surrogate model predictions.
    """

    def __init__(self) -> None:
        self._surrogate: Optional[Surrogate] = None

    @property
    def surrogate(self) -> Optional[Surrogate]:
        """Get the current surrogate model.

        Returns:
            The surrogate model or None if not yet set.
        """
        return getattr(self, "_surrogate", None)

    def update(
        self, surrogate: Surrogate, observations: Optional[Sequence[Observation]] = None
    ) -> None:
        """Update internal state with a new surrogate model.

        Args:
            surrogate: The surrogate model to use for predictions.
            observations: Optional observations to estimate any internal parameters.
        """
        self._surrogate = surrogate

    @abstractmethod
    def __call__(self, candidates: Sequence[Candidate]) -> Sequence[float]:
        """Compute acquisition values for given candidates.

        Args:
            candidates: Sequence of candidates to score.

        Returns:
            Sequence of acquisition scores for each candidate.
        """
        pass
