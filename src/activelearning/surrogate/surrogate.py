from abc import ABC, abstractmethod
from typing import Mapping, Sequence, Any

from activelearning.utils.types import Candidate, Observation


class Surrogate(ABC):
    """Abstract surrogate interface used by acquisitions and the AL loop."""

    @abstractmethod
    def fit(self, observations: Sequence[Observation]) -> None:
        """Fit the surrogate model to observations."""
        pass

    @abstractmethod
    def predict(self, candidates: Sequence[Candidate]) -> Mapping[str, Any]:
        """Return a prediction payload (keys are model-specific).

        Common keys include "mean", "std", "samples", or "posterior",
        but acquisitions should document the keys they require.
        """
        pass
