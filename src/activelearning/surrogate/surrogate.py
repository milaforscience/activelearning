from abc import ABC, abstractmethod
from typing import Mapping, Sequence, Any

from activelearning.utils.types import Candidate, Observation


class Surrogate(ABC):
    """Abstract surrogate interface used by acquisitions and the active learning loop.

    Surrogate models approximate the true evaluation function based on
    observed data, enabling efficient candidate evaluation.
    """

    @abstractmethod
    def fit(self, observations: Sequence[Observation]) -> None:
        """Fit the surrogate model to observations.

        Args:
            observations: Sequence of observations to train on.
        """
        pass

    @abstractmethod
    def predict(self, candidates: Sequence[Candidate]) -> Mapping[str, Any]:
        """Predict values for candidates.

        Returns a prediction payload with model-specific keys.
        Common keys include "mean", "std", "samples", or "posterior".
        Acquisition functions should document required keys.

        Args:
            candidates: Sequence of candidates to predict.

        Returns:
            Dictionary mapping prediction types to values.
        """
        pass
