from abc import ABC, abstractmethod
from typing import Mapping, Sequence, Any

from activelearning.utils.types import Candidate, Observation


class Surrogate(ABC):
    """Abstract surrogate interface used by acquisitions and the active learning loop.

    Surrogate models approximate the true evaluation function based on
    observed data, enabling efficient candidate evaluation.

    Note: The predict() method is optional. Surrogates that only work with
    specific acquisition functions (e.g., those using internal posterior
    representations) may not implement general prediction.
    """

    @abstractmethod
    def fit(self, observations: Sequence[Observation]) -> None:
        """Fit the surrogate model to observations.

        This method may be called multiple times during the active learning loop
        as new observations are collected. Implementations are not required to
        support incremental learning (full retraining is acceptable).

        Args:
            observations: Sequence of observations to train on.
        """
        pass

    def predict(self, candidates: Sequence[Candidate]) -> Mapping[str, Any]:
        """Predict values for candidates (optional method).

        Returns a prediction payload with model-specific keys. Acquisition
        functions should document which keys they require.

        Common prediction keys (conventions):
            "mean": List[float] - Predicted mean values for each candidate
            "std": List[float] - Predicted standard deviations (uncertainty)
            "posterior": object - Full posterior distribution (e.g., BoTorch posterior)
            "samples": List[List[float]] - Posterior samples for MC-based acquisitions

        Not all surrogates need to implement this method. Some may only work with
        specific acquisition functions that access internal model representations
        directly rather than through a general prediction interface.

        Args:
            candidates: Sequence of candidates to predict.

        Returns:
            Dictionary mapping prediction types to values. All sequences should
            have the same length and order as the input candidates.

        Raises:
            NotImplementedError: If this surrogate does not support general prediction.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict(). "
            "This surrogate may only work with specific acquisition functions."
        )
