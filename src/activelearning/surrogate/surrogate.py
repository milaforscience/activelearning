from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Sequence, Any

from activelearning.utils.types import Candidate, Observation


class Surrogate(ABC):
    """Abstract surrogate interface used by acquisitions and the active learning loop.

    Surrogate models approximate the oracles based on
    observed data, enabling efficient candidate evaluation.

    Notes
    -----
    The predict() method is optional. Surrogates that only work with
    specific acquisition functions (e.g., those using internal posterior
    representations) may not implement general prediction.
    """

    @abstractmethod
    def fit(self, observations: Iterable[Observation]) -> None:
        """Fit the surrogate model to observations.

        This method may be called multiple times during the active learning loop
        as new observations are collected. Implementations are not required to
        support incremental learning (full retraining is acceptable).

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of observations to train on. May be a Sequence
            (list, tuple) for small datasets or a one-pass iterable (DataLoader)
            for large datasets.

        Notes
        -----
            Implementations should validate their input requirements:
            - If you need len() or multiple passes, materialize to list or assert Sequence
            - If you can handle streaming data, consume the iterable directly
        """
        pass

    def set_fidelity_confidences(self, confidences: dict[int, float]) -> None:
        """Set per-fidelity confidence metadata for multi-fidelity surrogate models.

        This method is intended for surrogate models that operate in a multi-fidelity
        setting, where different fidelity levels have associated confidence values.
        The active learning loop calls this method before iterative fitting begins,
        passing the confidence values provided by the oracle.

        Surrogates that do not support or utilize fidelity-specific metadata can
        safely ignore this method, as the default implementation is a no-op.

        Parameters
        ----------
        confidences : dict[int, float]
            Mapping of fidelity levels (integer indices) to confidence
            values in the range [0, 1], where 1 indicates maximum confidence.
        """
        return None

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

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to predict.

        Returns
        -------
        result : Mapping[str, Any]
            Dictionary mapping prediction types to values. All sequences should
            have the same length and order as the input candidates.

        Raises
        ------
        NotImplementedError
            If this surrogate does not support general prediction.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict(). "
            "This surrogate may only work with specific acquisition functions."
        )
