from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Sequence, Any

from activelearning.utils.types import Candidate, Observation

from activelearning.dataset.dataset import Dataset


class Surrogate(ABC):
    """Abstract surrogate interface used by acquisitions and the active learning loop.

    Surrogate models approximate the oracles based on
    observed data, enabling efficient candidate evaluation.

    The single mandatory method is ``update(dataset)``, which is called by the
    active learning loop after each round of observations. Implementations may
    also expose ``fit(observations)`` as a convenience for standalone use (e.g.
    notebooks, tests), but this is not required by the base interface.

    Notes
    -----
    The predict() method is optional. Surrogates that only work with
    specific acquisition functions (e.g., those using internal posterior
    representations) may not implement general prediction.
    """

    @abstractmethod
    def update(self, dataset: Dataset) -> None:
        """Update the surrogate model with a dataset.

        This is the primary method called by the active learning loop after each
        round of observations. Implementations choose between full retraining or
        incremental updates internally.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing all observations and the latest observations.

        Notes
        -----
            - For full refit: call fit(dataset.get_observations_iterable())
            - For partial update: use dataset.get_latest_observations_iterable()
              with an incremental learning algorithm.

        Examples
        --------
            Full refit implementation:
            ```python
            def update(self, dataset):
                self.fit(dataset.get_observations_iterable())
            ```

            Partial update implementation:
            ```python
            def update(self, dataset):
                if self.use_partial_updates and self._is_fitted:
                    self._incremental_update(dataset.get_latest_observations_iterable())
                else:
                    self.fit(dataset.get_observations_iterable())
            ```
        """
        pass

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fit the surrogate model to observations from scratch (optional convenience method).

        Not called by the active learning loop — the loop always calls ``update(dataset)``.
        Useful for standalone use such as notebooks and tests.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of observations to train on.

        Raises
        ------
        NotImplementedError
            If this surrogate does not expose a standalone fit method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement fit(). "
            "Use update(dataset) instead."
        )

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
