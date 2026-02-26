from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Sequence, Any

from activelearning.utils.types import Candidate, Observation

from activelearning.dataset.dataset import Dataset


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

        This method is called at each iteration of the active learning loop,
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

    def update(self, dataset: "Dataset") -> None:
        """Update the surrogate model with a dataset (optional method).

        This method is called by the active learning loop to update the surrogate
        with new observations. Implementations can choose between:
        1. Full retraining: Use dataset.get_observations_iterable() to refit from scratch
        2. Partial updates: Use dataset.get_latest_observations_iterable() for incremental learning

        The choice between full and partial updates is implementation-specific and
        may be configurable via constructor parameters.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing all observations and latest observations.

        Notes
        -----
            - Implementations should decide internally whether to do full refit or partial update
            - For full refit: call fit(dataset.get_observations_iterable())
            - For partial update: use dataset.get_latest_observations_iterable() with
              incremental learning algorithms
            - The default implementation raises NotImplementedError

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
                    new_obs = dataset.get_latest_observations_iterable()
                    self._incremental_update(new_obs)
                else:
                    self.fit(dataset.get_observations_iterable())
            ```
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement update(). "
            "This surrogate may require explicit fit() calls."
        )
