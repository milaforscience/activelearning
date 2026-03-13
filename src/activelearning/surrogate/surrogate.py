from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Sequence, Any

from activelearning.utils.types import Candidate, Observation


class Surrogate(ABC):
    """Abstract surrogate interface used by acquisitions and the active learning loop.

    Surrogate models approximate the oracles based on observed data, enabling
    efficient candidate evaluation.

    Surrogates must implement ``updates_from_latest()`` to declare whether they
    support incremental updates from the latest observations or require full
    retraining on all observations.

    Notes
    -----
    The predict() method is optional. Surrogates that only work with
    specific acquisition functions (e.g., those using internal posterior
    representations) may not implement general prediction.
    """

    @abstractmethod
    def updates_from_latest(self) -> bool:
        """Declare whether this surrogate updates incrementally from latest observations.

        Surrogates that support incremental updates (i.e., updating the model
        without full retraining) should return ``True``. Surrogates that require
        full retraining on all observations should return ``False``.

          - Return ``True``  → ``update()`` will be called with only the latest observations.
          - Return ``False`` → ``fit()`` will be called with all observations.

        Returning ``False`` is the safe default. Return ``True`` only when the
        surrogate genuinely supports incremental learning.

        Returns
        -------
        bool
            ``True`` if this surrogate supports incremental updates via ``update()``,
            ``False`` if it requires full retraining via ``fit()``.
        """
        pass

    def update(self, observations: Iterable[Observation]) -> None:
        """Incrementally update the surrogate with the latest (new) observations.

        Called when ``updates_from_latest()`` returns ``True``. Implementations
        should update the model using only the provided observations, without
        full retraining. This is suited for online or continuous learning
        scenarios where efficiency matters.

        Surrogates that always return ``False`` from ``updates_from_latest()``
        do not need to implement this method.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of the most recent observations (from the latest oracle query).

        Notes
        -----
            This method is *not* called when ``updates_from_latest()`` returns
            ``False``. In that case the loop calls ``fit(observations)`` instead.

        Raises
        ------
        NotImplementedError
            If this surrogate does not implement incremental updates.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement update(). "
            "Override update() or ensure updates_from_latest() returns False and override fit()."
        )

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fit the surrogate model to observations from scratch.

        Called by the active learning loop when ``updates_from_latest()`` returns
        ``False``, passing the shared round iterable (same one used by acquisition
        and sampler). Also useful for standalone use such as notebooks and tests.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of all current observations to train on.

        Raises
        ------
        NotImplementedError
            If this surrogate does not implement fit().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement fit(). "
            "Override fit() or ensure updates_from_latest() returns True and override update()."
        )

    def is_fitted(self) -> bool:
        """Return whether the surrogate is ready to make predictions.

        The active learning loop calls this before passing the surrogate to the
        acquisition function. When ``False``, the loop skips ``acquisition.update()``
        and the acquisition falls back to uninformed (e.g. constant) scoring,
        enabling random candidate selection on the first round.

        The default implementation returns ``True``, which is correct for surrogates
        that are always ready to predict regardless of whether they have seen data
        (e.g. simple in-memory or analytic surrogates).

        .. important::
            If your surrogate builds its model from data — meaning ``predict()``
            will fail or produce undefined results before ``fit()`` or ``update()``
            has been called with at least one observation — you **must** override
            this method to return ``False`` until the model has been initialised.

        Returns
        -------
        bool
            ``True`` if the surrogate is ready for prediction, ``False`` otherwise.
        """
        return True

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
