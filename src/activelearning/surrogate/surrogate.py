from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from activelearning.utils.types import Candidate, Observation
from activelearning.runtime import ALRuntimeMixin


class Surrogate(ABC, ALRuntimeMixin):
    """Abstract surrogate interface for approximating the objective function.

    Surrogate models approximate the objective based on observed data, enabling
    efficient candidate evaluation without querying the oracles directly.

    Surrogates declare whether they support incremental updates or require full
    retraining via ``updates_from_latest()``.

    Notes
    -----
    The ``predict()`` method is optional. Surrogates that only work with
    specific acquisition functions via internal representations may not
    implement general prediction.
    """

    @abstractmethod
    def updates_from_latest(self) -> bool:
        """Declare whether this surrogate supports incremental updates.

        Surrogates that can update their model from new observations without
        full retraining should return ``True``. Surrogates that require full
        retraining on all observations should return ``False``.

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
        """Incrementally update the surrogate with new observations.

        Implementations should update the model using only the provided
        observations, without full retraining. This is suited for online or
        continuous learning scenarios where efficiency matters.

        Surrogates that always return ``False`` from ``updates_from_latest()``
        do not need to implement this method.

        Parameters
        ----------
        observations : Iterable[Observation]
            The most recent observations to incorporate into the model.

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
        """Fit the surrogate model to all current observations.

        Parameters
        ----------
        observations : Iterable[Observation]
            All current observations to train on.

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

    def predict(self, candidates: Sequence[Candidate]) -> Mapping[str, Any]:
        """Predict values for candidates (optional method).

        Returns a prediction payload with model-specific keys. Acquisition
        functions should document which keys they require.

        Common prediction keys (conventions):
            "mean": List[float] - Predicted mean values for each candidate
            "std": List[float] - Predicted standard deviations (uncertainty)
            "posterior": object - Full posterior distribution
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


class MultiFidelitySurrogate(Surrogate, ABC):
    """Runtime contract for surrogates that model multiple fidelity levels.

    Implementations receive the oracle's discrete-level-to-confidence mapping
    before their first fit or update. They must use the mapping consistently
    when encoding observations and candidates.
    """

    @abstractmethod
    def set_fidelity_confidences(self, confidences: dict[int, float]) -> None:
        """Set fidelity confidence metadata before fitting.

        Parameters
        ----------
        confidences : dict[int, float]
            Mapping from each declared integer fidelity level to its encoded
            confidence value.
        """


@runtime_checkable
class TargetFidelityProjector(Protocol):
    """Optional contract for model-space target-fidelity projection.

    Multi-fidelity acquisitions use this protocol only when they need to
    project encoded candidates to a designated target fidelity.
    """

    @property
    def is_multi_fidelity(self) -> bool:
        """Whether the surrogate currently models multiple fidelities."""

    def get_fidelity_dimension(self) -> int | None:
        """Return the fidelity column index in model-space inputs."""

    def get_target_fidelity_value(self) -> float | None:
        """Return the encoded model-space target fidelity value."""
