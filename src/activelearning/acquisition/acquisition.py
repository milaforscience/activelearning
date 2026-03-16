from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

from activelearning.runtime import ALRuntimeMixin
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


class Acquisition(ABC, ALRuntimeMixin):
    """Abstract acquisition interface used to evaluate candidates.

    Acquisition functions evaluate the utility of querying candidates
    based on surrogate model predictions.

    Surrogate Coupling:
        Most acquisition functions rely on the surrogate's predict() method,
        expecting specific keys in the prediction payload (e.g., "mean", "std").
        Some acquisitions may access surrogate internals directly (e.g., posterior
        distributions) instead of using predict(). Implementations should document
        their surrogate requirements.
    """

    def __init__(self) -> None:
        self._surrogate: Optional[Surrogate] = None

    @property
    def surrogate(self) -> Optional[Surrogate]:
        """Get the current surrogate model.

        Returns
        -------
        result : Optional[Surrogate]
            The surrogate model or None if not yet set.
        """
        return getattr(self, "_surrogate", None)

    def update(
        self, surrogate: Surrogate, observations: Optional[Iterable[Observation]] = None
    ) -> None:
        """Update internal state with a new surrogate model.

        Called by the active learning loop when the surrogate is refitted.
        The acquisition can use observations to estimate internal parameters
        (e.g., best observed value for Expected Improvement).

        Surrogate Compatibility:
            Implementations should validate that the surrogate provides required
            capabilities. If the acquisition needs predict() with specific keys,
            consider calling predict() here to fail early. If the acquisition
            accesses surrogate internals, validate the surrogate type/interface.

        Parameters
        ----------
        surrogate : Surrogate
            The surrogate model to use for computing acquisition values for candidates.
            Must provide capabilities required by this acquisition function.
        observations : Optional[Iterable[Observation]]
            Optional iterable of observations for estimating acquisition
            parameters (e.g., best observed value, noise estimates). May be a
            one-pass iterable. Materialize to list if multiple passes are needed.
        """
        self._surrogate = surrogate

    @abstractmethod
    def __call__(self, candidates: Sequence[Candidate]) -> Sequence[float]:
        """Compute acquisition values for given candidates.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to evaluate.

        Returns
        -------
        result : Sequence[float]
            Sequence of acquisition values, same length and order as input.
            Higher acquisition values typically indicate more valuable candidates to query.
        """
        pass
