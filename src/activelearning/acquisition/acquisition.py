from abc import ABC
from typing import Iterable, Optional

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


class Acquisition(ABC):
    """Abstract acquisition interface used to evaluate candidate utility.

    Acquisition functions assign utility scores to proposed queries using a
    surrogate model. Two scoring modes are supported:

    1. Singleton scoring:
       Scores each candidate independently.

    2. Batch scoring:
       Scores each candidate batch jointly, which is required for true q-batch
       acquisition functions where the value of a point depends on the other
       points selected alongside it.

    Surrogate Coupling
    ------------------
    Most acquisition functions depend on the surrogate for posterior information,
    predictive moments, samples, or internal model access.

    - Generic acquisitions may rely only on ``surrogate.predict()``.
    - Framework-specific acquisitions (e.g. BoTorch-based ones) may require
      richer surrogate capabilities such as direct access to a fitted model,
      encoded candidate tensors, training data, pending points, or fidelity
      metadata.

    Implementations should clearly document their surrogate requirements and
    validate compatibility in ``update()``.
    """

    def __init__(self) -> None:
        """Initialize the acquisition with no attached surrogate."""
        self._surrogate: Optional[Surrogate] = None

    @property
    def surrogate(self) -> Optional[Surrogate]:
        """Return the currently attached surrogate model.

        Returns
        -------
        result : Optional[Surrogate]
            The surrogate currently used by this acquisition, or ``None`` if
            ``update()`` has not been called yet.
        """
        return self._surrogate

    def update(
        self,
        surrogate: Surrogate,
        observations: Optional[Iterable[Observation]] = None,
    ) -> None:
        """Update internal state after the surrogate has been fit or refreshed.

        Called by the active learning loop whenever the surrogate is refit from
        scratch or incrementally updated. Implementations may use this hook to:

        - validate surrogate compatibility,
        - cache a typed surrogate reference,
        - compute auxiliary statistics from observations,
        - rebuild internal acquisition state.

        Parameters
        ----------
        surrogate : Surrogate
            The surrogate model to use for subsequent acquisition scoring.
        observations : Optional[Iterable[Observation]]
            Optional iterable of observations available at the current round.
            This may be a one-pass iterable, so implementations should
            materialize it if multiple passes are needed.

        Notes
        -----
        The base implementation stores the surrogate reference only. Subclasses
        should override this method when they need additional validation or
        bookkeeping, but should typically call ``super().update(...)`` first.
        """
        self._surrogate = surrogate

    def score(self, candidates: Iterable[Candidate]) -> list[float]:
        """Score candidates independently.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Iterable of candidates to score one-by-one. Implementations may
            materialize this iterable internally if multiple passes are needed.

        Returns
        -------
        result : list[float]
            Acquisition scores in the same order as the input candidates.
            Higher values indicate greater expected utility.

        Raises
        ------
        NotImplementedError
            If the acquisition does not support singleton scoring.

        Notes
        -----
        This method is for independent singleton scoring only. It should not be
        used to assign a joint score to an entire set of candidates.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support singleton scoring."
        )

    def score_batches(
        self,
        candidate_batches: Iterable[Iterable[Candidate]],
    ) -> list[float]:
        """Score candidate batches jointly.

        Parameters
        ----------
        candidate_batches : Iterable[Iterable[Candidate]]
            Iterable of candidate batches. Each inner iterable represents one
            jointly scored batch. Implementations may materialize these iterables
            internally if multiple passes, shape checks, or tensor stacking are
            needed.

        Returns
        -------
        result : list[float]
            Batch acquisition scores in the same order as the input batches.
            Higher values indicate greater expected utility for the batch as a
            whole.

        Raises
        ------
        NotImplementedError
            If the acquisition does not support joint batch scoring.

        Notes
        -----
        This method is required for true q-batch acquisition functions where the
        utility of a batch cannot be decomposed into independent per-candidate
        scores.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support joint batch scoring."
        )

    def supports_singleton_scoring(self) -> bool:
        """Return whether independent singleton scoring is supported.

        Returns
        -------
        result : bool
            ``True`` if ``score()`` is supported.
        """
        return False

    def supports_batch_scoring(self) -> bool:
        """Return whether joint batch scoring is supported.

        Returns
        -------
        result : bool
            ``True`` if ``score_batches()`` is supported.
        """
        return False
