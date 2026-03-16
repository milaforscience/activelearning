from abc import ABC
from typing import Iterable, Optional

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


class Acquisition(ABC):
    """Abstract acquisition interface used to evaluate candidate utility.

    Acquisition functions assign utility scores to proposed queries using a
    surrogate model. Two scoring modes are supported:

    1. Singleton scoring:
       Scores each candidate independently. Suitable when the utility of a
       candidate does not depend on other candidates being evaluated together.

    2. Batch scoring:
       Scores each candidate batch jointly. Required when the utility of a
       candidate depends on the other candidates selected alongside it.

    Subclasses are not required to implement both modes — only the modes
    relevant to the acquisition strategy need to be provided. Subclasses
    declare which modes they support via the ``supports_singleton_scoring``
    and ``supports_batch_scoring`` constructor flags. Callers should check
    these flags before invoking the corresponding scoring method.

    Surrogate Coupling
    ------------------
    Most acquisition functions depend on the surrogate for posterior information,
    predictive moments, or samples.

    - Simple acquisitions may rely only on ``surrogate.predict()``.
    - More specialised acquisitions may require richer surrogate capabilities
      such as direct model access, training data, or pending points.

    Implementations should clearly document their surrogate requirements and
    validate compatibility in ``update()``.
    """

    def __init__(
        self,
        *,
        supports_singleton_scoring: bool = False,
        supports_batch_scoring: bool = False,
    ) -> None:
        """Initialize the acquisition with capability flags.

        Subclasses declare their scoring capabilities by passing the appropriate
        flags here.  The base ``supports_singleton_scoring()`` and
        ``supports_batch_scoring()`` methods read from these flags, so
        subclasses should **not** override those methods — set the flags
        instead.

        Parameters
        ----------
        supports_singleton_scoring : bool, default=False
            Whether this acquisition supports independent candidate scoring via
            ``score()``.
        supports_batch_scoring : bool, default=False
            Whether this acquisition supports joint batch scoring via
            ``score_batches()``.
        """
        self._surrogate: Optional[Surrogate] = None
        self._supports_singleton_scoring = supports_singleton_scoring
        self._supports_batch_scoring = supports_batch_scoring

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
        that override this method should typically validate first, then call
        ``super().update(...)`` once validation succeeds.
        """
        self._surrogate = surrogate

    def score(self, candidates: Iterable[Candidate]) -> list[float]:
        """Score candidates independently.

        This method is intentionally optional. Subclasses that only support
        joint batch scoring need not implement it. Callers should check
        ``supports_singleton_scoring()`` before calling this method.

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
            Always, unless overridden by a subclass that supports singleton
            scoring.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support singleton scoring."
        )

    def score_batches(
        self,
        candidate_batches: Iterable[Iterable[Candidate]],
    ) -> list[float]:
        """Score candidate batches jointly.

        This method is intentionally optional. Subclasses that only support
        singleton scoring need not implement it. Callers should check
        ``supports_batch_scoring()`` before calling this method.

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
            Always, unless overridden by a subclass that supports joint batch
            scoring.
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
        return self._supports_singleton_scoring

    def supports_batch_scoring(self) -> bool:
        """Return whether joint batch scoring is supported.

        Returns
        -------
        result : bool
            ``True`` if ``score_batches()`` is supported.
        """
        return self._supports_batch_scoring
