from abc import ABC
from typing import Callable, Iterable, Optional

from activelearning.runtime import ALRuntimeMixin
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


class Acquisition(ABC, ALRuntimeMixin):
    """Abstract acquisition interface used to evaluate candidate utility.

    Acquisition functions assign utility scores to proposed queries using a
    surrogate model. Two scoring modes are supported:

    1. Singleton scoring:
       Scores each candidate independently. Suitable when the utility of a
       candidate does not depend on other candidates being evaluated together.

    2. Batch scoring:
       Scores each candidate batch jointly. Required when the utility of a
       candidate depends on the other candidates selected alongside it.

    Implement ``score()`` to support singleton scoring, ``score_batches()``
    to support joint batch scoring, or both. The ``supports_singleton_scoring``
    and ``supports_batch_scoring`` properties reflect whichever methods are
    implemented.

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

    def __init__(self) -> None:
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

        Called whenever the surrogate is refit or updated. Implementations
        may use this hook to:

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

    def score(
        self,
        candidates: Iterable[Candidate],
        cost_weighting: Optional[
            Callable[[list[float], list[Candidate]], list[float]]
        ] = None,
    ) -> list[float]:
        """Score candidates independently.

        This method is intentionally optional. Subclasses that only support
        joint batch scoring need not implement it. Callers should check
        ``supports_singleton_scoring`` before calling this method.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Iterable of candidates to score one-by-one. Implementations may
            materialize this iterable internally if multiple passes are needed.
        cost_weighting : callable, optional
            If provided, called as ``cost_weighting(raw_scores, candidates)``
            after scoring and its return value is used in place of the raw
            scores. The caller has full control over how cost is incorporated,
            for example dividing by per-candidate cost for cost-efficiency:
            ``lambda scores, cands: [s / cost(c) for s, c in
            zip(scores, cands)]``. Has no effect before ``update()`` has been
            called.

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
        cost_weighting: Optional[
            Callable[[list[float], list[list[Candidate]]], list[float]]
        ] = None,
    ) -> list[float]:
        """Score candidate batches jointly.

        This method is intentionally optional. Subclasses that only support
        singleton scoring need not implement it. Callers should check
        ``supports_batch_scoring`` before calling this method.

        Parameters
        ----------
        candidate_batches : Iterable[Iterable[Candidate]]
            Iterable of candidate batches. Each inner iterable represents one
            jointly scored batch. Implementations may materialize these iterables
            internally if multiple passes or shape checks are needed.
        cost_weighting : callable, optional
            If provided, called as ``cost_weighting(raw_scores, batches)``
            after scoring and its return value is used in place of the raw
            scores. The caller has full control over how cost is incorporated,
            for example dividing by total batch cost for cost-efficiency:
            ``lambda scores, batches: [s / sum_cost(b) for s, b in
            zip(scores, batches)]``. Has no effect before ``update()`` has
            been called.

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

    @property
    def supports_singleton_scoring(self) -> bool:
        """Return whether this acquisition supports independent singleton scoring.

        Returns
        -------
        result : bool
            ``True`` if ``score()`` is implemented.
        """
        return getattr(self.score, "__func__", None) != Acquisition.score

    @property
    def supports_batch_scoring(self) -> bool:
        """Return whether this acquisition supports joint batch scoring.

        Returns
        -------
        result : bool
            ``True`` if ``score_batches()`` is implemented.
        """
        return (
            getattr(self.score_batches, "__func__", None) != Acquisition.score_batches
        )
