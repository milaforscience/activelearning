from typing import Any, Callable, Optional, Sequence

from activelearning.selector.selector import Selector
from activelearning.utils.types import Candidate


class PendingAwareGreedyFantasySelector(Selector):
    """Greedy selector that conditions on already-selected candidates at each step.

    At each of up to ``k`` steps, the already-selected candidates are passed as
    "pending" to the acquisition function, which conditions its scores on the
    expected information gain from those pending observations before scoring the
    remaining candidates. This avoids redundant selections by accounting for
    inter-candidate dependencies.

    Requires an acquisition that exposes ``supports_pending_scoring = True`` and
    implements ``score_with_pending(candidates, pending_candidates)``. All
    BoTorch-backed acquisitions satisfy this requirement.

    Optional budget handling: if both ``cost_fn`` and ``round_budget`` are
    provided, only candidates whose addition keeps the running total cost within
    budget are considered feasible at each step.

    Parameters
    ----------
    k : int
        Maximum number of candidates to select per round.

    Raises
    ------
    ValueError
        If ``k < 1``.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        self.k = k

    def select(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Any] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
    ) -> list[Candidate]:
        """Greedily select up to k candidates using pending-aware rescoring.

        At each step the remaining candidates are rescored conditioned on the
        candidates selected so far (the pending set). The highest-scoring
        feasible candidate is added to the selection and the process repeats.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Pool of candidates to select from.
        acquisition : Optional[Any]
            Acquisition object with ``supports_pending_scoring = True`` and
            a ``score_with_pending(candidates, pending_candidates)`` method.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Function returning per-candidate costs. Must be provided together
            with ``round_budget`` to enable budget-constrained selection.
        round_budget : Optional[float]
            Total budget for the selected set. Must be provided together with
            ``cost_fn`` to enable budget-constrained selection.

        Returns
        -------
        result : list[Candidate]
            Selected candidates in greedy construction order. May contain
            fewer than ``k`` candidates if the pool or budget is exhausted.

        Raises
        ------
        ValueError
            If ``acquisition`` is not provided, if the acquisition does not
            support pending scoring, or if only one of ``cost_fn`` /
            ``round_budget`` is supplied.
        """
        if acquisition is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an acquisition function."
            )
        if not getattr(acquisition, "supports_pending_scoring", False):
            raise ValueError(
                f"{self.__class__.__name__} requires an acquisition that supports "
                "pending-aware scoring (supports_pending_scoring=True). "
                "All BoTorch acquisitions support this."
            )

        use_budget = (cost_fn is not None) or (round_budget is not None)
        if use_budget and (cost_fn is None or round_budget is None):
            raise ValueError(
                "Both cost_fn and round_budget must be provided together for "
                "budget-constrained selection."
            )

        remaining = list(candidates)
        if not remaining:
            return []

        selected: list[Candidate] = []
        spent = 0.0

        while remaining and len(selected) < self.k:
            if use_budget:
                costs = cost_fn(remaining)  # type: ignore[misc]
                if any(c < 0 for c in costs):
                    raise ValueError("Cost function returned a negative cost.")
                feasible_with_costs = [
                    (cand, cost)
                    for cand, cost in zip(remaining, costs)
                    if spent + cost <= round_budget  # type: ignore[operator]
                ]
                if not feasible_with_costs:
                    break
                feasible = [cand for cand, _ in feasible_with_costs]
                feasible_costs = [cost for _, cost in feasible_with_costs]
            else:
                feasible = remaining
                feasible_costs = None

            scores = acquisition.score_with_pending(feasible, list(selected))
            best_idx = max(range(len(feasible)), key=lambda i: scores[i])
            best = feasible[best_idx]
            selected.append(best)

            if use_budget:
                spent += feasible_costs[best_idx]  # type: ignore[index]

            remaining = [c for c in remaining if c is not best]

        return selected
