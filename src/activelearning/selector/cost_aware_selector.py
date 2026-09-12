from typing import Callable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.selector.selector import Selector
from activelearning.utils.types import Candidate

# Absolute tolerance for budget comparisons to avoid floating-point accumulation
# drift when many small-cost candidates are summed
_BUDGET_ATOL = 1e-9


class CostAwareSelector(Selector):
    """Selector that maximizes acquisition value within budget using greedy knapsack.

    Selects candidates by "bang for buck" (acquisition value/cost ratio) until the
    budget is exhausted. Does not require a fixed number of samples.

    Divides ``acquisition.score(candidates)`` by oracle cost, so any
    acquisition-level cost weighting is compounded here.
    """

    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Acquisition] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
    ) -> list[Candidate]:
        """Select candidates greedily to maximize utility within budget.

        Implements a greedy knapsack strategy:
        1. Compute acquisition values from acquisition function
        2. Compute costs from cost_fn
        3. Rank candidates by acquisition value/cost ratio (descending)
        4. Greedily select candidates until budget exhausted

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Pool of candidates to select from.
        acquisition : Optional[Acquisition]
            Acquisition function to compute acquisition values for candidates.
            Returned values are used as-is before dividing by
            ``cost_fn(candidates)``.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Function returning per-candidate costs.
        round_budget : Optional[float]
            Maximum budget for this selection round.
        Returns
        -------
        result : list[Candidate]
            Selected candidates in descending utility-per-cost order.

        Raises
        ------
        ValueError
            If acquisition, cost_fn, or round_budget not provided.
        """
        self._clear_selection_scores()
        if acquisition is None:
            raise ValueError("Acquisition function is required for CostAwareSelector.")
        if cost_fn is None:
            raise ValueError("Cost function is required for CostAwareSelector.")
        if round_budget is None:
            raise ValueError("Budget is required for CostAwareSelector.")

        if not candidates:
            return []

        # Get acquisition values and costs for all candidates
        acquisition_scores = list(acquisition.score(candidates))
        costs = list(cost_fn(candidates))
        if len(acquisition_scores) != len(candidates):
            raise ValueError("Acquisition scores must match the candidate pool.")
        if len(costs) != len(candidates):
            raise ValueError("Candidate costs must match the candidate pool.")

        # Reject negative costs and calculate bang-for-buck ratios
        ranking_scores = []
        for acq_value, cost in zip(acquisition_scores, costs):
            if cost < 0:
                raise ValueError("Cost function returned a negative cost.")
            if cost == 0:
                # Infinite value for zero cost - select first
                ratio = float("inf")
            else:
                ratio = acq_value / cost
            ranking_scores.append(ratio)

        # Sort by ratio descending (highest bang-for-buck first)
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: ranking_scores[index],
            reverse=True,
        )

        # Greedily select candidates until budget exhausted
        selected = []
        selected_indices = []
        budget_used = 0.0

        for idx in ranked_indices:
            candidate_cost = costs[idx]
            if budget_used + candidate_cost <= round_budget + _BUDGET_ATOL:
                selected.append(candidates[idx])
                selected_indices.append(idx)
                budget_used += candidate_cost
                if budget_used >= round_budget:
                    break

        if selected_indices:
            self._record_selection_scores(
                acquisition_scores,
                ranking_scores,
                selected_indices,
            )
        return selected
