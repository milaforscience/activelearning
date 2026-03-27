from typing import Callable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.selector.selector import Selector
from activelearning.utils.types import Candidate


class CostAwareSelector(Selector):
    """Selector that maximizes acquisition value within budget using greedy knapsack.

    Selects candidates by "bang for buck" (acquisition value/cost ratio) until the
    budget is exhausted. Does not require a fixed number of samples.
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
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Function returning per-candidate costs.
        round_budget : Optional[float]
            Maximum budget for this selection round.
        Returns
        -------
        result : list[Candidate]
            List of selected candidates within budget constraint.

        Raises
        ------
        ValueError
            If acquisition, cost_fn, or round_budget not provided.
        """
        if acquisition is None:
            raise ValueError("Acquisition function is required for CostAwareSelector.")
        if cost_fn is None:
            raise ValueError("Cost function is required for CostAwareSelector.")
        if round_budget is None:
            raise ValueError("Budget is required for CostAwareSelector.")

        if not candidates:
            return []

        # Get acquisition values and costs for all candidates
        acq_values = acquisition.score(candidates)
        costs = cost_fn(candidates)

        # Reject negative costs and calculate bang-for-buck ratios
        ratios = []
        for idx, (acq_value, cost) in enumerate(zip(acq_values, costs)):
            if cost < 0:
                raise ValueError("Cost function returned a negative cost.")
            if cost == 0:
                # Infinite value for zero cost - select first
                ratio = float("inf")
            else:
                ratio = acq_value / cost
            ratios.append((ratio, idx))

        # Sort by ratio descending (highest bang-for-buck first)
        ratios.sort(key=lambda x: x[0], reverse=True)

        # Greedily select candidates until budget exhausted
        selected = []
        budget_used = 0.0

        for _, idx in ratios:
            candidate_cost = costs[idx]
            if budget_used + candidate_cost <= round_budget:
                selected.append(candidates[idx])
                budget_used += candidate_cost
                if budget_used >= round_budget:
                    break

        return selected
