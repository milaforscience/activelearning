import pulp
from typing import Callable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.selector.selector import Selector
from activelearning.utils.types import Candidate


class KnapsackSelector(Selector):
    """Selector that solves a 0/1 knapsack problem over candidate utilities.

    Parameters
    ----------
    time_limit : float, optional
        Optional CBC time limit in seconds.
    verbose : bool, default=False
        Whether to emit solver logs.
    warm_start : bool, default=False
        Whether to seed CBC with the greedy knapsack solution before solving the
        exact mixed-integer program.
    """

    def __init__(
        self,
        time_limit: float | None = None,
        verbose: bool = False,
        warm_start: bool = False,
    ) -> None:
        self.time_limit = time_limit
        self.verbose = verbose
        self.warm_start = warm_start

    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Acquisition] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
    ) -> list[Candidate]:
        """Select the maximum-utility feasible candidate subset.

        Uses PuLP with CBC to solve the exact 0/1 knapsack problem. When
        ``warm_start`` is enabled, the greedy value-to-cost solution is used as
        the solver's initial assignment.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Pool of candidates to select from.
        acquisition : Optional[Acquisition]
            Acquisition function that scores the candidates.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Function returning per-candidate costs.
        round_budget : Optional[float]
            Budget limit for this round.

        Returns
        -------
        result : list[Candidate]
            Selected subset of candidates in original candidate order.

        Raises
        ------
        ValueError
            If acquisition, cost_fn, or round_budget is not provided, or if the
            solver fails to find any feasible solution.
        """
        if acquisition is None:
            raise ValueError("Acquisition function is required for KnapsackSelector.")
        if cost_fn is None:
            raise ValueError("Cost function is required for KnapsackSelector.")
        if round_budget is None:
            raise ValueError("Budget is required for KnapsackSelector.")
        if not candidates:
            return []

        # Acquisition scores should be non-negative
        acq_values = [max(0, v) for v in acquisition(candidates)]
        costs = cost_fn(candidates)
        if any(cost < 0 for cost in costs):
            raise ValueError("Cost function returned a negative cost.")

        # If acquisition scores are all zero, set constant score to pick cheapest items
        if all(v == 0 for v in acq_values):
            acq_values = [1.0] * len(candidates)

        # Initialize the Maximization problem
        prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)

        # Define Decision Variables (0 or 1 for each item)
        n = len(costs)
        x = [pulp.LpVariable(f"item_{i}", cat="Binary") for i in range(n)]

        if self.warm_start:
            warm_start_indices = set(
                greedy_knapsack_indices(acq_values, costs, round_budget)
            )
            for idx, variable in enumerate(x):
                variable.setInitialValue(1 if idx in warm_start_indices else 0)

        # Objective Function: Maximize total acquisition value
        prob += pulp.lpSum([(acq_values[i]) * x[i] for i in range(n)])

        # Constraint: Total cost must be <= budget
        prob += pulp.lpSum([costs[i] * x[i] for i in range(n)]) <= round_budget

        # Solve using the default CBC solver (included with PuLP)
        status = prob.solve(
            pulp.PULP_CBC_CMD(
                timeLimit=self.time_limit,
                msg=self.verbose,
                warmStart=self.warm_start,
            )
        )

        # Extract results
        max_value = pulp.value(prob.objective)
        if max_value is None:
            raise ValueError("No feasible solution found within time limit.")

        if pulp.LpStatus[status] != "Optimal":
            print(
                f"Warning: Optimization ended with status '{pulp.LpStatus[status]}'. "
                "The selected solution may not be optimal."
            )

        selected_candidates = [
            candidates[i] for i in range(n) if pulp.value(x[i]) > 0.5
        ]
        return selected_candidates


def greedy_knapsack_indices(
    values: Sequence[float],
    costs: Sequence[float],
    budget: float,
) -> list[int]:
    """Return greedy knapsack indices ranked by value-to-cost ratio.

    The helper sorts items by decreasing value-to-cost ratio, breaking ties by
    higher value, then lower cost, then original index. Zero-cost items are
    treated as having infinite ratio and are therefore considered first.

    Parameters
    ----------
    values : Sequence[float]
        Per-item objective values to maximize.
    costs : Sequence[float]
        Per-item costs constrained by the round budget.
    budget : float
        Maximum total cost allowed for the selected items.

    Returns
    -------
    result : list[int]
        Indices selected by the greedy heuristic, in the order they were added.

    Raises
    ------
    ValueError
        If ``values`` and ``costs`` do not have the same length.
    """
    if len(values) != len(costs):
        raise ValueError("Values and costs must have the same length.")

    ranked_items = [
        (index, value, cost) for index, (value, cost) in enumerate(zip(values, costs))
    ]
    ranked_items.sort(key=_greedy_knapsack_sort_key)

    selected_indices: list[int] = []
    remaining_budget = budget
    for index, _, cost in ranked_items:
        if cost > remaining_budget:
            continue

        selected_indices.append(index)
        remaining_budget -= cost

    return selected_indices


def _greedy_knapsack_sort_key(
    item: tuple[int, float, float],
) -> tuple[int, float, float, float, int]:
    """Return the greedy ranking key for an indexed knapsack item.

    Python sorts tuples in ascending lexicographic order, so this key encodes
    the greedy preference as:
    1. zero-cost items first;
    2. higher value-to-cost ratio first;
    3. higher absolute value first;
    4. lower cost first;
    5. lower original index first for deterministic ties.

    Descending criteria use negated values because ``list.sort()`` sorts
    smallest-to-largest.
    """
    index, value, cost = item
    if cost == 0:
        # Leading 0 sorts ahead of the non-zero case below, and the remaining
        # fields preserve the documented tie-break order among zero-cost items.
        return (0, 0.0, -value, cost, index)

    return (1, -(value / cost), -value, cost, index)
