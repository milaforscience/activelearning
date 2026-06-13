"""Budget tracking utilities for active learning runs."""

import logging
from typing import Callable
from activelearning.runtime import ALRuntimeMixin

logger = logging.getLogger(__name__)


class Budget(ALRuntimeMixin):
    """Manages budget allocation and consumption for active learning rounds.

    The Budget class tracks remaining budget and provides per-round budget
    allocation via a configurable schedule function. It ensures costs do not
    exceed available budget and provides consumption tracking.

    Attributes
    ----------
    available_budget : float
        Remaining budget available for consumption.
    schedule : Callable[[int], float]
        Function mapping round number to allocated budget for that round.
    """

    def __init__(
        self, available_budget: float, schedule: Callable[[int], float]
    ) -> None:
        """Initialize the Budget with total budget and scheduling function.

        Parameters
        ----------
        available_budget : float
            Total budget available for all active learning rounds.
        schedule : Callable[[int], float]
            Callable taking round number (int) and returning budget
            allocation (float) for that round.
        """
        available_budget = float(available_budget)
        if available_budget < 0:
            raise ValueError(
                f"Initial available_budget {available_budget:.2f} must be non-negative"
            )
        self.available_budget = available_budget
        self.schedule = schedule

    def validate_schedule(self, min_query_cost: float) -> None:
        """Validate that every round's budget can afford at least one query.

        Infers the number of active rounds by iterating the schedule until it
        returns 0 (out-of-range signal) or cumulative allocations exceed
        ``available_budget``. Call this at experiment setup time to fail fast
        when the schedule would produce rounds too cheap to query anything.

        Parameters
        ----------
        min_query_cost : float
            Cheapest possible single oracle query cost. The schedule must
            allocate at least this much budget for every round.

        Raises
        ------
        ValueError
            If any round's scheduled allocation is less than
            ``min_query_cost``.
        """
        underfunded_rounds = []
        cumulative = 0.0
        i = 0

        while True:
            allocation = self.schedule(i)
            if allocation <= 0.0:
                break
            cumulative += allocation
            if cumulative > self.available_budget:
                break
            if allocation < min_query_cost:
                underfunded_rounds.append((i, allocation))
            i += 1

        if underfunded_rounds:
            rounds_str = ", ".join(
                f"round {r} (budget={b:.4g})" for r, b in underfunded_rounds
            )
            raise ValueError(
                f"Budget schedule assigns less than the minimum oracle query "
                f"cost ({min_query_cost:.4g}) for: {rounds_str}. "
                f"The experiment would terminate prematurely because no oracle "
                f"query can be afforded in these rounds. Adjust the schedule "
                f"parameters so that every round receives at least "
                f"{min_query_cost:.4g} budget."
            )

    def get_round_budget(self, current_round: int) -> float:
        """Calculate the budget allocated for a specific active learning round.

        Uses the schedule function to determine the round budget, ensuring
        it does not exceed the currently available budget. If the schedule
        returns more than available, caps at available_budget and logs a warning.

        Parameters
        ----------
        current_round : int
            The active learning round number (0-indexed or 1-indexed
            depending on schedule implementation).

        Returns
        -------
        round_budget : float
            Budget allocated for the specified round, capped at available_budget.
        """
        scheduled_budget = self.schedule(current_round)

        if scheduled_budget > self.available_budget:
            logger.warning(
                f"Scheduled budget {scheduled_budget:.2f} for round {current_round} "
                f"exceeds available budget {self.available_budget:.2f}. "
                f"Capping at available budget."
            )
            return self.available_budget

        return scheduled_budget

    def consume(self, cost: float) -> None:
        """Consume budget by deducting the specified cost.

        Parameters
        ----------
        cost : float
            Amount to deduct from available_budget.

        Raises
        ------
        ValueError
            If cost exceeds available_budget.
        """
        if cost > self.available_budget:
            raise ValueError(
                f"Cost {cost:.2f} exceeds available budget {self.available_budget:.2f}"
            )

        self.available_budget -= cost

    def can_afford(self, cost: float) -> bool:
        """Check if the given cost can be afforded within available budget.

        This is a pure query method with no side effects. Use this to check
        affordability before attempting to consume budget.

        Parameters
        ----------
        cost : float
            Amount to check affordability for.

        Returns
        -------
        can_afford : bool
            True if cost <= available_budget, False otherwise.
        """
        return cost <= self.available_budget
