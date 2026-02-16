import logging
from typing import Callable

logger = logging.getLogger(__name__)


class Budget:
    """Manages budget allocation and consumption for active learning rounds.

    The Budget class tracks remaining budget and provides per-round budget
    allocation via a configurable schedule function. It ensures costs do not
    exceed available budget and provides consumption tracking.

    Attributes:
        available_budget: Remaining budget available for consumption.
        schedule: Function mapping round number to allocated budget for that round.
    """

    def __init__(
        self, available_budget: float, schedule: Callable[[int], float]
    ) -> None:
        """Initialize the Budget with total budget and scheduling function.

        Args:
            available_budget: Total budget available for all active learning rounds.
            schedule: Callable taking round number (int) and returning budget
                allocation (float) for that round.
        """
        self.available_budget = float(available_budget)
        self.schedule = schedule

    def get_round_budget(self, current_round: int) -> float:
        """Calculate the budget allocated for a specific active learning round.

        Uses the schedule function to determine the round budget, ensuring
        it does not exceed the currently available budget. If the schedule
        returns more than available, caps at available_budget and logs a warning.

        Args:
            current_round: The active learning round number (0-indexed or 1-indexed
                depending on schedule implementation).

        Returns:
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

        Args:
            cost: Amount to deduct from available_budget.

        Raises:
            ValueError: If cost exceeds available_budget.
        """
        if cost > self.available_budget:
            raise ValueError(
                f"Cost {cost:.2f} exceeds available budget {self.available_budget:.2f}"
            )

        self.available_budget -= cost
