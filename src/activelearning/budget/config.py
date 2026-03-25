from pydantic import BaseModel

from activelearning.budget.budget import Budget
from activelearning.budget.schedule_config import (
    ConstantScheduleConfig,
    ScheduleConfig,
    SigmoidIterationScheduleConfig,
    constant_schedule,
    sigmoid_iteration_schedule,
)


class BudgetConfig(BaseModel):
    available_budget: float
    schedule: ScheduleConfig

    def build(self) -> Budget:
        if isinstance(self.schedule, ConstantScheduleConfig):
            schedule = constant_schedule(self.schedule.value)
            return Budget(available_budget=self.available_budget, schedule=schedule)
        if isinstance(self.schedule, SigmoidIterationScheduleConfig):
            schedule = sigmoid_iteration_schedule(
                total_budget=self.available_budget,
                num_iterations=self.schedule.num_iterations,
                midpoint_fraction=self.schedule.midpoint_fraction,
                steepness=self.schedule.steepness,
            )
            return Budget(available_budget=self.available_budget, schedule=schedule)
        raise ValueError(f"Unsupported schedule type '{self.schedule}'.")
