"""Pydantic model for the configuration of the budget."""

from pydantic import BaseModel, Field
from typing import Annotated, Literal, Union
from activelearning.budget.budget import Budget
from activelearning.budget.budget_schedule import (
    constant_schedule,
    sigmoid_iteration_schedule,
)


class ConstantScheduleConfig(BaseModel):
    type: Literal["constant"] = "constant"
    value: float = Field(ge=0.0)


class SigmoidIterationScheduleConfig(BaseModel):
    type: Literal["sigmoid_iterations"] = "sigmoid_iterations"
    num_iterations: int = Field(gt=0)
    midpoint_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    steepness: float = Field(default=10.0, gt=0.0)


ScheduleConfig = Annotated[
    Union[
        ConstantScheduleConfig,
        SigmoidIterationScheduleConfig,
    ],
    Field(discriminator="type"),
]


class BudgetConfig(BaseModel):
    available_budget: float = Field(ge=0.0)
    schedule: ScheduleConfig
    max_rounds: int | None = Field(default=None, gt=0)

    def build(self) -> Budget:
        if isinstance(self.schedule, ConstantScheduleConfig):
            schedule = constant_schedule(self.schedule.value)
            return Budget(
                available_budget=self.available_budget,
                schedule=schedule,
                max_rounds=self.max_rounds,
            )
        if isinstance(self.schedule, SigmoidIterationScheduleConfig):
            schedule = sigmoid_iteration_schedule(
                total_budget=self.available_budget,
                num_iterations=self.schedule.num_iterations,
                midpoint_fraction=self.schedule.midpoint_fraction,
                steepness=self.schedule.steepness,
            )
            return Budget(
                available_budget=self.available_budget,
                schedule=schedule,
                max_rounds=self.max_rounds,
            )
        raise ValueError(f"Unsupported schedule type '{self.schedule}'.")
