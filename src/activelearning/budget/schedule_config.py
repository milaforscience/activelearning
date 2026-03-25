import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


def constant_schedule(value: float):
    def schedule(_: int) -> float:
        return value

    return schedule


def sigmoid_iteration_schedule(
    total_budget: float,
    num_iterations: int,
    midpoint_fraction: float,
    steepness: float,
):
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint_fraction)))

    # Use CDF increments so cumulative spend follows a sigmoid.
    weights = [
        _sigmoid((i + 1) / num_iterations) - _sigmoid(i / num_iterations)
        for i in range(num_iterations)
    ]
    weight_sum = sum(weights)
    allocations = [total_budget * weight / weight_sum for weight in weights]

    def schedule(current_round: int) -> float:
        if current_round < 0 or current_round >= num_iterations:
            return 0.0
        return allocations[current_round]

    return schedule


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
