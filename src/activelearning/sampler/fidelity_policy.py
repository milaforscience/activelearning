"""Shared fidelity policy types for samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import random
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from activelearning.utils.types import Candidate

FidelityAction = Literal["any", "first", "last"]


class FidelityPolicy(ABC):
    """Abstract base for policies that assign fidelities to sampled candidates."""

    @abstractmethod
    def sample_fidelities(self, count: int, rng: random.Random) -> list[int]:
        """Return ``count`` discrete fidelity assignments."""


@dataclass(frozen=True)
class FixedFidelityPolicy(FidelityPolicy):
    """Always assign the same fidelity level."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("Fixed fidelity value must be positive.")

    def sample_fidelities(self, count: int, rng: random.Random) -> list[int]:
        """Return the same fidelity for every requested sample."""

        del rng
        return [self.value] * count


@dataclass(frozen=True)
class UniformFidelityPolicy(FidelityPolicy):
    """Sample fidelities uniformly from a discrete set."""

    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Uniform fidelity policy values must not be empty.")
        if any(value <= 0 for value in self.values):
            raise ValueError("Uniform fidelity policy values must be positive.")

    def sample_fidelities(self, count: int, rng: random.Random) -> list[int]:
        """Draw uniformly random fidelity levels."""

        return rng.choices(self.values, k=count)


@dataclass(frozen=True)
class CostWeightedFidelityPolicy(FidelityPolicy):
    """Sample fidelities with probability inversely proportional to cost."""

    costs: dict[int, float]

    def __post_init__(self) -> None:
        if not self.costs:
            raise ValueError("Cost-weighted fidelity policy costs must not be empty.")
        invalid_costs = {
            fidelity: cost
            for fidelity, cost in self.costs.items()
            if fidelity <= 0 or cost <= 0
        }
        if invalid_costs:
            raise ValueError(
                "Cost-weighted fidelity policy requires positive fidelities and "
                f"strictly positive costs. Got: {invalid_costs}."
            )

    def sample_fidelities(self, count: int, rng: random.Random) -> list[int]:
        """Draw fidelity levels with inverse-cost weighting."""

        levels = sorted(self.costs)
        weights = [1.0 / self.costs[level] for level in levels]
        return rng.choices(levels, weights=weights, k=count)


@dataclass(frozen=True)
class JointSamplingFidelityPolicy:
    """Sample ``x`` and fidelity jointly through a multi-fidelity sampler state space."""

    n_fidelities: int
    action: FidelityAction = "any"

    def __post_init__(self) -> None:
        if self.n_fidelities <= 1:
            raise ValueError("joint_sampling requires n_fidelities > 1.")


DiscreteFidelityPolicy = Union[
    FixedFidelityPolicy,
    UniformFidelityPolicy,
    CostWeightedFidelityPolicy,
]
SamplerFidelityPolicy = Union[DiscreteFidelityPolicy, JointSamplingFidelityPolicy]


def apply_fidelity_policy(
    candidates: list[Candidate],
    policy: DiscreteFidelityPolicy,
    rng: random.Random,
) -> list[Candidate]:
    """Return candidates with fidelities reassigned by ``policy``."""

    sampled_fidelities = policy.sample_fidelities(len(candidates), rng)
    return [
        replace(candidate, fidelity=fidelity)
        for candidate, fidelity in zip(candidates, sampled_fidelities)
    ]


class FixedFidelityPolicyConfig(BaseModel):
    """Pydantic config for :class:`FixedFidelityPolicy`."""

    type: Literal["fixed"] = "fixed"
    value: int = Field(gt=0)

    def build(self) -> FixedFidelityPolicy:
        """Build the runtime fidelity policy."""

        return FixedFidelityPolicy(value=self.value)


class UniformFidelityPolicyConfig(BaseModel):
    """Pydantic config for :class:`UniformFidelityPolicy`."""

    type: Literal["uniform"] = "uniform"
    values: list[int]

    def build(self) -> UniformFidelityPolicy:
        """Build the runtime fidelity policy."""

        return UniformFidelityPolicy(values=tuple(self.values))


class CostWeightedFidelityPolicyConfig(BaseModel):
    """Pydantic config for :class:`CostWeightedFidelityPolicy`."""

    type: Literal["cost_weighted"] = "cost_weighted"
    costs: dict[int, float]

    def build(self) -> CostWeightedFidelityPolicy:
        """Build the runtime fidelity policy."""

        return CostWeightedFidelityPolicy(costs=dict(self.costs))


class JointSamplingFidelityPolicyConfig(BaseModel):
    """Pydantic config for :class:`JointSamplingFidelityPolicy`."""

    type: Literal["joint_sampling"] = "joint_sampling"
    n_fidelities: int = Field(gt=1)
    action: FidelityAction = "any"

    def build(self) -> JointSamplingFidelityPolicy:
        """Build the runtime fidelity policy."""

        return JointSamplingFidelityPolicy(
            n_fidelities=self.n_fidelities,
            action=self.action,
        )


DiscreteFidelityPolicyConfig = Annotated[
    Union[
        FixedFidelityPolicyConfig,
        UniformFidelityPolicyConfig,
        CostWeightedFidelityPolicyConfig,
    ],
    Field(discriminator="type"),
]

SamplerFidelityPolicyConfig = Annotated[
    Union[
        FixedFidelityPolicyConfig,
        UniformFidelityPolicyConfig,
        CostWeightedFidelityPolicyConfig,
        JointSamplingFidelityPolicyConfig,
    ],
    Field(discriminator="type"),
]
