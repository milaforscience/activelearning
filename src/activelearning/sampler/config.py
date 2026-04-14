from pathlib import Path
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
from activelearning.sampler.hypercube_sampler import HypercubeSampler
from activelearning.sampler.pool_file_sampler import PoolFileSampler
from activelearning.sampler.sampler import Sampler


class HypercubeSamplerConfig(BaseModel):
    type: Literal["HypercubeSampler"] = "HypercubeSampler"
    bounds: list[tuple[float, float]]
    num_samples: int = Field(gt=0)
    fidelities: dict[int, float] | list[int] | None = None
    point_strategy: Literal["uniform", "lhs"] = "uniform"

    def build(self) -> Sampler:
        return HypercubeSampler(
            bounds=self.bounds,
            num_samples=self.num_samples,
            fidelities=self.fidelities,
            point_strategy=self.point_strategy,
        )


class PoolFileSamplerConfig(BaseModel):
    """Configuration for :class:`~activelearning.sampler.pool_file_sampler.PoolFileSampler`.

    Parameters
    ----------
    candidate_pool_file : Path
        Path to a text file with one candidate entry per line.
    num_samples : int
        Maximum number of candidates to return per call.
    fidelities : list[int] or dict[int, float] or None
        Fidelity assignment strategy.  ``list[int]`` → uniform random;
        ``dict[int, float]`` → cost-inverse weighted; ``None`` → no fidelity.
    """

    type: Literal["PoolFileSampler"] = "PoolFileSampler"
    candidate_pool_file: Path
    num_samples: int = Field(gt=0)
    fidelities: dict[int, float] | list[int] | None = None

    def build(self, runtime=None) -> Sampler:
        return PoolFileSampler(
            candidate_pool_file=self.candidate_pool_file,
            num_samples=self.num_samples,
            fidelities=self.fidelities,
        )


SamplerConfig = Annotated[
    Union[HypercubeSamplerConfig, PoolFileSamplerConfig],
    Field(discriminator="type"),
]
