from pathlib import Path
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field, model_validator
from activelearning.applications.molecules.constants import SELFIES_VOCAB_SMALL
from activelearning.sampler.fidelity_policy import (
    DiscreteFidelityPolicyConfig,
    JointSamplingFidelityPolicyConfig,
    SamplerFidelityPolicyConfig,
)
from activelearning.sampler.hypercube_sampler import HypercubeSampler
from activelearning.sampler.sampler import Sampler
from activelearning.sampler.pool_file_sampler import PoolFileSampler
from activelearning.sampler.gflownet.config_utils import compose_gflownet_conf
from activelearning.sampler.gflownet.grid_sampler import GFlowNetGridSampler
from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler
from activelearning.sampler.random_token_sequence_sampler import (
    RandomTokenSequenceSampler,
)

_FidelityAction = Literal["any", "first", "last"]


class HypercubeSamplerConfig(BaseModel):
    type: Literal["HypercubeSampler"] = "HypercubeSampler"
    bounds: list[tuple[float, float]]
    num_samples: int = Field(gt=0)
    fidelity_policy: DiscreteFidelityPolicyConfig | None = None
    point_strategy: Literal["uniform", "lhs"] = "uniform"

    def build(self) -> Sampler:
        return HypercubeSampler(
            bounds=self.bounds,
            num_samples=self.num_samples,
            fidelity_policy=(
                self.fidelity_policy.build()
                if self.fidelity_policy is not None
                else None
            ),
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
    fidelity_policy : fidelity policy config or None
        Policy controlling candidate fidelity assignment.
    """

    type: Literal["PoolFileSampler"] = "PoolFileSampler"
    candidate_pool_file: Path
    num_samples: int = Field(gt=0)
    fidelity_policy: DiscreteFidelityPolicyConfig | None = None

    def build(self, runtime=None) -> Sampler:
        return PoolFileSampler(
            candidate_pool_file=self.candidate_pool_file,
            num_samples=self.num_samples,
            fidelity_policy=(
                self.fidelity_policy.build()
                if self.fidelity_policy is not None
                else None
            ),
        )


class RandomTokenSequenceSamplerConfig(BaseModel):
    """Configuration for random token-by-token sequence sampling."""

    type: Literal["RandomTokenSequenceSampler"] = "RandomTokenSequenceSampler"
    tokens: list[str] | Literal["SELFIES_VOCAB_SMALL"]
    num_samples: int = Field(gt=0)
    min_length: int = Field(default=1, ge=1)
    max_length: int = Field(ge=1)
    fidelity_policy: DiscreteFidelityPolicyConfig | None = None
    seed_offset: int = 0
    unique: bool = True
    max_attempts: int = Field(default=100000, gt=0)

    def build(self) -> Sampler:
        tokens = (
            list(SELFIES_VOCAB_SMALL)
            if self.tokens == "SELFIES_VOCAB_SMALL"
            else self.tokens
        )
        return RandomTokenSequenceSampler(
            tokens=tokens,
            num_samples=self.num_samples,
            min_length=self.min_length,
            max_length=self.max_length,
            fidelity_policy=(
                self.fidelity_policy.build()
                if self.fidelity_policy is not None
                else None
            ),
            seed_offset=self.seed_offset,
            unique=self.unique,
            max_attempts=self.max_attempts,
        )


class GFlowNetSamplerConfig(BaseModel):
    """Pydantic config for :class:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler`.

    GFlowNet component defaults (env, policy, loss, buffer, evaluator, logger,
    proxy) are loaded automatically from ``config/gflownet/`` via
    :func:`~activelearning.sampler.gflownet.config_utils.compose_gflownet_conf`.
    Only experiment-specific overrides need to be provided in ``conf``.

    Parameters
    ----------
    type : Literal["GFlowNetSampler"]
        Discriminator field for the :data:`SamplerConfig` union.
    n_samples : int
        Number of candidates to generate per :meth:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler.sample` call.
    fidelity_policy : fidelity policy config or None
        Shared sampler fidelity policy. ``joint_sampling`` enables true multi-fidelity
        GFlowNet sampling; other policies stamp fidelities after decoding.
    reward_fidelity : int or None
        Optional proxy-scoring fidelity override used by GFlowNet experiments
        that train proposals against one fidelity while returning candidates with
        another declared fidelity policy.
    log_dir : str or None
        Root directory for GFlowNet logs.  A temporary directory is created
        automatically when ``None``.
    conf : dict[str, Any] or None
        Experiment-specific overrides for the GFlowNet config, following the
        top-level key structure (``env``, ``gflownet``, ``policy``, ``logger``,
        ``proxy``, etc.).  Deep-merged over the YAML defaults when provided.
    """

    type: Literal["GFlowNetSampler"] = "GFlowNetSampler"
    n_samples: int = Field(gt=0)
    fidelity_policy: SamplerFidelityPolicyConfig | None = None
    reward_fidelity: int | None = Field(default=None, gt=0)
    log_dir: str | None = None
    conf: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_reward_fidelity(self) -> "GFlowNetSamplerConfig":
        if (
            isinstance(self.fidelity_policy, JointSamplingFidelityPolicyConfig)
            and self.reward_fidelity is not None
        ):
            raise ValueError(
                "reward_fidelity is not supported when fidelity_policy.type is joint_sampling."
            )
        return self

    def build(self) -> Sampler:
        return GFlowNetSampler(
            n_samples=self.n_samples,
            conf=compose_gflownet_conf(conf_overrides=self.conf, log_dir=self.log_dir),
            fidelity_policy=(
                self.fidelity_policy.build()
                if self.fidelity_policy is not None
                else None
            ),
            reward_fidelity=self.reward_fidelity,
        )


class GFlowNetGridSamplerConfig(GFlowNetSamplerConfig):
    """Pydantic config for :class:`~activelearning.sampler.gflownet.grid_sampler.GFlowNetGridSampler`.

    Extends :class:`GFlowNetSamplerConfig` with coordinate rescaling from the
    native grid domain to a target bounded domain.

    Parameters
    ----------
    type : Literal["GFlowNetGridSampler"]
        Discriminator field for the :data:`SamplerConfig` union.
    output_bounds : list[tuple[float, float]] or None
        Per-dimension ``(lower, upper)`` bounds to which grid coordinates are
        linearly rescaled.  When ``None``, coordinates are returned in the
        native ``[cell_min, cell_max]`` grid domain.
    """

    type: Literal["GFlowNetGridSampler"] = "GFlowNetGridSampler"  # type: ignore[assignment]
    output_bounds: list[tuple[float, float]] | None = None

    def build(self) -> Sampler:
        return GFlowNetGridSampler(
            n_samples=self.n_samples,
            conf=compose_gflownet_conf(conf_overrides=self.conf, log_dir=self.log_dir),
            output_bounds=self.output_bounds,
            fidelity_policy=(
                self.fidelity_policy.build()
                if self.fidelity_policy is not None
                else None
            ),
            reward_fidelity=self.reward_fidelity,
        )


SamplerConfig = Annotated[
    Union[
        HypercubeSamplerConfig,
        PoolFileSamplerConfig,
        RandomTokenSequenceSamplerConfig,
        GFlowNetSamplerConfig,
        GFlowNetGridSamplerConfig,
    ],
    Field(discriminator="type"),
]
