"""Pydantic models of samplers.

Changes in the interface of existing samplers should be reflected in this
configuration. New samplers should define their corresponding pydantic model here and
be added to ``SamplerConfig``.
"""

from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, PositiveFloat, StrictInt

from activelearning.sampler.hypercube_sampler import HypercubeSampler
from activelearning.sampler.sampler import Sampler
from activelearning.sampler.pool_file_sampler import PoolFileSampler
from activelearning.sampler.gflownet.config_utils import compose_gflownet_conf
from activelearning.sampler.gflownet.grid_sampler import GFlowNetGridSampler
from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler
from activelearning.utils.types import DEFAULT_FIDELITY

_FidelityAction = Literal["any", "first", "last"]
_FidelityLevels = Annotated[list[StrictInt], Field(min_length=1)]
_FidelityCosts = Annotated[
    dict[StrictInt, PositiveFloat],
    Field(min_length=1),
]
_Fidelities = _FidelityCosts | _FidelityLevels | None


def _resolve_fidelities(
    fidelities: _Fidelities,
) -> _FidelityCosts | _FidelityLevels:
    """Resolve omitted fidelity settings to the single-fidelity default."""
    if fidelities is None:
        return [DEFAULT_FIDELITY]
    return fidelities


class HypercubeSamplerConfig(BaseModel):
    type: Literal["HypercubeSampler"] = "HypercubeSampler"
    bounds: list[tuple[float, float]]
    num_samples: int = Field(gt=0)
    fidelities: _Fidelities = None
    point_strategy: Literal["uniform", "lhs"] = "uniform"

    def build(self) -> Sampler:
        return HypercubeSampler(
            bounds=self.bounds,
            num_samples=self.num_samples,
            fidelities=_resolve_fidelities(self.fidelities),
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
        ``dict[int, float]`` → cost-inverse weighted; ``None`` →
        :data:`~activelearning.utils.types.DEFAULT_FIDELITY` is used as the
        sole level (single-fidelity mode).
    """

    type: Literal["PoolFileSampler"] = "PoolFileSampler"
    candidate_pool_file: Path
    num_samples: int = Field(gt=0)
    fidelities: _Fidelities = None

    def build(self, runtime=None) -> Sampler:
        return PoolFileSampler(
            candidate_pool_file=self.candidate_pool_file,
            num_samples=self.num_samples,
            fidelities=_resolve_fidelities(self.fidelities),
        )


class GFlowNetSamplerConfig(BaseModel):
    """Pydantic config for :class:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler`.

    GFlowNet component defaults (policy, loss, buffer, evaluator, logger,
    proxy) are loaded automatically from ``config/gflownet/`` via
    :func:`~activelearning.sampler.gflownet.config_utils.compose_gflownet_conf`.
    The env config starts from ``env/base.yaml`` only — the ``_target_`` class
    and all env-specific fields must be provided in ``conf``.

    Parameters
    ----------
    type : Literal["GFlowNetSampler"]
        Discriminator field for the :data:`SamplerConfig` union.
    n_samples : int
        Number of candidates to generate per :meth:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler.sample` call.
    fidelities : list[int] or None
        Fidelity levels to generate.  When ``None``, the top-level config validator
        fills this from the oracle's fidelity set; the sampler then operates in
        single-fidelity mode (one level, stamped on every candidate).  A list with
        more than one entry enables multi-fidelity mode: each sampled candidate is
        assigned one of the listed values as its ``fidelity``.  Values must match
        the oracle's ``fidelity_costs`` keys (e.g. ``[1, 2, 3]`` for a three-level
        oracle).
    log_dir : str or None
        Root directory for GFlowNet logs.  A temporary directory is created
        automatically when ``None``.
    conf : dict[str, Any] or None
        Experiment-specific overrides for the GFlowNet config, following the
        top-level key structure (``env``, ``gflownet``, ``policy``, ``logger``,
        ``proxy``, etc.).  Deep-merged over the YAML defaults when provided.
        Must include ``env._target_`` and all required env fields.
    """

    type: Literal["GFlowNetSampler"] = "GFlowNetSampler"
    n_samples: int = Field(gt=0)
    fidelities: _FidelityLevels | None = None
    fidelity_action: _FidelityAction = "any"
    log_dir: str | None = None
    conf: dict[str, Any] | None = None

    def build(self) -> Sampler:
        return GFlowNetSampler(
            n_samples=self.n_samples,
            conf=compose_gflownet_conf(conf_overrides=self.conf, log_dir=self.log_dir),
            fidelities=_resolve_fidelities(self.fidelities),
            fidelity_action=self.fidelity_action,
        )


class GFlowNetGridSamplerConfig(GFlowNetSamplerConfig):
    """Pydantic config for :class:`~activelearning.sampler.gflownet.grid_sampler.GFlowNetGridSampler`.

    Extends :class:`GFlowNetSamplerConfig` with Grid-specific validation and
    the optional ``domain_bounds`` field for per-dimension coordinate ranges.

    Parameters
    ----------
    type : Literal["GFlowNetGridSampler"]
        Discriminator field for the :data:`SamplerConfig` union.
    domain_bounds : list of [lo, hi] pairs, optional
        Per-dimension coordinate ranges, one ``[lo, hi]`` pair per dimension.
        Length must equal ``conf.env.n_dim`` and each pair must satisfy
        ``lo < hi``. When ``None`` (default), all dimensions share the
        ``[cell_min, cell_max]`` range from ``conf.env``.
    """

    type: Literal["GFlowNetGridSampler"] = "GFlowNetGridSampler"  # type: ignore[assignment]
    domain_bounds: list[list[float]] | None = None

    def build(self) -> Sampler:
        return GFlowNetGridSampler(
            n_samples=self.n_samples,
            conf=compose_gflownet_conf(conf_overrides=self.conf, log_dir=self.log_dir),
            fidelities=_resolve_fidelities(self.fidelities),
            fidelity_action=self.fidelity_action,
            domain_bounds=self.domain_bounds,
        )


SamplerConfig = Annotated[
    Union[
        HypercubeSamplerConfig,
        PoolFileSamplerConfig,
        GFlowNetSamplerConfig,
        GFlowNetGridSamplerConfig,
    ],
    Field(discriminator="type"),
]
