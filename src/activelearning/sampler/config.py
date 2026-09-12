"""Pydantic models of samplers.

Changes in the interface of existing samplers should be reflected in this
configuration. Core samplers are registered by the core registry bootstrap;
application packages provide their sampler schemas through catalog mappings.
"""

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, overload

from pydantic import Field, PositiveFloat, StrictInt

from activelearning.config_registry import (
    BuildableConfig,
    registered_config,
)
from activelearning.sampler.exact_grid_sampler import ExactGridSampler
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


@overload
def _resolve_fidelities(
    fidelities: _FidelityLevels | None,
) -> _FidelityLevels: ...


@overload
def _resolve_fidelities(fidelities: _FidelityCosts) -> _FidelityCosts: ...


def _resolve_fidelities(
    fidelities: _Fidelities,
) -> _FidelityCosts | _FidelityLevels:
    """Resolve omitted fidelity settings to the single-fidelity default."""
    if fidelities is None:
        return [DEFAULT_FIDELITY]
    return fidelities


FidelityLevels = _FidelityLevels


def resolve_fidelities(
    fidelities: _Fidelities,
) -> _FidelityCosts | _FidelityLevels:
    """Resolve omitted fidelity settings for external sampler configs."""
    return _resolve_fidelities(fidelities)


class HypercubeSamplerConfig(BuildableConfig):
    """Configuration for uniform or Latin-hypercube numeric sampling.

    Parameters
    ----------
    bounds : list[tuple[float, float]]
        Inclusive lower and upper bounds for each numeric dimension.
    num_samples : int
        Number of candidates to generate.
    fidelities : list[int] or dict[int, float] or None
        Fidelity levels or cost mapping used to assign candidate fidelities.
        ``None`` selects the default single fidelity.
    point_strategy : {"uniform", "lhs"}
        Numeric point-generation strategy.
    """

    type: Literal["HypercubeSampler"] = "HypercubeSampler"
    output_representation: ClassVar[str] = "numeric"
    bounds: list[tuple[float, float]]
    num_samples: int = Field(gt=0)
    fidelities: _Fidelities = None
    point_strategy: Literal["uniform", "lhs"] = "uniform"

    def build(self) -> Sampler:
        """Build the configured numeric sampler.

        Returns
        -------
        Sampler
            A configured :class:`~activelearning.sampler.hypercube_sampler.HypercubeSampler`.
        """
        return HypercubeSampler(
            bounds=self.bounds,
            num_samples=self.num_samples,
            fidelities=_resolve_fidelities(self.fidelities),
            point_strategy=self.point_strategy,
        )


class ExactGridSamplerConfig(BuildableConfig):
    """Configuration for exhaustive or acquisition-guided grid sampling.

    Parameters
    ----------
    bounds : list[tuple[float, float]]
        Inclusive lower and upper bounds for each numeric dimension.
    points_per_dimension : list[int]
        Number of grid points for each dimension.
    fidelities : list[int] or dict[int, float] or None
        Fidelity levels or cost mapping used to assign candidate fidelities.
    num_samples : int, optional
        Maximum number of candidates returned per sampling call.
    use_acquisition_scores : bool
        Whether to rank grid points with acquisition scores.
    with_replacement : bool
        Whether previously selected points may be sampled again.
    """

    type: Literal["ExactGridSampler"] = "ExactGridSampler"
    output_representation: ClassVar[str] = "numeric"
    bounds: list[tuple[float, float]]
    points_per_dimension: list[int]
    fidelities: _Fidelities = None
    num_samples: int | None = Field(default=None, gt=0)
    use_acquisition_scores: bool = False
    with_replacement: bool = False

    def build(self) -> Sampler:
        """Build the configured grid sampler.

        Returns
        -------
        Sampler
            A configured :class:`~activelearning.sampler.exact_grid_sampler.ExactGridSampler`.
        """
        return ExactGridSampler(
            bounds=self.bounds,
            points_per_dimension=self.points_per_dimension,
            fidelities=_resolve_fidelities(self.fidelities),
            num_samples=self.num_samples,
            use_acquisition_scores=self.use_acquisition_scores,
            with_replacement=self.with_replacement,
        )


class PoolFileSamplerConfig(BuildableConfig):
    """Configuration for :class:`~activelearning.sampler.pool_file_sampler.PoolFileSampler`.

    Parameters
    ----------
    candidate_pool_file : Path
        Path to a text file with one candidate entry per line.
    num_samples : int
        Maximum number of candidates to return per call.
    fidelities : list[int] or dict[int, float] or None
        Fidelity assignment strategy.  ``list[int]`` -> uniform random;
        ``dict[int, float]`` -> cost-inverse weighted; ``None`` ->
        :data:`~activelearning.utils.types.DEFAULT_FIDELITY` is used as the
        sole level (single-fidelity mode).
    """

    type: Literal["PoolFileSampler"] = "PoolFileSampler"
    output_representation: ClassVar[str | None] = None
    candidate_pool_file: Path
    num_samples: int = Field(gt=0)
    fidelities: _Fidelities = None

    def build(self, runtime=None) -> Sampler:
        """Build the configured pool-file sampler.

        Parameters
        ----------
        runtime : object, optional
            Runtime context accepted by the sampler configuration interface.
            The pool-file sampler does not use it.

        Returns
        -------
        Sampler
            A configured :class:`~activelearning.sampler.pool_file_sampler.PoolFileSampler`.
        """
        return PoolFileSampler(
            candidate_pool_file=self.candidate_pool_file,
            num_samples=self.num_samples,
            fidelities=_resolve_fidelities(self.fidelities),
        )


class GFlowNetSamplerConfig(BuildableConfig):
    """Pydantic config for :class:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler`.

    GFlowNet component defaults (policy, loss, buffer, evaluator, logger,
    proxy) are loaded automatically from ``config/gflownet/`` via
    :func:`~activelearning.sampler.gflownet.config_utils.compose_gflownet_conf`.
    The env config starts from ``env/base.yaml`` only - the ``_target_`` class
    and all env-specific fields must be provided in ``conf``.

    Parameters
    ----------
    type : Literal["GFlowNetSampler"]
        Discriminator field for the registered :data:`SamplerConfig` contract.
    n_samples : int
        Number of candidates to generate per sampler call.
    fidelities : list[int] or None
        Fidelity levels to generate. When ``None``, the top-level config
        validator fills this from the oracle's fidelity set.
    log_dir : str or None
        Root directory for GFlowNet logs. A temporary directory is created
        automatically when ``None``.
    conf : dict[str, Any] or None
        Experiment-specific overrides for the GFlowNet config. Deep-merged
        over the YAML defaults when provided.
    """

    type: Literal["GFlowNetSampler"] = "GFlowNetSampler"
    output_representation: ClassVar[str | None] = None
    n_samples: int = Field(gt=0)
    fidelities: _FidelityLevels | None = None
    fidelity_action: _FidelityAction = "any"
    log_dir: str | None = None
    conf: dict[str, Any] | None = None

    def build(self) -> Sampler:
        """Build the configured GFlowNet sampler.

        Returns
        -------
        Sampler
            A configured :class:`~activelearning.sampler.gflownet.gflownet_sampler.GFlowNetSampler`.
        """
        return GFlowNetSampler(
            n_samples=self.n_samples,
            conf=compose_gflownet_conf(conf_overrides=self.conf, log_dir=self.log_dir),
            fidelities=_resolve_fidelities(self.fidelities),
            fidelity_action=self.fidelity_action,
        )


class GFlowNetGridSamplerConfig(GFlowNetSamplerConfig):
    """Pydantic config for :class:`~activelearning.sampler.gflownet.grid_sampler.GFlowNetGridSampler`.

    Extends :class:`GFlowNetSamplerConfig` with the optional ``domain_bounds``
    field for per-dimension coordinate ranges.
    """

    type: Literal["GFlowNetGridSampler"] = "GFlowNetGridSampler"  # type: ignore[assignment]
    domain_bounds: list[list[float]] | None = None

    def build(self) -> Sampler:
        """Build the configured grid-specialized GFlowNet sampler.

        Returns
        -------
        Sampler
            A configured :class:`~activelearning.sampler.gflownet.grid_sampler.GFlowNetGridSampler`.
        """
        return GFlowNetGridSampler(
            n_samples=self.n_samples,
            conf=compose_gflownet_conf(conf_overrides=self.conf, log_dir=self.log_dir),
            fidelities=_resolve_fidelities(self.fidelities),
            fidelity_action=self.fidelity_action,
            domain_bounds=self.domain_bounds,
        )


SAMPLER_CONFIGS = (
    HypercubeSamplerConfig,
    ExactGridSamplerConfig,
    PoolFileSamplerConfig,
    GFlowNetSamplerConfig,
    GFlowNetGridSamplerConfig,
)
SamplerConfig = registered_config("sampler")
