from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field
from activelearning.sampler.hypercube_sampler import HypercubeSampler
from activelearning.sampler.sampler import Sampler
from activelearning.sampler.gflownet.config_utils import compose_gflownet_conf
from activelearning.sampler.gflownet.grid_sampler import GFlowNetGridSampler
from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler


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
    n_fidelities : int
        Number of fidelity levels.  ``1`` means single-fidelity.
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
    n_fidelities: int = 1
    log_dir: str | None = None
    conf: dict[str, Any] | None = None

    def build(self) -> Sampler:
        return GFlowNetSampler(
            n_samples=self.n_samples,
            conf=compose_gflownet_conf(conf_overrides=self.conf, log_dir=self.log_dir),
            n_fidelities=self.n_fidelities,
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
            n_fidelities=self.n_fidelities,
        )


SamplerConfig = Annotated[
    Union[HypercubeSamplerConfig, GFlowNetSamplerConfig, GFlowNetGridSamplerConfig],
    Field(discriminator="type"),
]
