from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field
from activelearning.sampler.hypercube_sampler import HypercubeSampler
from activelearning.sampler.sampler import Sampler
from activelearning.sampler.gflownet.config_utils import compose_gflownet_conf
from activelearning.sampler.gflownet.grid_sampler import GFlowNetGridSampler
from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler

_FidelityAction = Literal["any", "first", "last"]


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
    n_fidelities : int
        Number of fidelity levels. ``1`` means single-fidelity.
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
    n_fidelities: int = Field(default=1, ge=1)
    fidelity_action: _FidelityAction = "any"
    log_dir: str | None = None
    conf: dict[str, Any] | None = None

    def build(self) -> Sampler:
        return GFlowNetSampler(
            n_samples=self.n_samples,
            conf=compose_gflownet_conf(conf_overrides=self.conf, log_dir=self.log_dir),
            n_fidelities=self.n_fidelities,
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
            n_fidelities=self.n_fidelities,
            fidelity_action=self.fidelity_action,
            domain_bounds=self.domain_bounds,
        )


SamplerConfig = Annotated[
    Union[HypercubeSamplerConfig, GFlowNetSamplerConfig, GFlowNetGridSamplerConfig],
    Field(discriminator="type"),
]
