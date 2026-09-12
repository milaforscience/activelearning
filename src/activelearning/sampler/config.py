"""Pydantic models of samplers.

Changes in the interface of existing samplers should be reflected in this
configuration. New samplers should define their corresponding pydantic model here and
be added to ``SamplerConfig``.
"""

from pathlib import Path
from typing import Annotated, Any, Literal, Union, overload

from pydantic import BaseModel, Field, PositiveFloat, StrictInt

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


class ExactGridSamplerConfig(BaseModel):
    type: Literal["ExactGridSampler"] = "ExactGridSampler"
    bounds: list[tuple[float, float]]
    points_per_dimension: list[int]
    fidelities: _Fidelities = None
    num_samples: int | None = Field(default=None, gt=0)
    use_acquisition_scores: bool = False
    with_replacement: bool = False

    def build(self) -> Sampler:
        return ExactGridSampler(
            bounds=self.bounds,
            points_per_dimension=self.points_per_dimension,
            fidelities=_resolve_fidelities(self.fidelities),
            num_samples=self.num_samples,
            use_acquisition_scores=self.use_acquisition_scores,
            with_replacement=self.with_replacement,
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


class S3GFNSamplerConfig(BaseModel):
    """Configuration for the soft-constrained S3-GFN molecule sampler.

    The language model generates canonical connected SMILES, which are passed
    unchanged to the active-learning loop after validation. Multiple configured
    fidelities add one terminal fidelity action; a single fidelity keeps the
    molecule-only trajectory. Install the ``molecules`` extra before using
    this sampler. ``trust_remote_code`` defaults to ``True`` because the
    default GP-MoLFormer checkpoint requires custom Hugging Face model code;
    only enable it for trusted model repositories.
    """

    type: Literal["S3GFNSampler"] = "S3GFNSampler"
    n_samples: int = Field(gt=0)
    fidelities: _FidelityLevels | None = None
    model_name_or_path: str = Field(
        default="ibm-research/GP-MoLFormer-Uniq",
        min_length=1,
    )
    tokenizer_name_or_path: str = Field(
        default="ibm-research/MoLFormer-XL-both-10pct",
        min_length=1,
    )
    trust_remote_code: bool = True
    # GP-MoLFormer redraws its linear-attention random features unless this is
    # set, and the checkpoint config defaults it to False. Keep it True so the
    # frozen prior scores a molecule identically across calls.
    deterministic_eval: bool | None = True
    cache_dir: str | None = None
    max_length: int = Field(default=140, ge=2)
    batch_size: int = Field(default=64, gt=0)
    replay_batch_size: int = Field(default=64, gt=0)
    n_train_steps: int = Field(default=5000, gt=0)
    num_warmup_steps: int = Field(default=100, ge=0)
    learning_rate: PositiveFloat = 1.0e-4
    log_z_learning_rate: PositiveFloat = 1.0e-3
    beta: PositiveFloat = 50.0
    aux_coefficient: float = Field(default=1.0e-4, ge=0.0)
    buffer_size: int = Field(default=6400, gt=0)
    sa_threshold: float = Field(default=4.0, ge=0.0)
    sampling_temperature: PositiveFloat = 1.0
    gradient_clip_norm: PositiveFloat = 10.0
    max_generation_attempts: int | None = Field(default=None, gt=0)
    seed: int = Field(default=42, ge=0)

    def build(self) -> Sampler:
        """Build the sampler lazily so base installs need no Transformers."""
        from activelearning.sampler.s3gfn.sampler import S3GFNSampler

        return S3GFNSampler(
            n_samples=self.n_samples,
            fidelities=_resolve_fidelities(self.fidelities),
            model_name_or_path=self.model_name_or_path,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            trust_remote_code=self.trust_remote_code,
            deterministic_eval=self.deterministic_eval,
            cache_dir=self.cache_dir,
            max_length=self.max_length,
            batch_size=self.batch_size,
            replay_batch_size=self.replay_batch_size,
            n_train_steps=self.n_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            learning_rate=self.learning_rate,
            log_z_learning_rate=self.log_z_learning_rate,
            beta=self.beta,
            aux_coefficient=self.aux_coefficient,
            buffer_size=self.buffer_size,
            sa_threshold=self.sa_threshold,
            sampling_temperature=self.sampling_temperature,
            gradient_clip_norm=self.gradient_clip_norm,
            max_generation_attempts=self.max_generation_attempts,
            seed=self.seed,
        )


SamplerConfig = Annotated[
    Union[
        HypercubeSamplerConfig,
        ExactGridSamplerConfig,
        PoolFileSamplerConfig,
        GFlowNetSamplerConfig,
        GFlowNetGridSamplerConfig,
        S3GFNSamplerConfig,
    ],
    Field(discriminator="type"),
]
