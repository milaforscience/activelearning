"""Pydantic models of samplers.

Changes in the interface of existing samplers should be reflected in this
configuration. New built-in samplers should be added to the explicit
discriminated union below.
"""

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Union, overload

from pydantic import BaseModel, Field, PositiveFloat, StrictInt, model_validator

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


class ExactGridSamplerConfig(BaseModel):
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


class S3GFNSamplerConfig(BaseModel):
    """Configuration for the soft-constrained S3-GFN molecule sampler.

    The language model generates canonical connected SMILES, which are passed
    unchanged to the active-learning loop after validation. Multiple configured
    fidelities add one terminal fidelity action; a single fidelity keeps the
    molecule-only trajectory. Install the ``molecules`` extra before using
    this sampler. ``trust_remote_code`` defaults to ``True`` because the
    default GP-MoLFormer checkpoint requires custom Hugging Face model code;
    only enable it for trusted model repositories.

    Parameters
    ----------
    n_samples : int
    Number of candidates generated per sampling call.
    fidelities : list[int], optional
    Fidelity levels assigned to generated candidates.
    model_name_or_path : str, default="ibm-research/GP-MoLFormer-Uniq"
    Hugging Face policy model identifier or local path.
    tokenizer_name_or_path : str, default="ibm-research/MoLFormer-XL-both-10pct"
    Hugging Face tokenizer identifier or local path.
    trust_remote_code : bool, default=True
    Whether loading may execute repository-provided model code.
    deterministic_eval : bool, optional
    Whether policy evaluation uses deterministic random features.
    performance_mode : {"optimized", "eager"}, default="optimized"
    Performance preset for the S3-GFN model. The optimized preset enables the
    validated BF16 and compilation stack. Use ``"eager"`` to opt out. Explicit
    low-level performance fields override the selected preset.
    compile_strategy : {"none", "training_only", "training_and_generation"}, default="training_and_generation"
    Select eager execution, policy compilation during training only, or policy
    compilation during training and final generation.
    torch_compile_mode : str, default="default"
    TorchInductor mode used when compilation is enabled.
    torch_compile_dynamic : bool or None, default=None
    Dynamic-shape policy passed to :func:`torch.compile`.
    attention_mask_adapter : bool, default=True
    Enable the pinned GP-MoLFormer attention-mask compile adapter.
    compile_prior_scorer : bool, default=True
    Compile frozen-prior sequence scoring as a separate no-grad graph.
    model_dtype : {"float32", "bfloat16"}, default="bfloat16"
    Floating-point dtype for the S3-GFN model and loss tensors.
    cache_dir : str, optional
    Directory for Hugging Face model and tokenizer files.
    max_length : int, default=140
    Maximum generated sequence length.
    batch_size : int, default=64
    Number of trajectories generated during each training step.
    replay_batch_size : int, default=64
    Replay-buffer batch size used during training.
    generation_batch_size : int, optional
    Number of trajectories generated per final candidate-generation call.
    When omitted, ``batch_size`` is used.
    n_train_steps : int, default=5000
    Number of policy training steps.
    num_warmup_steps : int, default=100
    Number of learning-rate warm-up steps.
    learning_rate : float, default=1e-4
    Policy learning rate.
    log_z_learning_rate : float, default=1e-3
    Learning rate for the log-partition estimate.
    beta : float, default=50.0
    GFlowNet loss temperature parameter.
    aux_coefficient : float, default=1e-4
    Weight of the auxiliary loss.
    buffer_size : int, default=6400
    Maximum replay-buffer size.
    sa_threshold : float, default=4.0
    Synthetic-accessibility threshold.
    sampling_temperature : float, default=1.0
    Sampling temperature applied during generation.
    gradient_clip_norm : float, default=10.0
    Maximum gradient norm during policy training.
    max_generation_attempts : int, optional
    Maximum attempts to produce the requested number of valid candidates.
    seed : int, default=42
    Random seed for policy training and sampling.
    """

    type: Literal["S3GFNSampler"] = "S3GFNSampler"
    output_representation: ClassVar[str] = "smiles"
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
    performance_mode: Literal["optimized", "eager"] = "optimized"
    compile_strategy: Literal["none", "training_only", "training_and_generation"] = (
        "training_and_generation"
    )
    torch_compile_mode: str = "default"
    torch_compile_dynamic: bool | None = None
    attention_mask_adapter: bool = True
    compile_prior_scorer: bool = True
    model_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    cache_dir: str | None = None
    max_length: int = Field(default=140, ge=2)
    batch_size: int = Field(default=64, gt=0)
    replay_batch_size: int = Field(default=64, gt=0)
    generation_batch_size: int | None = Field(default=None, gt=0)
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

    @model_validator(mode="before")
    @classmethod
    def _apply_performance_mode(cls, data: Any) -> Any:
        """Fill omitted performance fields from the selected preset."""
        if not isinstance(data, dict):
            return data

        values = dict(data)
        mode = values.get("performance_mode", "optimized")
        preset = {
            "optimized": {
                "compile_strategy": "training_and_generation",
                "torch_compile_mode": "default",
                "torch_compile_dynamic": None,
                "attention_mask_adapter": True,
                "compile_prior_scorer": True,
                "model_dtype": "bfloat16",
            },
            "eager": {
                "compile_strategy": "none",
                "torch_compile_mode": "default",
                "torch_compile_dynamic": True,
                "attention_mask_adapter": False,
                "compile_prior_scorer": False,
                "model_dtype": "float32",
            },
        }.get(mode)
        if preset is not None:
            for field_name, default in preset.items():
                values.setdefault(field_name, default)
        return values

    def build(self) -> Sampler:
        """Build the sampler lazily so base installs need no Transformers.

        Returns
        -------
        Sampler
            A configured :class:`~activelearning.sampler.s3gfn.sampler.S3GFNSampler`.

        Raises
        ------
        ImportError
            If the optional molecules dependencies are not installed.
        """
        from activelearning.sampler.s3gfn.sampler import S3GFNSampler

        return S3GFNSampler(
            n_samples=self.n_samples,
            fidelities=_resolve_fidelities(self.fidelities),
            model_name_or_path=self.model_name_or_path,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            trust_remote_code=self.trust_remote_code,
            deterministic_eval=self.deterministic_eval,
            compile_strategy=self.compile_strategy,
            torch_compile_mode=self.torch_compile_mode,
            torch_compile_dynamic=self.torch_compile_dynamic,
            attention_mask_adapter=self.attention_mask_adapter,
            compile_prior_scorer=self.compile_prior_scorer,
            model_dtype=self.model_dtype,
            cache_dir=self.cache_dir,
            max_length=self.max_length,
            batch_size=self.batch_size,
            replay_batch_size=self.replay_batch_size,
            generation_batch_size=self.generation_batch_size,
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
