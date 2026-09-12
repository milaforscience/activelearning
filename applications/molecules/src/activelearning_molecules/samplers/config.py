"""Configuration models for molecular samplers."""

from typing import Any, ClassVar, Literal

from pydantic import Field, PositiveFloat, model_validator

from activelearning.config_registry import BuildableConfig
from activelearning.sampler.config import FidelityLevels, resolve_fidelities
from activelearning.sampler.sampler import Sampler


class S3GFNSamplerConfig(BuildableConfig):
    """Configuration for the acquisition-guided S3-GFN molecule sampler."""

    type: Literal["S3GFNSampler"] = "S3GFNSampler"
    output_representation: ClassVar[str] = "smiles"
    n_samples: int = Field(gt=0)
    fidelities: FidelityLevels | None = None
    model_name_or_path: str = Field(
        default="ibm-research/GP-MoLFormer-Uniq",
        min_length=1,
    )
    tokenizer_name_or_path: str = Field(
        default="ibm-research/MoLFormer-XL-both-10pct",
        min_length=1,
    )
    trust_remote_code: bool = True
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
        """Fill omitted low-level performance fields from the selected preset."""
        if not isinstance(data, dict):
            return data

        values = dict(data)
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
        }.get(values.get("performance_mode", "optimized"))
        if preset is not None:
            for field_name, default in preset.items():
                values.setdefault(field_name, default)
        return values

    def build(self) -> Sampler:
        """Build the molecular sampler lazily."""
        from activelearning_molecules.samplers.s3gfn.sampler import S3GFNSampler

        return S3GFNSampler(
            n_samples=self.n_samples,
            fidelities=resolve_fidelities(self.fidelities),
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


MOLECULE_SAMPLER_CONFIGS = (S3GFNSamplerConfig,)

__all__ = ["MOLECULE_SAMPLER_CONFIGS", "S3GFNSamplerConfig"]
