"""Tests for S3-GFN configuration."""

import pytest
from pydantic import TypeAdapter

from activelearning.config_registry import create_config_registry
from activelearning.sampler.config import SamplerConfig
from activelearning_molecules.config_catalogs import CONFIG_CATALOGS
from activelearning_molecules.samplers.config import S3GFNSamplerConfig


def test_s3gfn_config_defaults_enable_optimized_stack() -> None:
    config = S3GFNSamplerConfig(n_samples=4)

    assert config.beta == 50.0
    assert config.aux_coefficient == 1.0e-4
    assert config.deterministic_eval is True
    assert config.performance_mode == "optimized"
    assert config.compile_strategy == "training_and_generation"
    assert config.torch_compile_mode == "default"
    assert config.torch_compile_dynamic is None
    assert config.attention_mask_adapter is True
    assert config.compile_prior_scorer is True
    assert config.model_dtype == "bfloat16"
    assert config.generation_batch_size is None


def test_s3gfn_config_eager_mode_disables_optimizations() -> None:
    config = S3GFNSamplerConfig(n_samples=4, performance_mode="eager")

    assert config.compile_strategy == "none"
    assert config.torch_compile_mode == "default"
    assert config.torch_compile_dynamic is True
    assert config.attention_mask_adapter is False
    assert config.compile_prior_scorer is False
    assert config.model_dtype == "float32"


def test_s3gfn_config_explicit_performance_fields_override_preset() -> None:
    config = S3GFNSamplerConfig(
        n_samples=4,
        performance_mode="eager",
        compile_strategy="training_and_generation",
        torch_compile_dynamic=None,
        attention_mask_adapter=True,
        compile_prior_scorer=True,
        model_dtype="bfloat16",
    )

    assert config.compile_strategy == "training_and_generation"
    assert config.torch_compile_dynamic is None
    assert config.attention_mask_adapter is True
    assert config.compile_prior_scorer is True
    assert config.model_dtype == "bfloat16"


def test_s3gfn_config_round_trips_through_sampler_union() -> None:
    config = TypeAdapter(SamplerConfig).validate_python(
        {
            "type": "S3GFNSampler",
            "n_samples": 4,
            "fidelities": [1, 2],
            "max_generation_attempts": 128,
            "performance_mode": "optimized",
            "compile_strategy": "training_only",
            "torch_compile_mode": "default",
            "torch_compile_dynamic": None,
            "attention_mask_adapter": True,
            "compile_prior_scorer": True,
            "model_dtype": "bfloat16",
            "generation_batch_size": 128,
        },
        context={
            "config_registry": create_config_registry(
                {"activelearning-molecules": CONFIG_CATALOGS}
            )
        },
    )

    assert isinstance(config, S3GFNSamplerConfig)
    assert config.fidelities == [1, 2]
    assert config.max_generation_attempts == 128
    assert config.performance_mode == "optimized"
    assert config.compile_strategy == "training_only"
    assert config.torch_compile_mode == "default"
    assert config.torch_compile_dynamic is None
    assert config.attention_mask_adapter is True
    assert config.compile_prior_scorer is True
    assert config.model_dtype == "bfloat16"
    assert config.generation_batch_size == 128


@pytest.mark.parametrize(
    "field", ["model_dtype", "compile_strategy", "performance_mode"]
)
def test_s3gfn_config_rejects_unsupported_enum_values(field: str) -> None:
    with pytest.raises(ValueError):
        TypeAdapter(SamplerConfig).validate_python(
            {
                "type": "S3GFNSampler",
                "n_samples": 4,
                field: "unsupported",
            }
        )


def test_s3gfn_config_rejects_nonpositive_generation_batch_size() -> None:
    with pytest.raises(ValueError):
        S3GFNSamplerConfig(n_samples=4, generation_batch_size=0)
