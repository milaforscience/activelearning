"""Tests for S3-GFN configuration."""

from pydantic import TypeAdapter

from activelearning.sampler.config import S3GFNSamplerConfig, SamplerConfig


def test_s3gfn_config_defaults_match_upstream_training_defaults() -> None:
    config = S3GFNSamplerConfig(n_samples=4)

    assert config.beta == 50.0
    assert config.aux_coefficient == 1.0e-4
    assert config.deterministic_eval is True


def test_s3gfn_config_round_trips_through_sampler_union() -> None:
    config = TypeAdapter(SamplerConfig).validate_python(
        {
            "type": "S3GFNSampler",
            "n_samples": 4,
            "fidelities": [1, 2],
            "max_generation_attempts": 128,
        }
    )

    assert isinstance(config, S3GFNSamplerConfig)
    assert config.fidelities == [1, 2]
    assert config.max_generation_attempts == 128
