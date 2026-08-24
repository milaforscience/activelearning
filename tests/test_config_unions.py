"""Tests for explicit Pydantic component configuration unions."""

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from activelearning.applications.molecules.config import (
    EncoderConfig,
    GPMoLFormerSmilesEncoderConfig,
    MiniMolSmilesEncoderConfig,
    MoLFormerSmilesEncoderConfig,
    SelfiesTransformerEncoderConfig,
)
from activelearning.sampler.config import (
    ExactGridSamplerConfig,
    GFlowNetGridSamplerConfig,
    GFlowNetSamplerConfig,
    HypercubeSamplerConfig,
    PoolFileSamplerConfig,
    S3GFNSamplerConfig,
    SamplerConfig,
)
from activelearning.surrogate.config import (
    BoTorchGPSurrogateConfig,
    DummyMeanSurrogateConfig,
    SurrogateConfig,
)
from activelearning.surrogate.dkl.config import (
    ExactDKLSurrogateConfig,
    VariationalDKLSurrogateConfig,
)


@pytest.mark.parametrize(
    ("config_type", "expected_type"),
    [
        ("SelfiesTransformerEncoder", SelfiesTransformerEncoderConfig),
        ("GPMoLFormerSmilesEncoder", GPMoLFormerSmilesEncoderConfig),
        ("MoLFormerSmilesEncoder", MoLFormerSmilesEncoderConfig),
        ("MiniMolSmilesEncoder", MiniMolSmilesEncoderConfig),
    ],
)
def test_encoder_union_selects_config_by_type(
    config_type: str,
    expected_type: type[object],
) -> None:
    """EncoderConfig selects each built-in encoder from its discriminator."""
    parsed = TypeAdapter(EncoderConfig).validate_python({"type": config_type})

    assert isinstance(parsed, expected_type)


@pytest.mark.parametrize(
    ("config_type", "config_data", "expected_type"),
    [
        (
            "HypercubeSampler",
            {"bounds": [[0.0, 1.0]], "num_samples": 2},
            HypercubeSamplerConfig,
        ),
        (
            "ExactGridSampler",
            {"bounds": [[0.0, 1.0]], "points_per_dimension": [2]},
            ExactGridSamplerConfig,
        ),
        (
            "PoolFileSampler",
            {"candidate_pool_file": "pool.txt", "num_samples": 2},
            PoolFileSamplerConfig,
        ),
        ("GFlowNetSampler", {"n_samples": 2}, GFlowNetSamplerConfig),
        ("GFlowNetGridSampler", {"n_samples": 2}, GFlowNetGridSamplerConfig),
        ("S3GFNSampler", {"n_samples": 2}, S3GFNSamplerConfig),
    ],
)
def test_sampler_union_selects_config_by_type(
    config_type: str,
    config_data: dict[str, object],
    expected_type: type[object],
) -> None:
    """SamplerConfig selects each built-in sampler from its discriminator."""
    parsed = TypeAdapter(SamplerConfig).validate_python(
        {"type": config_type, **config_data}
    )

    assert isinstance(parsed, expected_type)


@pytest.mark.parametrize(
    ("config_type", "expected_type"),
    [
        ("DummyMeanSurrogate", DummyMeanSurrogateConfig),
        ("BoTorchGPSurrogate", BoTorchGPSurrogateConfig),
        ("ExactDKLSurrogate", ExactDKLSurrogateConfig),
        ("VariationalDKLSurrogate", VariationalDKLSurrogateConfig),
    ],
)
def test_surrogate_union_selects_config_by_type(
    config_type: str,
    expected_type: type[object],
) -> None:
    """SurrogateConfig selects each built-in surrogate from its discriminator."""
    data: dict[str, object] = {"type": config_type}
    if config_type in {"ExactDKLSurrogate", "VariationalDKLSurrogate"}:
        data["encoder"] = {"type": "SelfiesTransformerEncoder"}

    parsed = TypeAdapter(SurrogateConfig).validate_python(data)

    assert isinstance(parsed, expected_type)


def test_dkl_config_parses_nested_encoder_union() -> None:
    """DKL configs validate nested encoder mappings through EncoderConfig."""
    config = ExactDKLSurrogateConfig.model_validate(
        {
            "encoder": {"type": "MoLFormerSmilesEncoder", "latent_dim": 32},
        }
    )

    assert isinstance(config.encoder, MoLFormerSmilesEncoderConfig)
    assert config.encoder.latent_dim == 32


def test_minimol_encoder_config_parses_checkpoint_path() -> None:
    """MiniMol configs preserve a custom predictor checkpoint path."""
    config = MiniMolSmilesEncoderConfig.model_validate(
        {
            "type": "MiniMolSmilesEncoder",
            "checkpoint_path": "weights/minimol-finetuned.pth",
        }
    )

    assert config.checkpoint_path == Path("weights/minimol-finetuned.pth")


def test_minimol_encoder_config_defaults_to_32_latent_features() -> None:
    """MiniMol uses the compact projected representation by default."""
    config = MiniMolSmilesEncoderConfig()

    assert config.latent_dim == 32


@pytest.mark.parametrize(
    ("config_type", "config_adapter"),
    [
        ("UnknownEncoder", EncoderConfig),
        ("UnknownSampler", SamplerConfig),
        ("UnknownSurrogate", SurrogateConfig),
    ],
)
def test_unknown_config_type_is_rejected(
    config_type: str,
    config_adapter: object,
) -> None:
    """Static unions reject discriminators outside the built-in catalog."""
    with pytest.raises(ValidationError, match="Input tag"):
        TypeAdapter(config_adapter).validate_python({"type": config_type})


def test_variational_dkl_config_parses_nested_encoder_union() -> None:
    """Variational DKL configs use the same nested encoder contract."""
    config = VariationalDKLSurrogateConfig.model_validate(
        {
            "encoder": {"type": "SelfiesTransformerEncoder"},
            "num_inducing": 8,
        }
    )

    assert isinstance(config.encoder, SelfiesTransformerEncoderConfig)
    assert config.num_inducing == 8
