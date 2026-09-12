"""Tests for explicit Pydantic component configuration unions."""

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from activelearning.surrogate.encoder_config import (
    EncoderConfig,
    FixedEncoderConfig,
    GPMoLFormerSmilesEncoderConfig,
    MiniMolAmpcSmilesFixedEncoderConfig,
    MiniMolAmpcSmilesEncoderConfig,
    MiniMolSmilesFixedEncoderConfig,
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
    VariationalGPSurrogateConfig,
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
        ("VariationalGPSurrogate", VariationalGPSurrogateConfig),
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
    elif config_type == "VariationalGPSurrogate":
        data["encoder"] = {"type": "MiniMolSmilesFixedEncoder"}

    parsed = TypeAdapter(SurrogateConfig).validate_python(data)

    assert isinstance(parsed, expected_type)


@pytest.mark.parametrize(
    ("config_type", "config_data", "expected_type"),
    [
        (
            "MiniMolSmilesFixedEncoder",
            {},
            MiniMolSmilesFixedEncoderConfig,
        ),
        (
            "MiniMolAmpcSmilesFixedEncoder",
            {"checkpoint_path": "minimol_resources/model/final.pt"},
            MiniMolAmpcSmilesFixedEncoderConfig,
        ),
    ],
)
def test_fixed_encoder_union_selects_config_by_type(
    config_type: str,
    config_data: dict[str, object],
    expected_type: type[object],
) -> None:
    """FixedEncoderConfig dispatches each fixed MiniMol implementation."""
    parsed = TypeAdapter(FixedEncoderConfig).validate_python(
        {"type": config_type, **config_data}
    )

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


def test_minimol_ampc_encoder_config_parses_checkpoint_and_package_paths() -> None:
    """The full-trunk MiniMol config preserves its local package paths."""
    config = MiniMolAmpcSmilesEncoderConfig.model_validate(
        {
            "type": "MiniMolAmpcSmilesEncoder",
            "checkpoint_path": "minimol_resources/model/final.pt",
            "package_path": "minimol_resources",
            "device": "cpu",
        }
    )

    assert config.checkpoint_path == Path("minimol_resources/model/final.pt")
    assert config.package_path == Path("minimol_resources")
    assert config.device == "cpu"


def test_encoder_union_selects_minimol_ampc_config() -> None:
    """EncoderConfig dispatches the full-trunk MiniMol discriminator."""
    parsed = TypeAdapter(EncoderConfig).validate_python(
        {
            "type": "MiniMolAmpcSmilesEncoder",
            "checkpoint_path": "minimol_resources/model/final.pt",
        }
    )

    assert isinstance(parsed, MiniMolAmpcSmilesEncoderConfig)


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


def test_variational_gp_config_parses_fixed_encoder() -> None:
    """The fixed-feature GP config parses its nested encoder union."""
    config = VariationalGPSurrogateConfig.model_validate(
        {
            "encoder": {
                "type": "MiniMolAmpcSmilesFixedEncoder",
                "checkpoint_path": "minimol_resources/model/final.pt",
            },
            "num_inducing": 8,
        }
    )

    assert isinstance(
        config.encoder,
        MiniMolAmpcSmilesFixedEncoderConfig,
    )
    assert config.num_inducing == 8


def test_variational_gp_config_resolves_multi_fidelity_target() -> None:
    """The fixed-feature GP derives fidelity mode and target from the oracle."""
    config = VariationalGPSurrogateConfig.model_validate(
        {
            "encoder": {
                "type": "MiniMolSmilesFixedEncoder",
            },
        }
    )

    resolved = config.resolve_fidelity_confidences({1: 0.25, 3: 1.0})

    assert resolved.is_multi_fidelity is True
    assert resolved.target_fidelity == 3
