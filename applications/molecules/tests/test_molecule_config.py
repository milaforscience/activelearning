"""Configuration tests for the reusable molecular application."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import TypeAdapter, ValidationError

from activelearning.config import ActiveLearningConfig
from activelearning.config_registry import create_config_registry
from activelearning.sampler.config import SamplerConfig
from activelearning.surrogate.dkl.config import ExactDKLSurrogateConfig
from activelearning.surrogate.encoder_config import EncoderConfig
from activelearning.utils.config_loader import load_config, parse_config
from activelearning_molecules import main as molecule_main
from activelearning_molecules.config_catalogs import CONFIG_CATALOGS
from activelearning_molecules.encoders.config import (
    GPMoLFormerSmilesEncoderConfig,
    MiniMolSmilesEncoderConfig,
    MoLFormerSmilesEncoderConfig,
    SelfiesTransformerEncoderConfig,
)
from activelearning_molecules.samplers.config import S3GFNSamplerConfig


def _registry():
    """Return a registry populated with the molecular application."""
    return create_config_registry({"activelearning-molecules": CONFIG_CATALOGS})


def test_encoder_registry_selects_molecular_config_by_type() -> None:
    """The application registers every retained molecular encoder schema."""
    registry = _registry()
    expected = {
        "SelfiesTransformerEncoder": SelfiesTransformerEncoderConfig,
        "GPMoLFormerSmilesEncoder": GPMoLFormerSmilesEncoderConfig,
        "MoLFormerSmilesEncoder": MoLFormerSmilesEncoderConfig,
        "MiniMolSmilesEncoder": MiniMolSmilesEncoderConfig,
    }

    for type_name, expected_type in expected.items():
        parsed = TypeAdapter(EncoderConfig).validate_python(
            {"type": type_name},
            context={"config_registry": registry},
        )
        assert isinstance(parsed, expected_type)


def test_dkl_config_parses_nested_molecular_encoder() -> None:
    """DKL retains the concrete application encoder in its nested field."""
    config = ExactDKLSurrogateConfig.model_validate(
        {
            "encoder": {
                "type": "MoLFormerSmilesEncoder",
                "latent_dim": 32,
            },
        },
        context={"config_registry": _registry()},
    )

    assert isinstance(config.encoder, MoLFormerSmilesEncoderConfig)
    assert config.encoder.latent_dim == 32


def test_minimol_encoder_config_preserves_checkpoint_path() -> None:
    """MiniMol configs preserve a custom predictor checkpoint path."""
    config = MiniMolSmilesEncoderConfig.model_validate(
        {
            "type": "MiniMolSmilesEncoder",
            "checkpoint_path": "weights/minimol-finetuned.pth",
        }
    )

    assert config.checkpoint_path == Path("weights/minimol-finetuned.pth")
    assert config.latent_dim == 32


def test_s3gfn_sampler_is_registered_by_the_application() -> None:
    """The S3-GFN schema is available through the shared sampler contract."""
    config = TypeAdapter(SamplerConfig).validate_python(
        {"type": "S3GFNSampler", "n_samples": 2},
        context={"config_registry": _registry()},
    )

    assert isinstance(config, S3GFNSamplerConfig)


def test_core_loader_does_not_activate_molecular_catalogs() -> None:
    """The generic loader remains isolated from application packages."""
    config = load_config(Path("applications/molecules/config/s3gfn_exact.yaml"))

    with pytest.raises(ValidationError, match="Unknown sampler configuration type"):
        parse_config(config, ActiveLearningConfig)


def test_molecular_cli_supplies_application_catalogs() -> None:
    """The molecular command owns application composition."""
    with patch.object(molecule_main, "run") as run:
        molecule_main.main(["config.yaml"])

    run.assert_called_once_with(
        ["config.yaml"],
        catalogs={"activelearning-molecules": CONFIG_CATALOGS},
        program_name="activelearning-molecules",
    )
