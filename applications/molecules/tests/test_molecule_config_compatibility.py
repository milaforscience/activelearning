"""Tests for cross-component active-learning configuration validation."""

from collections.abc import Callable
from pathlib import Path
import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from activelearning.config import ActiveLearningConfig
from activelearning.utils.config_loader import load_config, parse_config
from activelearning_molecules.config_catalogs import CONFIG_CATALOGS


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _parse_mutated_config(
    relative_path: str,
    mutate: Callable[[DictConfig], None],
) -> ActiveLearningConfig:
    """Load an example configuration, apply a test mutation, and parse it."""
    config = load_config(REPOSITORY_ROOT / relative_path)
    mutate(config)
    return parse_config(
        config,
        ActiveLearningConfig,
        catalogs={"activelearning-molecules": CONFIG_CATALOGS},
    )


def _set_dkl_encoder(
    config: DictConfig,
    surrogate_type: str,
    encoder_type: str,
) -> None:
    """Replace the DKL surrogate and encoder discriminators in a config."""
    config.surrogate.type = surrogate_type
    config.surrogate.encoder = OmegaConf.create({"type": encoder_type})


def test_s3gfn_rejects_selfies_encoder() -> None:
    """S3-GFN must not feed generated SMILES to a SELFIES encoder."""
    with pytest.raises(
        ValidationError,
        match=(
            "S3GFNSamplerConfig produces representation 'smiles'.*"
            "SelfiesTransformerEncoderConfig expects 'selfies'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/s3gfn_exact.yaml",
            lambda config: _set_dkl_encoder(
                config,
                "ExactDKLSurrogate",
                "SelfiesTransformerEncoder",
            ),
        )


def test_s3gfn_rejects_selfies_oracle() -> None:
    """S3-GFN must be paired with an oracle that interprets SMILES."""
    with pytest.raises(
        ValidationError,
        match=(
            "S3GFNSamplerConfig produces representation 'smiles'.*"
            "XTBIPEAOracleConfig expects 'selfies'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/s3gfn_exact.yaml",
            lambda config: setattr(config.oracle, "mol_repr", "selfies"),
        )


def test_s3gfn_rejects_non_molecular_oracle() -> None:
    """S3-GFN output must match the oracle's numeric input declaration."""
    with pytest.raises(
        ValidationError,
        match=(
            "S3GFNSamplerConfig produces representation 'smiles'.*"
            "BraninOracleConfig expects 'numeric'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/s3gfn_exact.yaml",
            lambda config: setattr(
                config,
                "oracle",
                OmegaConf.create(
                    {
                        "type": "BraninOracle",
                        "fidelity_costs": {1: 1.0},
                    }
                ),
            ),
        )


def test_selfies_encoder_rejects_smiles_oracle() -> None:
    """A SELFIES encoder must match the oracle's declared representation."""
    with pytest.raises(
        ValidationError,
        match=(
            "SelfiesTransformerEncoderConfig produces representation 'selfies'.*"
            "XTBIPEAOracleConfig expects 'smiles'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/exact.yaml",
            lambda config: setattr(config.oracle, "mol_repr", "smiles"),
        )


def test_smiles_encoder_rejects_selfies_oracle() -> None:
    """A SMILES encoder must match the oracle's declared representation."""
    with pytest.raises(
        ValidationError,
        match=(
            "GPMoLFormerSmilesEncoderConfig produces representation 'smiles'.*"
            "XTBIPEAOracleConfig expects 'selfies'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/exact.yaml",
            lambda config: _set_dkl_encoder(
                config,
                "ExactDKLSurrogate",
                "GPMoLFormerSmilesEncoder",
            ),
        )


def test_molformer_smiles_encoder_config_parses() -> None:
    """The encoder-style MoLFormer is selectable as a SMILES DKL encoder."""
    config = _parse_mutated_config(
        "applications/molecules/config/s3gfn_exact.yaml",
        lambda config: _set_dkl_encoder(
            config,
            "ExactDKLSurrogate",
            "MoLFormerSmilesEncoder",
        ),
    )

    assert config.surrogate.encoder.type == "MoLFormerSmilesEncoder"
    assert config.surrogate.encoder.pooling == "mean"


def test_dkl_encoder_rejects_non_molecular_oracle() -> None:
    """Molecular DKL encoders require an oracle with representation metadata."""
    with pytest.raises(
        ValidationError,
        match=(
            "SelfiesTransformerEncoderConfig produces representation 'selfies'.*"
            "BraninOracleConfig expects 'numeric'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/exact.yaml",
            lambda config: setattr(
                config,
                "oracle",
                OmegaConf.create(
                    {
                        "type": "BraninOracle",
                        "fidelity_costs": {1: 1.0},
                    }
                ),
            ),
        )


def test_numeric_sampler_rejects_molecular_dkl_encoder() -> None:
    """Known numeric samplers must not feed coordinates to molecular encoders."""
    with pytest.raises(
        ValidationError,
        match=(
            "HypercubeSamplerConfig produces representation 'numeric'.*"
            "SelfiesTransformerEncoderConfig expects 'selfies'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/exact.yaml",
            lambda config: setattr(
                config,
                "sampler",
                OmegaConf.create(
                    {
                        "type": "HypercubeSampler",
                        "bounds": [[0.0, 1.0]],
                        "num_samples": 4,
                        "fidelities": [1],
                    }
                ),
            ),
        )


def test_numeric_sampler_rejects_molecular_oracle() -> None:
    """Known numeric samplers must not feed coordinates to XTBIPEA."""
    with pytest.raises(
        ValidationError,
        match=(
            "HypercubeSamplerConfig produces representation 'numeric'.*"
            "XTBIPEAOracleConfig expects 'selfies'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/exact.yaml",
            lambda config: (
                setattr(
                    config,
                    "sampler",
                    OmegaConf.create(
                        {
                            "type": "HypercubeSampler",
                            "bounds": [[0.0, 1.0]],
                            "num_samples": 4,
                            "fidelities": [1],
                        }
                    ),
                ),
                setattr(
                    config,
                    "surrogate",
                    OmegaConf.create({"type": "DummyMeanSurrogate"}),
                ),
                setattr(
                    config,
                    "acquisition",
                    OmegaConf.create({"type": "DummyAcquisition"}),
                ),
            ),
        )


def test_botorch_acquisition_rejects_dummy_surrogate() -> None:
    """BoTorch acquisitions require a BoTorch-compatible surrogate."""
    with pytest.raises(
        ValidationError,
        match="UpperConfidenceBoundConfig requires a BoTorch-compatible surrogate",
    ):
        _parse_mutated_config(
            "applications/molecules/config/exact.yaml",
            lambda config: setattr(
                config,
                "surrogate",
                OmegaConf.create({"type": "DummyMeanSurrogate"}),
            ),
        )


def test_generic_botorch_surrogate_rejects_s3gfn_strings() -> None:
    """Generic BoTorch surrogates must not consume S3-GFN SMILES directly."""
    with pytest.raises(
        ValidationError,
        match=(
            "S3GFNSamplerConfig produces representation 'smiles'.*"
            "BoTorchGPSurrogateConfig expects 'numeric'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/s3gfn_exact.yaml",
            lambda config: setattr(
                config,
                "surrogate",
                OmegaConf.create({"type": "BoTorchGPSurrogate"}),
            ),
        )


def test_generic_botorch_surrogate_rejects_molecular_pool_inputs() -> None:
    """Numeric surrogates must reject molecular inputs from opaque pools."""
    with pytest.raises(
        ValidationError,
        match=(
            "BoTorchGPSurrogateConfig expects representation 'numeric'.*"
            "XTBIPEAOracleConfig expects 'selfies'"
        ),
    ):
        _parse_mutated_config(
            "applications/molecules/config/exact.yaml",
            lambda config: setattr(
                config,
                "surrogate",
                OmegaConf.create({"type": "BoTorchGPSurrogate"}),
            ),
        )
