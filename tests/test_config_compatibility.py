"""Tests for domain-neutral configuration compatibility rules."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from activelearning.config import ActiveLearningConfig
from activelearning.utils.config_loader import load_config, parse_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_botorch_acquisition_rejects_dummy_surrogate() -> None:
    """BoTorch acquisitions require a BoTorch-compatible surrogate."""
    config = load_config(REPOSITORY_ROOT / "config" / "branin" / "single_fidelity.yaml")
    config.surrogate = OmegaConf.create({"type": "DummyMeanSurrogate"})

    with pytest.raises(
        ValidationError,
        match=(
            "QMultiFidelityLowerBoundMaxValueEntropyConfig requires a "
            "BoTorch-compatible surrogate"
        ),
    ):
        parse_config(config, ActiveLearningConfig)
