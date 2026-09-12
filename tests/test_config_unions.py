"""Tests for registry-backed Pydantic component configuration dispatch."""

import pytest
from pydantic import TypeAdapter, ValidationError

from activelearning.acquisition.config import AcquisitionConfig
from activelearning.dataset.config import DatasetConfig
from activelearning.logger.config import LoggerConfig
from activelearning.oracle.config import OracleConfig
from activelearning.selector.config import SelectorConfig
from activelearning.sampler.config import (
    ExactGridSamplerConfig,
    GFlowNetGridSamplerConfig,
    GFlowNetSamplerConfig,
    HypercubeSamplerConfig,
    PoolFileSamplerConfig,
    SamplerConfig,
)
from activelearning.surrogate.config import (
    BoTorchGPSurrogateConfig,
    DummyMeanSurrogateConfig,
    SurrogateConfig,
)
from activelearning.surrogate.encoder_config import (
    EncoderConfig,
    FixedEncoderConfig,
)


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
    ],
)
def test_sampler_registry_selects_config_by_type(
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
    ],
)
def test_surrogate_registry_selects_config_by_type(
    config_type: str,
    expected_type: type[object],
) -> None:
    """SurrogateConfig selects each built-in surrogate from its discriminator."""
    parsed = TypeAdapter(SurrogateConfig).validate_python({"type": config_type})

    assert isinstance(parsed, expected_type)


@pytest.mark.parametrize(
    ("config_type", "config_adapter"),
    [
        ("UnknownDataset", DatasetConfig),
        ("UnknownAcquisition", AcquisitionConfig),
        ("UnknownEncoder", EncoderConfig),
        ("UnknownFixedEncoder", FixedEncoderConfig),
        ("UnknownLogger", LoggerConfig),
        ("UnknownSampler", SamplerConfig),
        ("UnknownSelector", SelectorConfig),
        ("UnknownSurrogate", SurrogateConfig),
        ("UnknownOracle", OracleConfig),
    ],
)
def test_unknown_config_type_is_rejected(
    config_type: str,
    config_adapter: object,
) -> None:
    """Registry-backed config contracts reject unknown discriminators."""
    with pytest.raises(ValidationError, match="Unknown") as error:
        TypeAdapter(config_adapter).validate_python({"type": config_type})
    assert config_type in str(error.value)
