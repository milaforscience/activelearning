"""Tests for configuration catalogs and registry dispatch."""

from typing import ClassVar, Literal

import pytest
from pydantic import TypeAdapter, ValidationError

from activelearning.acquisition.config import AcquisitionConfig
from activelearning.config import _validate_component_compatibility
from activelearning.config_registry import (
    ConfigRegistry,
    create_config_registry,
)
from activelearning.dataset.config import DatasetConfig
from activelearning.logger.config import LoggerConfig, MultiLoggerConfig
from activelearning.oracle.config import CompositeOracleConfig, OracleConfig
from activelearning.sampler.config import SamplerConfig
from activelearning.selector.config import SelectorConfig
from activelearning.surrogate.config import SurrogateConfig
from activelearning.surrogate.dkl.config import ExactDKLSurrogateConfig
from activelearning.surrogate.encoder_config import EncoderConfig, FixedEncoderConfig


def _buildable_model(type_name: str) -> type:
    """Create a minimal buildable config model for registry tests."""
    from activelearning.config_registry import BuildableConfig

    return type(
        f"{type_name}Config",
        (BuildableConfig,),
        {
            "__annotations__": {"type": Literal[type_name]},
            "type": type_name,
            "build": lambda self: object(),
        },
    )


def test_core_registry_parses_builtin_components_and_is_isolated() -> None:
    """Core-only parsing works without discovering application packages."""
    registry = create_config_registry()

    sampler = TypeAdapter(SamplerConfig).validate_python(
        {"type": "HypercubeSampler", "bounds": [[0.0, 1.0]], "num_samples": 2},
        context={"config_registry": registry},
    )
    oracle = TypeAdapter(OracleConfig).validate_python(
        {"type": "BraninOracle", "fidelity_costs": {1: 1.0}},
        context={"config_registry": registry},
    )

    assert sampler.type == "HypercubeSampler"
    assert oracle.type == "BraninOracle"
    with pytest.raises(ValidationError, match="Unknown sampler configuration type"):
        TypeAdapter(SamplerConfig).validate_python(
            {"type": "S3GFNSampler", "n_samples": 2},
            context={"config_registry": registry},
        )


def test_explicit_package_catalogs_extend_all_public_categories() -> None:
    """A package mapping extends every public registry namespace uniformly."""
    models = {
        category: _buildable_model(f"External{category.title()}")
        for category in (
            "dataset",
            "surrogate",
            "acquisition",
            "sampler",
            "selector",
            "oracle",
            "logger",
            "encoder",
            "fixed_encoder",
        )
    }
    registry = create_config_registry(
        {"test-package": {category: (model,) for category, model in models.items()}}
    )

    aliases = {
        "dataset": DatasetConfig,
        "surrogate": SurrogateConfig,
        "acquisition": AcquisitionConfig,
        "sampler": SamplerConfig,
        "selector": SelectorConfig,
        "oracle": OracleConfig,
        "logger": LoggerConfig,
        "encoder": EncoderConfig,
        "fixed_encoder": FixedEncoderConfig,
    }
    for category, config_type in aliases.items():
        parsed = TypeAdapter(config_type).validate_python(
            {"type": f"External{category.title()}"},
            context={"config_registry": registry},
        )
        assert isinstance(parsed, models[category])


def test_nested_registered_configs_use_the_selected_registry() -> None:
    """Composite configs preserve external concrete models recursively."""
    registry = create_config_registry()
    fake_oracle = _buildable_model("FakeOracle")
    fake_oracle.__annotations__.update(
        {
            "fidelity_costs": dict[int, float],
            "input_representation": ClassVar[str],
        }
    )
    fake_oracle.input_representation = "numeric"
    fake_encoder = _buildable_model("FakeEncoder")
    fake_encoder.__annotations__["input_representation"] = ClassVar[str]
    fake_encoder.input_representation = "numeric"
    registry.install(
        {"oracle": (fake_oracle,), "encoder": (fake_encoder,)},
        origin="test-package",
    )

    composite = CompositeOracleConfig.model_validate(
        {
            "type": "CompositeOracle",
            "sub_oracles": [
                {"type": "FakeOracle", "fidelity_costs": {1: 1.0}},
            ],
        },
        context={"config_registry": registry},
    )
    dkl = ExactDKLSurrogateConfig.model_validate(
        {"encoder": {"type": "FakeEncoder"}},
        context={"config_registry": registry},
    )

    assert isinstance(composite.sub_oracles[0], fake_oracle)
    assert isinstance(dkl.encoder, fake_encoder)


def test_external_logger_can_be_nested_but_multilogger_cannot() -> None:
    """Registered logger leaves work inside MultiLogger without recursion."""
    registry = create_config_registry()
    external_logger = _buildable_model("ExternalLogger")
    registry.install({"logger": (external_logger,)}, origin="test-package")

    parsed = MultiLoggerConfig.model_validate(
        {
            "type": "MultiLogger",
            "loggers": [{"type": "ExternalLogger"}],
        },
        context={"config_registry": registry},
    )
    assert isinstance(parsed.loggers[0], external_logger)

    with pytest.raises(ValidationError, match="cannot contain another MultiLogger"):
        MultiLoggerConfig.model_validate(
            {
                "type": "MultiLogger",
                "loggers": [{"type": "MultiLogger", "loggers": []}],
            },
            context={"config_registry": registry},
        )


def test_duplicate_component_registration_names_the_conflict() -> None:
    """Duplicate discriminator names fail with both package identities."""
    registry = ConfigRegistry()
    first = _buildable_model("Duplicate")
    second = _buildable_model("Duplicate")
    registry.install({"sampler": (first,)}, origin="first-package")

    with pytest.raises(
        ValueError,
        match=r"already registered.*first-package.*cannot register",
    ):
        registry.install({"sampler": (second,)}, origin="second-package")


def test_invalid_catalog_models_are_rejected() -> None:
    """Catalog mappings must contain BuildableConfig classes."""
    with pytest.raises(TypeError, match="BuildableConfig subclass"):
        create_config_registry().install(
            {"sampler": ("not a model",)},
            origin="invalid-package",
        )


def test_unknown_catalog_categories_are_rejected() -> None:
    """Catalog mappings cannot silently add unsupported namespaces."""
    with pytest.raises(ValueError, match="Unsupported configuration category"):
        create_config_registry().install(
            {"unknown": ()},
            origin="invalid-package",
        )


def test_botorch_compatibility_uses_explicit_config_flags() -> None:
    """Third-party configs can declare compatibility without core class checks."""
    acquisition = _buildable_model("RequiresBoTorch")
    acquisition.requires_botorch_surrogate = True
    compatible_surrogate = _buildable_model("BoTorchCompatible")
    compatible_surrogate.is_botorch_compatible = True
    incompatible_surrogate = _buildable_model("NotBoTorchCompatible")
    sampler = _buildable_model("CapabilitySampler")
    oracle = _buildable_model("CapabilityOracle")

    _validate_component_compatibility(
        sampler=sampler(),
        surrogate=compatible_surrogate(),
        acquisition=acquisition(),
        oracle=oracle(),
    )

    with pytest.raises(ValueError, match="requires a BoTorch-compatible surrogate"):
        _validate_component_compatibility(
            sampler=sampler(),
            surrogate=incompatible_surrogate(),
            acquisition=acquisition(),
            oracle=oracle(),
        )


def test_declared_sampler_requires_an_oracle_representation() -> None:
    """Known sampler output cannot be validated against an opaque oracle."""
    sampler = _buildable_model("KnownOutputSampler")
    sampler.output_representation = "smiles"

    with pytest.raises(ValueError, match="does not declare its input_representation"):
        _validate_component_compatibility(
            sampler=sampler(),
            surrogate=_buildable_model("GenericSurrogate")(),
            acquisition=_buildable_model("GenericAcquisition")(),
            oracle=_buildable_model("OpaqueOracle")(),
        )
