"""Configuration-level tests for the core benchmark examples."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from activelearning.config import ActiveLearningConfig
from activelearning.monitoring.run_writer import JSONLinesRunWriter
from activelearning.utils.config_loader import load_and_parse, load_config, parse_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    (
        "overlay_name",
        "sampler_type",
        "sampler_fidelities",
        "acquisition_type",
        "oracle_costs",
        "oracle_confidences",
        "surrogate_is_multi_fidelity",
    ),
    [
        pytest.param(
            "mf_gfn",
            "ExactGridSampler",
            [1, 2, 3],
            "QMultiFidelityLowerBoundMaxValueEntropy",
            {1: 0.01, 2: 0.1, 3: 1.0},
            {1: 0.1, 2: 0.5, 3: 1.0},
            True,
            id="mf-gfn",
        ),
        pytest.param(
            "random",
            "HypercubeSampler",
            [1, 2, 3],
            "QMultiFidelityLowerBoundMaxValueEntropy",
            {1: 0.01, 2: 0.1, 3: 1.0},
            {1: 0.1, 2: 0.5, 3: 1.0},
            True,
            id="random",
        ),
        pytest.param(
            "sf_low_fid",
            "ExactGridSampler",
            [1],
            "QMultiFidelityLowerBoundMaxValueEntropy",
            {1: 0.01},
            {1: 0.1},
            False,
            id="sf-low",
        ),
        pytest.param(
            "sf_mid_fid",
            "ExactGridSampler",
            [2],
            "QMultiFidelityLowerBoundMaxValueEntropy",
            {2: 0.1},
            {2: 0.5},
            False,
            id="sf-mid",
        ),
        pytest.param(
            "sf_high_fid",
            "ExactGridSampler",
            [3],
            "QMultiFidelityLowerBoundMaxValueEntropy",
            {3: 1.0},
            {3: 1.0},
            False,
            id="sf-high",
        ),
    ],
)
def test_branin_benchmark_configs_parse(
    overlay_name: str,
    sampler_type: str,
    sampler_fidelities: list[int],
    acquisition_type: str,
    oracle_costs: dict[int, float],
    oracle_confidences: dict[int, float],
    surrogate_is_multi_fidelity: bool,
) -> None:
    """Ensure each public Branin benchmark overlay resolves as intended."""
    config = load_and_parse(
        [
            REPOSITORY_ROOT / "config" / "branin_benchmark" / "base.yaml",
            REPOSITORY_ROOT / "config" / "branin_benchmark" / f"{overlay_name}.yaml",
        ],
        ActiveLearningConfig,
    )

    assert config.oracle.type == "BraninOracle"
    assert config.sampler.type == sampler_type
    assert config.sampler.fidelities == sampler_fidelities
    assert config.acquisition.type == acquisition_type
    assert config.oracle.fidelity_costs == oracle_costs
    assert config.oracle.fidelity_confidences == oracle_confidences
    assert config.surrogate.is_multi_fidelity is surrogate_is_multi_fidelity
    assert config.budget.available_budget == 100.0
    assert config.budget.max_rounds == 300
    assert config.diagnostics.enabled is True
    assert config.diagnostics.figure_interval == 1
    assert config.diagnostics.max_points == 1000
    assert config.run_writer is not None
    assert config.run_writer.output_dir == Path(
        f"outputs/branin_benchmark/{overlay_name}/seed_42"
    )


def test_branin_benchmark_base_config_parses_run_writer(tmp_path) -> None:
    """The Branin benchmark config should resolve a runnable run writer."""
    config = load_and_parse(
        [
            REPOSITORY_ROOT / "config" / "branin_benchmark" / "base.yaml",
            REPOSITORY_ROOT / "config" / "branin_benchmark" / "mf_gfn.yaml",
        ],
        ActiveLearningConfig,
    )

    assert config.run_writer is not None
    run_writer = config.run_writer.model_copy(
        update={"output_dir": tmp_path / "benchmark-run"}
    ).build()
    assert isinstance(run_writer, JSONLinesRunWriter)


@pytest.mark.parametrize("config_name", ["single_fidelity", "multi_fidelity"])
def test_branin_tutorial_config_parses(config_name: str) -> None:
    """The standalone Branin tutorial configurations remain valid."""
    config = load_and_parse(
        REPOSITORY_ROOT / "config" / "branin" / f"{config_name}.yaml",
        ActiveLearningConfig,
    )
    assert config.oracle.type == "BraninOracle"
    assert config.sampler.type == "HypercubeSampler"


@pytest.mark.parametrize("config_name", ["single_fidelity", "multi_fidelity"])
def test_hartmann_tutorial_config_parses(config_name: str) -> None:
    """The standalone Hartmann tutorial configurations remain valid."""
    config = load_and_parse(
        REPOSITORY_ROOT / "config" / "hartmann" / f"{config_name}.yaml",
        ActiveLearningConfig,
    )
    assert config.oracle.type == "Hartmann6DOracle"


def test_aim_logging_overlay_parses_when_merged_with_base_config() -> None:
    """The Aim logging overlay remains schema-valid."""
    config = load_and_parse(
        [
            REPOSITORY_ROOT / "config" / "hartmann" / "single_fidelity.yaml",
            REPOSITORY_ROOT / "config" / "aim_logging.yaml",
        ],
        ActiveLearningConfig,
    )

    assert config.logger is not None
    assert config.logger.type == "MultiLogger"


def test_single_fidelity_config_derives_sampler_level_and_surrogate_mode() -> None:
    """The oracle's sole level drives omitted sampler and surrogate settings."""
    raw_config = load_config(
        REPOSITORY_ROOT / "config" / "hartmann" / "single_fidelity.yaml"
    )
    raw_config.sampler.fidelities = None

    config = parse_config(raw_config, ActiveLearningConfig)

    assert config.sampler.fidelities == [1]
    assert config.surrogate.is_multi_fidelity is False


def test_multi_fidelity_config_rejects_fidelity_agnostic_surrogate() -> None:
    """A multi-level oracle requires a fidelity-aware surrogate config."""
    raw_config = load_config(
        REPOSITORY_ROOT / "config" / "hartmann" / "multi_fidelity.yaml"
    )
    raw_config.surrogate = {"type": "DummyMeanSurrogate"}

    with pytest.raises(ValidationError, match="does not support multi-fidelity"):
        parse_config(raw_config, ActiveLearningConfig)


def test_sampler_fidelity_validation_rejects_invalid_values() -> None:
    """Sampler fidelity settings must be non-empty and match the oracle."""
    raw_config = load_config(
        REPOSITORY_ROOT / "config" / "hartmann" / "multi_fidelity.yaml"
    )
    raw_config.sampler.fidelities = [1, 4]

    with pytest.raises(ValidationError, match=r"Sampler fidelities \[4\]"):
        parse_config(raw_config, ActiveLearningConfig)


def test_composite_oracle_rejects_conflicting_confidences() -> None:
    """Sub-oracles must agree when they expose the same fidelity level."""
    raw_config = load_config(
        REPOSITORY_ROOT / "config" / "hartmann" / "multi_fidelity.yaml"
    )
    first_oracle = OmegaConf.to_container(raw_config.oracle, resolve=True)
    second_oracle = OmegaConf.to_container(raw_config.oracle, resolve=True)
    first_oracle["fidelity_confidences"] = {1: 0.1, 2: 0.5, 3: 1.0}
    second_oracle["fidelity_confidences"] = {1: 0.2, 2: 0.5, 3: 1.0}
    raw_config.oracle = {
        "type": "CompositeOracle",
        "sub_oracles": [first_oracle, second_oracle],
    }

    with pytest.raises(ValidationError, match="inconsistent confidence"):
        parse_config(raw_config, ActiveLearningConfig)
