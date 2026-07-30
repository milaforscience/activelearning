"""Configuration-level tests for the checked-in tutorial examples."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from activelearning.config import ActiveLearningConfig
from activelearning.utils.config_loader import load_and_parse, load_config, parse_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_multi_fidelity_branin_tutorial_config_parses() -> None:
    """Ensure the multi-fidelity tutorial config matches the current schema."""
    config_path = REPOSITORY_ROOT / "config" / "branin" / "multi_fidelity.yaml"

    config = load_and_parse(config_path, ActiveLearningConfig)

    assert config.oracle.type == "BraninOracle"
    assert config.budget.available_budget == 300.0
    assert config.runtime.seed == 42
    assert config.logger is not None


def test_single_fidelity_branin_tutorial_config_parses() -> None:
    """Ensure the single-fidelity tutorial config matches the current schema."""
    config_path = REPOSITORY_ROOT / "config" / "branin" / "single_fidelity.yaml"

    config = load_and_parse(config_path, ActiveLearningConfig)

    assert config.oracle.type == "BraninOracle"
    assert config.budget.schedule.type == "constant"
    assert config.runtime.seed == 42
    assert config.logger is not None


def test_single_fidelity_config_derives_sampler_level_and_surrogate_mode() -> None:
    """The oracle's sole level must drive omitted sampler and surrogate settings."""
    config_path = REPOSITORY_ROOT / "config" / "branin" / "single_fidelity.yaml"
    raw_config = load_config(config_path)
    raw_config.sampler.fidelities = None

    config = parse_config(raw_config, ActiveLearningConfig)

    assert config.sampler.fidelities == [1]
    assert config.surrogate.is_multi_fidelity is False


def test_single_fidelity_dkl_config_clears_target_level() -> None:
    """A target level is irrelevant and unsafe when DKL fidelity input is disabled."""
    config_path = REPOSITORY_ROOT / "config" / "molecules" / "exact.yaml"
    raw_config = load_config(config_path)
    raw_config.surrogate.target_fidelity = 99

    config = parse_config(raw_config, ActiveLearningConfig)

    assert config.surrogate.multi_fidelity is False
    assert config.surrogate.target_fidelity is None


def test_multi_fidelity_hartmann_tutorial_config_parses() -> None:
    """Ensure the multi-fidelity Hartmann tutorial config matches the current schema."""
    config_path = REPOSITORY_ROOT / "config" / "hartmann" / "multi_fidelity.yaml"

    config = load_and_parse(config_path, ActiveLearningConfig)

    assert config.oracle.type == "Hartmann6DOracle"
    assert config.budget.available_budget == 100.0
    assert config.budget.schedule.type == "constant"
    assert config.runtime.seed == 42
    assert config.logger is not None


def test_single_fidelity_hartmann_tutorial_config_parses() -> None:
    """Ensure the single-fidelity Hartmann tutorial config matches the current schema."""
    config_path = REPOSITORY_ROOT / "config" / "hartmann" / "single_fidelity.yaml"

    config = load_and_parse(config_path, ActiveLearningConfig)

    assert config.oracle.type == "Hartmann6DOracle"
    assert config.budget.schedule.type == "constant"
    assert config.runtime.seed == 42
    assert config.logger is not None


def test_aim_logging_overlay_parses_when_merged_with_base_config() -> None:
    """Ensure the Aim logging overlay remains schema-valid when merged with a base config."""
    config = load_and_parse(
        [
            REPOSITORY_ROOT / "config" / "branin" / "single_fidelity.yaml",
            REPOSITORY_ROOT / "config" / "aim_logging.yaml",
        ],
        ActiveLearningConfig,
    )

    assert config.runtime.seed == 42
    assert config.logger is not None
    assert config.logger.type == "MultiLogger"


def test_molecule_dkl_exact_gflownet_config_parses() -> None:
    """Ensure the SELFIES GFlowNet molecule tutorial config matches the schema."""
    config_path = REPOSITORY_ROOT / "config" / "molecules" / "gflownet_exact.yaml"

    config = load_and_parse(config_path, ActiveLearningConfig)

    assert config.sampler.type == "GFlowNetSampler"
    assert config.sampler.fidelities == [1]
    assert config.selector.type == "TopKAcquisitionSelector"
    assert config.oracle.type == "XTBIPEAOracle"
    assert config.acquisition.type == "UpperConfidenceBound"


def test_molecule_dkl_exact_multi_fidelity_gflownet_config_parses() -> None:
    """Ensure the exact SELFIES multi-fidelity GFlowNet config matches the schema."""
    config_path = (
        REPOSITORY_ROOT / "config" / "molecules" / "gflownet_exact_multi_fidelity.yaml"
    )

    config = load_and_parse(config_path, ActiveLearningConfig)

    assert config.sampler.type == "GFlowNetSampler"
    assert config.sampler.fidelities == [1, 2, 3]
    assert config.selector.type == "TopKAcquisitionSelector"
    assert config.oracle.type == "XTBIPEAOracle"
    assert config.acquisition.type == "QMultiFidelityLowerBoundMaxValueEntropy"
    assert config.oracle.num_conformers == 2
    assert config.oracle.per_fidelity_num_conformers == {1: 1, 2: 2, 3: 4}


def test_dkl_target_is_derived_from_highest_oracle_confidence() -> None:
    """Explicit confidences, rather than costs, determine an omitted DKL target."""
    config_path = REPOSITORY_ROOT / "config" / "molecules" / "exact_multi_fidelity.yaml"
    raw_config = load_config(config_path)
    raw_config.surrogate.target_fidelity = None
    raw_config.oracle.fidelity_confidences = {1: 0.2, 2: 1.0, 3: 0.7}

    config = parse_config(raw_config, ActiveLearningConfig)

    assert config.surrogate.multi_fidelity is True
    assert config.surrogate.target_fidelity == 2


def test_dkl_target_is_derived_through_composite_oracle() -> None:
    """Composite oracle metadata must support the same target derivation."""
    config_path = REPOSITORY_ROOT / "config" / "molecules" / "exact_multi_fidelity.yaml"
    raw_config = load_config(config_path)
    raw_config.surrogate.target_fidelity = None
    raw_config.oracle.fidelity_confidences = {1: 0.2, 2: 1.0, 3: 0.7}
    sub_oracle = OmegaConf.to_container(raw_config.oracle, resolve=True)
    raw_config.oracle = {
        "type": "CompositeOracle",
        "sub_oracles": [sub_oracle],
    }

    config = parse_config(raw_config, ActiveLearningConfig)

    assert config.surrogate.multi_fidelity is True
    assert config.surrogate.target_fidelity == 2


def test_dkl_target_must_be_declared_by_oracle() -> None:
    """An explicit target outside the oracle fidelity set must be rejected."""
    config_path = REPOSITORY_ROOT / "config" / "molecules" / "exact_multi_fidelity.yaml"
    raw_config = load_config(config_path)
    raw_config.surrogate.target_fidelity = 99

    with pytest.raises(ValidationError, match="target_fidelity 99"):
        parse_config(raw_config, ActiveLearningConfig)


def test_sampler_fidelities_must_not_be_empty() -> None:
    """An empty configured fidelity set must fail during parsing."""
    config_path = REPOSITORY_ROOT / "config" / "branin" / "multi_fidelity.yaml"
    raw_config = load_config(config_path)
    raw_config.sampler.fidelities = []

    with pytest.raises(ValidationError):
        parse_config(raw_config, ActiveLearningConfig)


def test_sampler_fidelities_must_not_contain_duplicates() -> None:
    """Duplicate IDs must not change the sampler's derived fidelity mode."""
    config_path = REPOSITORY_ROOT / "config" / "branin" / "single_fidelity.yaml"
    raw_config = load_config(config_path)
    raw_config.sampler.fidelities = [1, 1]

    with pytest.raises(ValidationError, match="duplicate"):
        parse_config(raw_config, ActiveLearningConfig)


@pytest.mark.parametrize("cost", [0.0, -1.0])
def test_sampler_fidelity_costs_must_be_positive(cost: float) -> None:
    """Invalid sampler costs must fail during parsing."""
    config_path = REPOSITORY_ROOT / "config" / "branin" / "single_fidelity.yaml"
    raw_config = load_config(config_path)
    raw_config.sampler.fidelities = {1: cost}

    with pytest.raises(ValidationError):
        parse_config(raw_config, ActiveLearningConfig)


def test_sampler_fidelities_must_belong_to_oracle() -> None:
    """Sampler levels outside the oracle's authoritative set must be rejected."""
    config_path = REPOSITORY_ROOT / "config" / "branin" / "multi_fidelity.yaml"
    raw_config = load_config(config_path)
    raw_config.sampler.fidelities = [1, 4]

    with pytest.raises(ValidationError, match="Sampler fidelities \\[4\\]"):
        parse_config(raw_config, ActiveLearningConfig)


def test_composite_oracle_rejects_conflicting_confidences() -> None:
    """Sub-oracles must agree when they expose the same fidelity level."""
    config_path = REPOSITORY_ROOT / "config" / "branin" / "multi_fidelity.yaml"
    raw_config = load_config(config_path)
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


def test_molecule_dkl_variational_multi_fidelity_gflownet_config_parses() -> None:
    """Ensure the variational SELFIES multi-fidelity GFlowNet config matches the schema."""
    config_path = (
        REPOSITORY_ROOT
        / "config"
        / "molecules"
        / "gflownet_variational_multi_fidelity.yaml"
    )

    config = load_and_parse(config_path, ActiveLearningConfig)

    assert config.sampler.type == "GFlowNetSampler"
    assert config.sampler.fidelities == [1, 2, 3]
    assert config.selector.type == "TopKAcquisitionSelector"
    assert config.oracle.type == "XTBIPEAOracle"
    assert config.acquisition.type == "QMultiFidelityLowerBoundMaxValueEntropy"
    assert config.oracle.num_conformers == 2
    assert config.oracle.per_fidelity_num_conformers == {1: 1, 2: 2, 3: 4}


def test_molecule_dkl_exact_pool_config_parses() -> None:
    """Ensure the exact single-fidelity pool-based molecule example matches the schema."""
    config_path = REPOSITORY_ROOT / "config" / "molecules" / "exact.yaml"

    config = load_and_parse(config_path, ActiveLearningConfig)

    assert config.sampler.type == "PoolFileSampler"
    assert config.surrogate.type == "ExactSelfiesDKLSurrogate"
    assert config.acquisition.type == "UpperConfidenceBound"
    assert config.oracle.type == "XTBIPEAOracle"
    assert config.sampler.fidelities == [1]


def test_molecule_dkl_exact_multi_fidelity_pool_config_parses() -> None:
    """Ensure the exact multi-fidelity pool-based molecule example matches the schema."""
    config_path = REPOSITORY_ROOT / "config" / "molecules" / "exact_multi_fidelity.yaml"

    config = load_and_parse(config_path, ActiveLearningConfig)

    assert config.sampler.type == "PoolFileSampler"
    assert config.surrogate.type == "ExactSelfiesDKLSurrogate"
    assert config.acquisition.type == "QMultiFidelityLowerBoundMaxValueEntropy"
    assert config.selector.type == "CostAwareSelector"
    assert config.oracle.type == "XTBIPEAOracle"
    assert config.sampler.fidelities == [1, 2, 3]
