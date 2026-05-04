"""Configuration-level tests for the checked-in tutorial examples."""

from pathlib import Path

from activelearning.config import ActiveLearningConfig
from activelearning.utils.config_loader import load_and_parse


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
    assert config.sampler.fixed_fidelity == 1
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
    assert config.sampler.n_fidelities == 3
    assert config.sampler.fixed_fidelity is None
    assert config.selector.type == "TopKAcquisitionSelector"
    assert config.oracle.type == "XTBIPEAOracle"
    assert config.acquisition.type == "QMultiFidelityLowerBoundMaxValueEntropy"
    assert config.acquisition.cost_aware_utility is not None
    assert config.oracle.num_conformers == 2
    assert config.oracle.per_fidelity_num_conformers == {1: 1, 2: 2, 3: 4}
    assert config.acquisition.cost_aware_utility.fidelity_costs == {
        1: 1.0,
        2: 3.5,
        3: 7.0,
    }


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
    assert config.sampler.n_fidelities == 3
    assert config.sampler.fixed_fidelity is None
    assert config.selector.type == "TopKAcquisitionSelector"
    assert config.oracle.type == "XTBIPEAOracle"
    assert config.acquisition.type == "QMultiFidelityLowerBoundMaxValueEntropy"
    assert config.acquisition.cost_aware_utility is not None
    assert config.oracle.num_conformers == 2
    assert config.oracle.per_fidelity_num_conformers == {1: 1, 2: 2, 3: 4}
    assert config.acquisition.cost_aware_utility.fidelity_costs == {
        1: 1.0,
        2: 3.5,
        3: 7.0,
    }


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
    assert config.acquisition.cost_aware_utility is None
    assert config.selector.type == "CostAwareSelector"
    assert config.oracle.type == "XTBIPEAOracle"
    assert config.sampler.fidelities == [1, 2, 3]
