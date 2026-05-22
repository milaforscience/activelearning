import csv
from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from activelearning.config import ActiveLearningConfig
from activelearning.dataset.config import CSVInitialDataConfig
from activelearning.utils.config_loader import load_and_parse, load_config

REPRODUCE_PAPER_CONFIG_ROOT = Path("scripts/configs/reproduce_paper")
PAPER_METHODS: tuple[str, ...] = ("mf_gfn", "sf_gfn", "random", "random_fid_gfn")
GFLOWNET_METHODS = frozenset({"mf_gfn", "sf_gfn", "random_fid_gfn"})
MULTI_FIDELITY_METHODS = frozenset({"mf_gfn", "random", "random_fid_gfn"})

TASK_GROUP_BY_TASK = {
    "branin": "synthetic",
    "hartmann": "synthetic",
    "molecules_ea": "molecules",
    "molecules_ip": "molecules",
}
TOTAL_BUDGET_BY_TASK = {
    "branin": 300.0,
    "hartmann": 100.0,
    "molecules_ea": 1260.0,
    "molecules_ip": 1260.0,
}
ROUND_BUDGET_BY_TASK = {
    "branin": 30.0,
    "hartmann": 10.0,
    "molecules_ea": 896.0,
    "molecules_ip": 896.0,
}
TOP_K_SELECTOR_SAMPLES_BY_TASK = {
    "branin": 30,
    "hartmann": 10,
    "molecules_ea": 128,
    "molecules_ip": 128,
}
INITIAL_DATA_BY_TASK = {
    "branin": {
        "single": "scripts/configs/reproduce_paper/data/branin/initial_single_fidelity.csv",
        "multi": "scripts/configs/reproduce_paper/data/branin/initial_multi_fidelity.csv",
    },
    "hartmann": {
        "single": "scripts/configs/reproduce_paper/data/hartmann/initial_single_fidelity.csv",
        "multi": "scripts/configs/reproduce_paper/data/hartmann/initial_multi_fidelity.csv",
    },
    "molecules_ea": {
        "single": "scripts/configs/reproduce_paper/data/molecules_ea/initial_single_fidelity.csv",
        "multi": "scripts/configs/reproduce_paper/data/molecules_ea/initial_multi_fidelity.csv",
    },
    "molecules_ip": {
        "single": "scripts/configs/reproduce_paper/data/molecules_ip/initial_single_fidelity.csv",
        "multi": "scripts/configs/reproduce_paper/data/molecules_ip/initial_multi_fidelity.csv",
    },
}
ORACLE_COSTS_BY_TASK = {
    "branin": {1: 0.01, 2: 0.1, 3: 1.0},
    "hartmann": {1: 0.125, 2: 0.25, 3: 1.0},
    "molecules_ea": {1: 1.0, 2: 3.5, 3: 7.0},
    "molecules_ip": {1: 1.0, 2: 3.5, 3: 7.0},
}
MOLECULE_INITIAL_ROW_COUNT = {
    "molecules_ea": {"single": 135, "multi": 699},
    "molecules_ip": {"single": 135, "multi": 705},
}
MOLECULE_MULTI_FIDELITY_COUNTS = {
    "molecules_ea": {1: 624, 2: 61, 3: 14},
    "molecules_ip": {1: 630, 2: 61, 3: 14},
}
SYNTHETIC_GFLOWNET_CONFIG = {
    "branin": {
        "bounds": [[-5.0, 10.0], [0.0, 15.0]],
        "n_dim": 2,
        "grid_length": 100,
        "candidate_sample_count": 150,
        "random_action_prob": 0.0,
        "optimizer_batch_size": 16,
        "reward_beta": 1.0,
        "reward_rho": 1.0,
        "fidelity_action": "first",
    },
    "hartmann": {
        "bounds": [[0.0, 1.0]] * 6,
        "n_dim": 6,
        "grid_length": 10,
        "candidate_sample_count": 50,
        "random_action_prob": 0.001,
        "optimizer_batch_size": 32,
        "reward_beta": 0.01,
        "reward_rho": 1.0,
        "fidelity_action": "any",
    },
}


@pytest.mark.parametrize("task", sorted(TASK_GROUP_BY_TASK))
@pytest.mark.parametrize("method", PAPER_METHODS)
def test_layered_reproduction_configs_parse_as_activelearning_config(
    task: str,
    method: str,
) -> None:
    """Every runnable reproduction config should parse directly through ActiveLearningConfig."""

    parsed = load_and_parse(
        _config_paths(task, method),
        ActiveLearningConfig,
        overrides=["runtime.seed=7"],
    )

    assert parsed.selector.type == "TopKAcquisitionSelector"
    assert parsed.model_dump().get("reproduce_paper") is None


@pytest.mark.parametrize("task", sorted(TASK_GROUP_BY_TASK))
@pytest.mark.parametrize("method", PAPER_METHODS)
def test_layered_configs_resolve_run_paths_and_method_specific_initial_data(
    task: str,
    method: str,
) -> None:
    """Resolved configs should encode run naming and choose the right initial-data CSV."""

    config = _resolved_config(task, method)
    expected_initial_data_mode = "single" if method == "sf_gfn" else "multi"

    assert config["logger"]["run_name"] == f"{task}-{method}-seed7"
    assert (
        config["run_writer"]["output_dir"]
        == f"outputs/reproduce_paper/{task}/{method}/seed_7"
    )
    assert (
        config["dataset"]["initial_data"]["path"]
        == INITIAL_DATA_BY_TASK[task][expected_initial_data_mode]
    )
    assert config["selector"]["num_samples"] == TOP_K_SELECTOR_SAMPLES_BY_TASK[task]
    assert config["budget"]["available_budget"] == pytest.approx(
        TOTAL_BUDGET_BY_TASK[task]
    )
    assert config["budget"]["schedule"]["value"] == pytest.approx(
        ROUND_BUDGET_BY_TASK[task]
    )


@pytest.mark.parametrize("task", ("branin", "hartmann"))
@pytest.mark.parametrize("method", sorted(GFLOWNET_METHODS))
def test_synthetic_gflownet_configs_take_domain_and_proxy_settings_from_task_yaml(
    task: str,
    method: str,
) -> None:
    """Synthetic GFN configs should resolve all task-specific domain metadata in YAML."""

    config = _resolved_config(task, method)
    expected = SYNTHETIC_GFLOWNET_CONFIG[task]

    assert config["sampler"]["n_samples"] == expected["candidate_sample_count"]
    assert config["sampler"]["output_bounds"] == expected["bounds"]
    assert config["sampler"]["conf"]["env"]["n_dim"] == expected["n_dim"]
    assert config["sampler"]["conf"]["env"]["length"] == expected["grid_length"]
    assert config["sampler"]["conf"]["gflownet"]["random_action_prob"] == pytest.approx(
        expected["random_action_prob"]
    )
    assert (
        config["sampler"]["conf"]["gflownet"]["optimizer"]["batch_size"]["forward"]
        == expected["optimizer_batch_size"]
    )
    assert config["sampler"]["conf"]["proxy"] == {
        "reward_beta": pytest.approx(expected["reward_beta"]),
        "reward_rho": pytest.approx(expected["reward_rho"]),
    }


@pytest.mark.parametrize("task", ("branin", "hartmann"))
@pytest.mark.parametrize("method", PAPER_METHODS)
def test_synthetic_reproduction_configs_use_train_data_candidate_sets_for_mes(
    task: str,
    method: str,
) -> None:
    """Synthetic MES configs should match the reference train-data candidate sets."""

    config = _resolved_config(task, method)

    assert (
        config["acquisition"]["candidate_set_spec"]["type"]
        == "TrainDataCandidateSetSpec"
    )


@pytest.mark.parametrize("task", sorted(TASK_GROUP_BY_TASK))
@pytest.mark.parametrize("method", sorted(MULTI_FIDELITY_METHODS))
def test_multifidelity_reproduction_configs_resolve_costs_from_task_metadata(
    task: str,
    method: str,
) -> None:
    """Multi-fidelity methods should keep paper costs in resolved config values."""

    config = _resolved_config(task, method)

    assert _normalize_numeric_key_dict(
        config["oracle"]["fidelity_costs"]
    ) == pytest.approx(ORACLE_COSTS_BY_TASK[task])
    assert _normalize_numeric_key_dict(
        config["acquisition"]["cost_aware_utility"]["fidelity_costs"]
    ) == pytest.approx(ORACLE_COSTS_BY_TASK[task])


def test_branin_initial_data_csvs_match_paper_initialization_counts() -> None:
    """Branin reproduction CSVs should match the paper's SF and MF initialization sizes."""

    branin_data_root = REPRODUCE_PAPER_CONFIG_ROOT / "data" / "branin"

    assert _read_fidelity_counts(branin_data_root / "initial_single_fidelity.csv") == {
        3: 4
    }
    assert _read_fidelity_counts(branin_data_root / "initial_multi_fidelity.csv") == {
        1: 20,
        2: 20,
        3: 2,
    }


def test_molecule_random_config_uses_random_token_sequence_sampler() -> None:
    """The molecule random baseline should stay a plain framework sampler config."""

    config = _resolved_config("molecules_ea", "random")

    assert config["sampler"]["type"] == "RandomTokenSequenceSampler"
    assert config["sampler"]["tokens"] == "SELFIES_VOCAB_SMALL"
    assert config["sampler"]["min_length"] == 1
    assert config["sampler"]["max_length"] == 64


def test_molecule_ip_config_negates_runtime_targets_without_extra_metadata() -> None:
    """The IP task should negate runtime targets without redundant YAML flags."""

    config = _resolved_config("molecules_ip", "mf_gfn")

    assert config["dataset"]["negate_initial_targets"] is True
    assert config["oracle"]["type"] == "XTBIPEAOracle"
    assert config["oracle"].get("negate_score") in {None, False}
    assert config["reproduce_paper"].get("negate_score") in {None, False}


@pytest.mark.parametrize("task", ("molecules_ea", "molecules_ip"))
@pytest.mark.parametrize("mode", ("single", "multi"))
def test_molecule_initial_data_csvs_parse_with_expected_counts(
    task: str,
    mode: str,
) -> None:
    """Molecule reproduction CSVs should load cleanly through the dataset parser."""

    csv_path = (
        REPRODUCE_PAPER_CONFIG_ROOT / "data" / task / f"initial_{mode}_fidelity.csv"
    )
    observations = CSVInitialDataConfig(
        path=csv_path,
        x_columns="selfies",
        y_column="y",
        fidelity_column="fidelity" if mode == "multi" else None,
    ).load_observations()

    assert len(observations) == MOLECULE_INITIAL_ROW_COUNT[task][mode]
    assert {
        observation.metadata["source_split"]
        for observation in observations
        if observation.metadata is not None
    } == {"train"}
    if mode == "single":
        assert all(observation.fidelity is None for observation in observations)
    else:
        assert (
            Counter(
                int(observation.fidelity)
                for observation in observations
                if observation.fidelity is not None
            )
            == MOLECULE_MULTI_FIDELITY_COUNTS[task]
        )


def _resolved_config(task: str, method: str) -> dict:
    """Return one resolved reproduction config as a plain dictionary."""

    return OmegaConf.to_container(
        load_config(_config_paths(task, method), overrides=["runtime.seed=7"]),
        resolve=True,
    )


def _config_paths(task: str, method: str) -> list[str]:
    """Return the runnable config path for one task/method pair."""

    task_group = TASK_GROUP_BY_TASK[task]
    return [
        str(REPRODUCE_PAPER_CONFIG_ROOT / task_group / task / f"{method}.yaml"),
    ]


def _normalize_numeric_key_dict(payload: dict[object, object]) -> dict[int, float]:
    """Normalize OmegaConf dictionaries that may serialize numeric keys as strings."""

    return {int(key): float(value) for key, value in payload.items()}


def _read_fidelity_counts(path: Path) -> dict[int, int]:
    """Return the per-fidelity observation counts recorded in one initial-data CSV."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return dict(sorted(Counter(int(row["fidelity"]) for row in rows).items()))
