"""Configuration composition tests for the xTB IP/EA benchmark."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from activelearning.config import ActiveLearningConfig
from activelearning.utils.config_loader import load_and_parse
from activelearning_molecules.config_catalogs import CONFIG_CATALOGS
from activelearning_molecules.samplers.config import RandomMoleculeSamplerConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPOSITORY_ROOT / "applications/molecules/config/xtb_ipea_benchmark"
CATALOGS = {"activelearning-molecules": CONFIG_CATALOGS}


def load_benchmark_config(task: str, method: str) -> ActiveLearningConfig:
    """Load one benchmark composition through the production registries."""
    task_kind = "sf" if method == "sf_s3gfn" else "mf"
    method_overlay = {
        "sf_s3gfn": "s3gfn",
        "mf_s3gfn": "s3gfn",
        "random_fidelity_s3gfn": "random_fidelity_s3gfn",
        "random": "random",
    }[method]
    return load_and_parse(
        [
            CONFIG_ROOT / "base.yaml",
            CONFIG_ROOT / "encoders/gp_molformer.yaml",
            CONFIG_ROOT / f"tasks/{task}_{task_kind}.yaml",
            CONFIG_ROOT / f"methods/{method_overlay}.yaml",
        ],
        ActiveLearningConfig,
        catalogs=CATALOGS,
    )


@pytest.mark.parametrize("task", ["ea", "ip"])
@pytest.mark.parametrize(
    "method",
    ["sf_s3gfn", "mf_s3gfn", "random_fidelity_s3gfn", "random"],
)
def test_all_public_benchmark_compositions_parse(task: str, method: str) -> None:
    """All two-task/four-method compositions resolve through Pydantic."""
    config = load_benchmark_config(task, method)

    assert config.surrogate.type == "VariationalGPSurrogate"
    assert config.surrogate.encoder.type == "GPMoLFormerSmilesFixedEncoder"
    assert config.surrogate.encoder.input_representation == "smiles"
    assert config.oracle.task == task
    assert config.oracle.mol_repr == "smiles"
    assert config.selector.num_samples == 128
    assert config.budget.available_budget == 1260.0
    assert config.run_writer is not None
    assert config.run_writer.output_dir == Path(
        "outputs/xtb_ipea_benchmark/default/seed_42"
    )

    if method == "sf_s3gfn":
        assert config.oracle.fidelity_costs == {3: 7.0}
        assert config.oracle.per_fidelity_num_conformers == {3: 4}
        assert config.sampler.fidelities == [3]
        assert config.acquisition.type == "QLowerBoundMaxValueEntropy"
        assert config.dataset.initial_data.default_fidelity == 3
        assert config.surrogate.is_multi_fidelity is False
    elif method == "mf_s3gfn":
        assert config.oracle.fidelity_costs == {1: 1.0, 2: 3.5, 3: 7.0}
        assert config.oracle.per_fidelity_num_conformers == {1: 1, 2: 2, 3: 4}
        assert config.sampler.fidelities == [1, 2, 3]
        assert config.acquisition.type == "QMultiFidelityLowerBoundMaxValueEntropy"
        assert config.sampler.fidelity_policy == "learned"
        assert config.surrogate.is_multi_fidelity is True
    elif method == "random_fidelity_s3gfn":
        assert config.sampler.fidelity_policy == "uniform"
        assert config.sampler.reward_fidelity == 3
        assert config.sampler.fidelities == [1, 2, 3]
    else:
        assert isinstance(config.sampler, RandomMoleculeSamplerConfig)
        assert config.sampler.fidelities == [1, 2, 3]


def test_ip_initial_targets_are_negated_and_sf_rows_use_fidelity_three() -> None:
    """IP signs and the SF default fidelity are resolved at data load time."""
    config = load_benchmark_config("ip", "sf_s3gfn")
    observations = config.dataset.initial_data.load_observations(
        negate_targets=config.dataset.negate_initial_targets
    )

    assert observations
    assert {observation.fidelity for observation in observations} == {3}
    assert observations[0].y < 0.0


def test_encoder_overlay_can_be_replaced_with_minimol(tmp_path: Path) -> None:
    """The task/method overlays remain valid with a different encoder overlay."""
    encoder = tmp_path / "minimol.yaml"
    encoder.write_text(
        "surrogate:\n  encoder:\n    type: MiniMolSmilesFixedEncoder\n",
        encoding="utf-8",
    )

    config = load_and_parse(
        [
            CONFIG_ROOT / "base.yaml",
            encoder,
            CONFIG_ROOT / "tasks/ea_mf.yaml",
            CONFIG_ROOT / "methods/s3gfn.yaml",
        ],
        ActiveLearningConfig,
        catalogs=CATALOGS,
    )

    assert config.surrogate.encoder.type == "MiniMolSmilesFixedEncoder"
    assert config.oracle.task == "ea"


def test_sf_task_rejects_random_fidelity_method_by_construction() -> None:
    """The public runner's method mapping cannot compose random SF runs."""
    with pytest.raises(ValidationError, match="at least two oracle fidelities"):
        load_and_parse(
            [
                CONFIG_ROOT / "base.yaml",
                CONFIG_ROOT / "encoders/gp_molformer.yaml",
                CONFIG_ROOT / "tasks/ea_sf.yaml",
                CONFIG_ROOT / "methods/random_fidelity_s3gfn.yaml",
            ],
            ActiveLearningConfig,
            catalogs=CATALOGS,
        )
