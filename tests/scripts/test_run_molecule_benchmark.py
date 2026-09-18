from pathlib import Path

import pytest

from scripts.run_molecule_benchmark import (
    DEFAULT_METHODS,
    DEFAULT_SEEDS,
    DEFAULT_TASKS,
    build_jobs,
)


def make_config_tree(root: Path) -> None:
    """Create the benchmark paths required by the command builder."""
    config_root = root / "applications" / "molecules" / "config" / "xtb_ipea_benchmark"
    for relative_path in (
        "base.yaml",
        "encoders/gp_molformer.yaml",
        "methods/s3gfn.yaml",
        "methods/random_fidelity_s3gfn.yaml",
        "methods/random.yaml",
        "tasks/ea_sf.yaml",
        "tasks/ea_mf.yaml",
        "tasks/ip_sf.yaml",
        "tasks/ip_mf.yaml",
        "data/ea_sf.csv",
        "data/ea_mf.csv",
        "data/ip_sf.csv",
        "data/ip_mf.csv",
    ):
        path = config_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def test_default_matrix_has_all_task_method_seed_combinations(tmp_path: Path) -> None:
    """The default matrix expands to the documented 24 independent runs."""
    make_config_tree(tmp_path)

    jobs = build_jobs(repository_root=tmp_path)

    assert len(jobs) == len(DEFAULT_TASKS) * len(DEFAULT_METHODS) * len(DEFAULT_SEEDS)
    assert len({job.output_dir for job in jobs}) == len(jobs)
    assert all("runtime.seed=" in " ".join(job.command) for job in jobs)


def test_sf_method_uses_sf_task_and_s3gfn_overlay(tmp_path: Path) -> None:
    """SF-S3-GFN selects the single-fidelity task overlay."""
    make_config_tree(tmp_path)

    job = build_jobs(
        repository_root=tmp_path,
        tasks=("ea",),
        methods=("sf_s3gfn",),
        seeds=(42,),
    )[0]

    assert any("tasks/ea_sf.yaml" in argument for argument in job.command)
    assert any("methods/s3gfn.yaml" in argument for argument in job.command)
    assert job.output_dir == tmp_path / "outputs/xtb_ipea_benchmark/ea/sf_s3gfn/seed_42"


def test_random_methods_use_multi_fidelity_task(tmp_path: Path) -> None:
    """Random methods always use the multi-fidelity initialization."""
    make_config_tree(tmp_path)

    jobs = build_jobs(
        repository_root=tmp_path,
        tasks=("ip",),
        methods=("random_fidelity_s3gfn", "random"),
        seeds=(43,),
        extra_overrides=("budget.available_budget=7",),
    )

    assert all(
        any("tasks/ip_mf.yaml" in argument for argument in job.command) for job in jobs
    )
    assert all("budget.available_budget=7" in job.command for job in jobs)


def test_unknown_method_and_negative_seed_are_rejected(tmp_path: Path) -> None:
    """Invalid public matrix values fail before any subprocess is launched."""
    make_config_tree(tmp_path)

    with pytest.raises(ValueError, match="Unknown method"):
        build_jobs(repository_root=tmp_path, methods=("ppo",))
    with pytest.raises(ValueError, match="non-negative"):
        build_jobs(repository_root=tmp_path, seeds=(-1,))
