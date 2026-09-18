"""Run the composable xTB IP/EA molecule benchmark."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_TASKS = ("ea", "ip")
DEFAULT_METHODS = (
    "sf_s3gfn",
    "mf_s3gfn",
    "random_fidelity_s3gfn",
    "random",
)


@dataclass(frozen=True)
class BenchmarkJob:
    """One resolved benchmark command and its output directory."""

    task: str
    method: str
    seed: int
    command: tuple[str, ...]
    output_dir: Path


def build_jobs(
    *,
    repository_root: Path,
    tasks: Sequence[str] = DEFAULT_TASKS,
    methods: Sequence[str] = DEFAULT_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    extra_overrides: Sequence[str] = (),
    executable: Sequence[str] = ("uv", "run", "activelearning-molecules"),
) -> list[BenchmarkJob]:
    """Build the validated task/method/seed command matrix.

    Parameters
    ----------
    repository_root : Path
        Repository root containing the molecule application and configs.
    tasks : sequence of str
        Task names, ``"ea"`` and/or ``"ip"``.
    methods : sequence of str
        Public method identifiers.
    seeds : sequence of int
        Runtime seeds to run.
    extra_overrides : sequence of str
        User-supplied OmegaConf overrides applied before runner-owned metadata.
    executable : sequence of str
        Command prefix used to invoke the molecule CLI.
    """
    task_values = _validate_choices(tasks, DEFAULT_TASKS, "task")
    method_values = _validate_choices(methods, DEFAULT_METHODS, "method")
    seed_values = _validate_seeds(seeds)
    root = repository_root.resolve()
    config_root = root / "applications" / "molecules" / "config" / "xtb_ipea_benchmark"

    jobs: list[BenchmarkJob] = []
    for task in task_values:
        for method in method_values:
            task_name = "sf" if method == "sf_s3gfn" else "mf"
            task_config = config_root / "tasks" / f"{task}_{task_name}.yaml"
            method_config = config_root / "methods" / f"{_method_overlay(method)}.yaml"
            data_file = config_root / "data" / f"{task}_{task_name}.csv"
            paths = [
                config_root / "base.yaml",
                config_root / "encoders" / "gp_molformer.yaml",
                task_config,
                method_config,
            ]
            missing = [path for path in [*paths, data_file] if not path.is_file()]
            if missing:
                formatted = ", ".join(str(path) for path in missing)
                raise FileNotFoundError(
                    f"Benchmark config file(s) not found: {formatted}"
                )

            for seed in seed_values:
                output_dir = (
                    root
                    / "outputs"
                    / "xtb_ipea_benchmark"
                    / task
                    / method
                    / f"seed_{seed}"
                )
                config_args = [str(path.relative_to(root)) for path in paths]
                runner_overrides = [
                    *extra_overrides,
                    f"runtime.seed={seed}",
                    f"sampler.seed={seed}",
                    f"benchmark.task={task}",
                    f"benchmark.method={method}",
                    f"logger.run_name={task}-{method}-seed{seed}",
                    f"run_writer.output_dir={output_dir}",
                ]
                command = tuple([*executable, *config_args, *runner_overrides])
                jobs.append(
                    BenchmarkJob(
                        task=task,
                        method=method,
                        seed=seed,
                        command=command,
                        output_dir=output_dir,
                    )
                )
    return jobs


def main(argv: Sequence[str] | None = None) -> None:
    """Parse options and run or print the selected benchmark jobs."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    tasks = tuple(args.task) if args.task else DEFAULT_TASKS
    methods = tuple(args.method) if args.method else DEFAULT_METHODS
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    executable = _resolve_executable(args.executable)
    jobs = build_jobs(
        repository_root=repository_root,
        tasks=tasks,
        methods=methods,
        seeds=seeds,
        extra_overrides=args.override,
        executable=executable,
    )

    for job in jobs:
        print(f"--- {job.task}/{job.method} seed={job.seed} ---")
        print(shlex.join(job.command))
        if not args.dry_run:
            subprocess.run(job.command, cwd=repository_root, check=True)


def _build_parser() -> argparse.ArgumentParser:
    """Build the benchmark runner argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the composable xTB IP/EA molecule benchmark."
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=DEFAULT_TASKS,
        help="Task to run; repeat for both EA and IP. Defaults to both.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=DEFAULT_METHODS,
        help="Method to run; repeat for multiple methods. Defaults to all.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Seed to run; repeat for multiple seeds. Defaults to 42, 43, 44.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="OmegaConf key=value override applied to each job.",
    )
    parser.add_argument(
        "--executable",
        default=None,
        help="CLI executable or command prefix; defaults to "
        "ACTIVELEARNING_MOLECULES_COMMAND or 'uv run activelearning-molecules'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching experiments.",
    )
    return parser


def _resolve_executable(value: str | None) -> tuple[str, ...]:
    """Resolve an executable command prefix from the CLI or environment."""
    raw = value or os.environ.get(
        "ACTIVELEARNING_MOLECULES_COMMAND",
        "uv run activelearning-molecules",
    )
    executable = tuple(shlex.split(raw))
    if not executable:
        raise ValueError("The molecule CLI executable cannot be empty.")
    return executable


def _validate_choices(
    values: Sequence[str],
    allowed: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    """Validate and deduplicate a sequence of public identifiers."""
    allowed_set = set(allowed)
    normalized = tuple(dict.fromkeys(values))
    invalid = sorted(set(normalized) - allowed_set)
    if invalid:
        raise ValueError(f"Unknown {label}(s): {', '.join(invalid)}.")
    if not normalized:
        raise ValueError(f"At least one {label} is required.")
    return normalized


def _validate_seeds(values: Sequence[int]) -> tuple[int, ...]:
    """Validate and deduplicate non-negative runtime seeds."""
    normalized = tuple(dict.fromkeys(values))
    if not normalized:
        raise ValueError("At least one seed is required.")
    if any(seed < 0 for seed in normalized):
        raise ValueError("Seeds must be non-negative.")
    return normalized


def _method_overlay(method: str) -> str:
    """Map a public method identifier to its YAML overlay name."""
    if method in {"sf_s3gfn", "mf_s3gfn"}:
        return "s3gfn"
    return method


if __name__ == "__main__":
    main()
