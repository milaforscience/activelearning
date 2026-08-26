"""Run resumable CUDA experiments for the optimized S3-GFN benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

DEFAULT_OUTPUT_DIR = Path("benchmarks/results/optimized_s3gfn_cuda_experiments")
CHECKPOINT_NAME = "checkpoint.json"

_ABLATION_FLAGS = {
    "precision": "fp32",
    "fixed_feature_maps": False,
    "parallel_cuda_rollout": False,
    "compile_mode": "eager",
    "deferred_sync": False,
    "carried_prior_scores": False,
    "overlap_online_prior": False,
    "combined_aux_policy_batch": False,
    "stop_check_interval": 1,
    "prior_cache_enabled": True,
    "prior_cache_capacity": 8192,
}


def _ablations(**overrides: Any) -> dict[str, Any]:
    """Return one complete optimized-model ablation configuration."""
    values = dict(_ABLATION_FLAGS)
    values.update(overrides)
    return values


STAGE_REGISTRY: dict[str, dict[str, Any]] = {
    "reference": {
        "implementation": "reference",
        "family": "baseline",
        "parent": None,
        "stage_index": 0,
        "description": "Current reference S3-GFN benchmark path.",
        "compile": {
            "precision": "fp32",
            "compile_mode": "eager",
        },
        "optimizations": {
            "terminal_state_cache": False,
            "prior_cache": False,
        },
        "ablations": _ablations(prior_cache_enabled=False),
    },
    "optimized-baseline": {
        "implementation": "optimized",
        "family": "baseline",
        "parent": "reference",
        "stage_index": 0,
        "description": "Current optimized S3-GFN baseline.",
        "compile": {
            "precision": "fp32",
            "compile_mode": "eager",
        },
        "optimizations": {
            "terminal_state_cache": True,
            "prior_cache": True,
        },
        "ablations": _ablations(),
    },
    "stage-1-precision": {
        "implementation": "optimized",
        "family": "stage-1-precision",
        "parent": "optimized-baseline",
        "stage_index": 1,
        "description": "Precision and TF32 enabled without later stages.",
        "compile": {
            "precision": "cuda_auto",
            "compile_mode": "eager",
        },
        "optimizations": {"precision": True, "tf32": True},
        "ablations": _ablations(precision="cuda_auto"),
    },
    "stage-1-cumulative": {
        "implementation": "optimized",
        "family": "cumulative",
        "parent": "optimized-baseline",
        "stage_index": 1,
        "description": "Cumulative stack through mixed precision and TF32.",
        "compile": {
            "precision": "cuda_auto",
            "compile_mode": "eager",
        },
        "optimizations": {"precision": True, "tf32": True},
        "ablations": _ablations(precision="cuda_auto"),
    },
    "stage-2-feature-maps": {
        "implementation": "optimized",
        "family": "stage-2-feature-maps",
        "parent": "optimized-baseline",
        "stage_index": 2,
        "description": "Fixed random-feature projections without later stages.",
        "compile": {
            "precision": "fp32",
            "compile_mode": "eager",
        },
        "optimizations": {"fixed_feature_maps": True},
        "ablations": _ablations(fixed_feature_maps=True),
    },
    "stage-2-cumulative": {
        "implementation": "optimized",
        "family": "cumulative",
        "parent": "stage-1-cumulative",
        "stage_index": 2,
        "description": "Cumulative stack through fixed feature maps.",
        "compile": {
            "precision": "cuda_auto",
            "compile_mode": "eager",
        },
        "optimizations": {"precision": True, "tf32": True, "fixed_feature_maps": True},
        "ablations": _ablations(
            precision="cuda_auto",
            fixed_feature_maps=True,
        ),
    },
    "stage-3-rollout": {
        "implementation": "optimized",
        "family": "stage-3-rollout",
        "parent": "optimized-baseline",
        "stage_index": 3,
        "description": "Device-resident parallel rollout without later stages.",
        "compile": {
            "precision": "fp32",
            "compile_mode": "eager",
        },
        "optimizations": {"parallel_cuda_rollout": True},
        "ablations": _ablations(parallel_cuda_rollout=True),
    },
    "stage-3-cumulative": {
        "implementation": "optimized",
        "family": "cumulative",
        "parent": "stage-2-cumulative",
        "stage_index": 3,
        "description": "Cumulative stack through parallel CUDA rollout.",
        "compile": {
            "precision": "cuda_auto",
            "compile_mode": "eager",
        },
        "optimizations": {
            "precision": True,
            "fixed_feature_maps": True,
            "parallel_cuda_rollout": True,
        },
        "ablations": _ablations(
            precision="cuda_auto",
            fixed_feature_maps=True,
            parallel_cuda_rollout=True,
        ),
    },
    "stage-4-compile": {
        "implementation": "optimized",
        "family": "stage-4-compile",
        "parent": "optimized-baseline",
        "stage_index": 4,
        "description": "Regional compilation without later stages.",
        "compile": {
            "precision": "fp32",
            "compile_mode": "default",
        },
        "optimizations": {"compile": True},
        "ablations": _ablations(compile_mode="default"),
    },
    "stage-4-cumulative": {
        "implementation": "optimized",
        "family": "cumulative",
        "parent": "stage-3-cumulative",
        "stage_index": 4,
        "description": "Cumulative stack through regional compilation.",
        "compile": {
            "precision": "cuda_auto",
            "compile_mode": "default",
        },
        "optimizations": {
            "precision": True,
            "fixed_feature_maps": True,
            "parallel_cuda_rollout": True,
            "compile": True,
        },
        "ablations": _ablations(
            precision="cuda_auto",
            fixed_feature_maps=True,
            parallel_cuda_rollout=True,
            compile_mode="default",
        ),
    },
    "stage-5-sync": {
        "implementation": "optimized",
        "family": "stage-5-sync",
        "parent": "optimized-baseline",
        "stage_index": 5,
        "description": "Deferred CUDA scalar synchronization without later stages.",
        "compile": {
            "precision": "fp32",
            "compile_mode": "eager",
        },
        "optimizations": {"deferred_sync": True},
        "ablations": _ablations(deferred_sync=True),
    },
    "stage-5-cumulative": {
        "implementation": "optimized",
        "family": "cumulative",
        "parent": "stage-4-cumulative",
        "stage_index": 5,
        "description": "Cumulative stack through deferred synchronization.",
        "compile": {
            "precision": "cuda_auto",
            "compile_mode": "default",
        },
        "optimizations": {
            "precision": True,
            "fixed_feature_maps": True,
            "parallel_cuda_rollout": True,
            "compile": True,
            "deferred_sync": True,
        },
        "ablations": _ablations(
            precision="cuda_auto",
            fixed_feature_maps=True,
            parallel_cuda_rollout=True,
            compile_mode="default",
            deferred_sync=True,
        ),
    },
    "stage-6-prior": {
        "implementation": "optimized",
        "family": "stage-6-prior",
        "parent": "optimized-baseline",
        "stage_index": 6,
        "description": "Carried deterministic replay prior scores without later stages.",
        "compile": {
            "precision": "fp32",
            "compile_mode": "eager",
        },
        "optimizations": {"carried_prior_scores": True},
        "ablations": _ablations(carried_prior_scores=True),
    },
    "stage-6-cumulative": {
        "implementation": "optimized",
        "family": "cumulative",
        "parent": "stage-5-cumulative",
        "stage_index": 6,
        "description": "Cumulative stack through carried replay prior scores.",
        "compile": {
            "precision": "cuda_auto",
            "compile_mode": "default",
        },
        "optimizations": {
            "precision": True,
            "fixed_feature_maps": True,
            "parallel_cuda_rollout": True,
            "compile": True,
            "deferred_sync": True,
            "carried_prior_scores": True,
        },
        "ablations": _ablations(
            precision="cuda_auto",
            fixed_feature_maps=True,
            parallel_cuda_rollout=True,
            compile_mode="default",
            deferred_sync=True,
            carried_prior_scores=True,
        ),
    },
    "stage-7-overlap-online-prior": {
        "implementation": "optimized",
        "family": "stage-7-overlap-online-prior",
        "parent": "optimized-baseline",
        "stage_index": 7,
        "description": "CUDA prior/policy stream overlap without later stages.",
        "compile": {
            "precision": "fp32",
            "compile_mode": "eager",
        },
        "optimizations": {"overlap_online_prior": True},
        "ablations": _ablations(overlap_online_prior=True),
    },
    "stage-7-combined-aux-policy-batch": {
        "implementation": "optimized",
        "family": "stage-7-combined-aux-policy-batch",
        "parent": "optimized-baseline",
        "stage_index": 7,
        "description": "Combined positive/negative replay policy batch without later stages.",
        "compile": {
            "precision": "fp32",
            "compile_mode": "eager",
        },
        "optimizations": {"combined_aux_policy_batch": True},
        "ablations": _ablations(combined_aux_policy_batch=True),
    },
    "stage-7-cumulative": {
        "implementation": "optimized",
        "family": "cumulative",
        "parent": "stage-6-cumulative",
        "stage_index": 7,
        "description": "Cumulative stack through CUDA overlap and replay batching.",
        "compile": {
            "precision": "cuda_auto",
            "compile_mode": "default",
        },
        "optimizations": {
            "precision": True,
            "fixed_feature_maps": True,
            "parallel_cuda_rollout": True,
            "compile": True,
            "deferred_sync": True,
            "carried_prior_scores": True,
            "overlap_online_prior": True,
            "combined_aux_policy_batch": True,
        },
        "ablations": _ablations(
            precision="cuda_auto",
            fixed_feature_maps=True,
            parallel_cuda_rollout=True,
            compile_mode="default",
            deferred_sync=True,
            carried_prior_scores=True,
            overlap_online_prior=True,
            combined_aux_policy_batch=True,
        ),
    },
}

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "description": "Short CUDA smoke profile.",
        "sequence_length": 64,
        "max_length": 64,
        "warmup": 1,
        "iterations": 3,
        "smoke_steps": 0,
    },
    "full": {
        "description": "Longer CUDA comparison profile.",
        "sequence_length": 140,
        "max_length": 140,
        "warmup": 2,
        "iterations": 5,
        "smoke_steps": 10,
    },
}


@dataclass(frozen=True)
class RunSpec:
    """One stage/profile/seed/batch benchmark invocation."""

    run_id: str
    stage: str
    implementation: str
    profile: str
    seed: int
    batch_size: int
    sequence_length: int
    max_length: int
    warmup: int
    iterations: int
    smoke_steps: int
    real_model: bool
    model_name: str
    tokenizer_name: str
    cache_dir: str | None
    hidden_size: int
    vocabulary_size: int
    gpu_utilization_trace: bool
    ablations: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkerResult:
    """Structured worker outcome returned to the orchestrator."""

    run_id: str
    stage: str
    benchmark: dict[str, Any]


class CUDAValidationError(RuntimeError):
    """Raised when CUDA validation fails before a benchmark run starts."""


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a simple linear-interpolated percentile."""
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize numeric samples with median, p95, and dispersion fields."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev": None,
            "cv": None,
        }
    numeric = [float(value) for value in values]
    median = statistics.median(numeric)
    stdev = 0.0 if len(numeric) == 1 else statistics.stdev(numeric)
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": median,
        "p95": _percentile(numeric, 0.95),
        "min": min(numeric),
        "max": max(numeric),
        "stdev": stdev,
        "cv": None if median == 0.0 else stdev / median,
    }


def _atomic_write_text(path: Path, payload: str) -> None:
    """Write text atomically inside the project tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write one JSON file atomically."""
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the flattened summary CSV atomically."""
    if not rows:
        _atomic_write_text(path, "")
        return
    fieldnames = sorted({key for row in rows for key in row})
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def validate_cuda_environment(device: str = "cuda") -> dict[str, Any]:
    """Validate CUDA availability before any benchmarked model download."""
    if device != "cuda":
        raise CUDAValidationError("CUDA experiment runner only supports --device cuda.")
    if not torch.cuda.is_available():
        raise CUDAValidationError("CUDA is not available in this environment.")
    index = torch.cuda.current_device()
    return {
        "device": device,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "current_device": index,
        "device_name": torch.cuda.get_device_name(index),
    }


def _command_output(command: Sequence[str]) -> str | None:
    """Return trimmed command output, or ``None`` when metadata is unavailable."""
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _git_metadata() -> dict[str, Any]:
    """Collect the commit and dirty-tree state for reproducibility."""
    commit = _command_output(["git", "rev-parse", "HEAD"])
    status = _command_output(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": None if commit is None else bool(status),
    }


def collect_environment_metadata() -> dict[str, Any]:
    """Collect lightweight environment metadata for the experiment manifest."""
    metadata: dict[str, Any] = {
        "created_at": _utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": (
            None if not torch.cuda.is_available() else torch.backends.cudnn.version()
        ),
        "git": _git_metadata(),
    }
    if torch.cuda.is_available():
        driver_version = _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        metadata["cuda"] = {
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "driver_version": driver_version,
        }
    return metadata


def _parse_stage_names(stage_names: Sequence[str]) -> list[str]:
    """Validate and normalize the selected stage names."""
    unknown = [name for name in stage_names if name not in STAGE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}")
    return list(stage_names)


def resolve_profile(
    name: str,
    *,
    warmup_override: int | None,
    iterations_override: int | None,
    sequence_length_override: int | None,
    max_length_override: int | None,
    smoke_steps_override: int | None,
) -> dict[str, Any]:
    """Resolve one named profile with optional CLI overrides."""
    if name not in PROFILE_PRESETS:
        raise ValueError(f"Unknown profile: {name}")
    profile = dict(PROFILE_PRESETS[name])
    if warmup_override is not None:
        profile["warmup"] = warmup_override
    if iterations_override is not None:
        profile["iterations"] = iterations_override
    if sequence_length_override is not None:
        profile["sequence_length"] = sequence_length_override
    if max_length_override is not None:
        profile["max_length"] = max_length_override
    if smoke_steps_override is not None:
        profile["smoke_steps"] = smoke_steps_override
    return profile


def plan_runs(args: argparse.Namespace) -> list[RunSpec]:
    """Build the cross-product run plan for the selected stages and profiles."""
    stage_names = _parse_stage_names(args.stages)
    run_specs: list[RunSpec] = []
    for stage_name in stage_names:
        stage = STAGE_REGISTRY[stage_name]
        for profile_name in args.profiles:
            profile = resolve_profile(
                profile_name,
                warmup_override=args.warmup,
                iterations_override=args.iterations,
                sequence_length_override=args.sequence_length,
                max_length_override=args.max_length,
                smoke_steps_override=args.smoke_steps,
            )
            for seed in args.seeds:
                for batch_size in args.batch_sizes:
                    run_id = (
                        f"{stage_name}__{profile_name}__seed{seed}__batch{batch_size}"
                    )
                    run_specs.append(
                        RunSpec(
                            run_id=run_id,
                            stage=stage_name,
                            implementation=stage["implementation"],
                            profile=profile_name,
                            seed=seed,
                            batch_size=batch_size,
                            sequence_length=profile["sequence_length"],
                            max_length=profile["max_length"],
                            warmup=profile["warmup"],
                            iterations=profile["iterations"],
                            smoke_steps=profile["smoke_steps"],
                            real_model=not args.fake_model,
                            model_name=args.model_name,
                            tokenizer_name=args.tokenizer_name,
                            cache_dir=args.cache_dir,
                            hidden_size=args.hidden_size,
                            vocabulary_size=args.vocabulary_size,
                            gpu_utilization_trace=args.gpu_utilization_trace,
                            ablations=dict(stage["ablations"]),
                        )
                    )
    return run_specs


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load an existing checkpoint if present."""
    if not path.exists():
        return {"completed_runs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def update_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Persist checkpoint state atomically."""
    checkpoint["updated_at"] = _utc_now()
    atomic_write_json(path, checkpoint)


def build_manifest(
    args: argparse.Namespace, run_specs: Sequence[RunSpec]
) -> dict[str, Any]:
    """Create the experiment manifest written before execution starts."""
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "output_dir": str(Path(args.output_dir)),
        "resume": bool(args.resume),
        "device": args.device,
        "stages": args.stages,
        "profiles": args.profiles,
        "seeds": args.seeds,
        "batch_sizes": args.batch_sizes,
        "stage_registry": STAGE_REGISTRY,
        "profile_presets": PROFILE_PRESETS,
        "environment": collect_environment_metadata(),
        "cli_arguments": vars(args),
        "model": {
            "real_model": not args.fake_model,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "cache_dir": args.cache_dir,
            "hidden_size": args.hidden_size,
            "vocabulary_size": args.vocabulary_size,
        },
        "run_plan": [asdict(run_spec) for run_spec in run_specs],
    }


def build_benchmark_namespace(
    run_spec: RunSpec,
    *,
    gpu_trace_path: str | None,
) -> argparse.Namespace:
    """Translate one run spec into benchmark CLI arguments."""
    ablations = dict(
        _ABLATION_FLAGS if run_spec.ablations is None else run_spec.ablations
    )
    return argparse.Namespace(
        device="cuda",
        real_model=run_spec.real_model,
        implementation=run_spec.implementation,
        model_name=run_spec.model_name,
        tokenizer_name=run_spec.tokenizer_name,
        cache_dir=run_spec.cache_dir,
        batch_size=run_spec.batch_size,
        sequence_length=run_spec.sequence_length,
        max_length=run_spec.max_length,
        hidden_size=run_spec.hidden_size,
        vocabulary_size=run_spec.vocabulary_size,
        warmup=run_spec.warmup,
        iterations=run_spec.iterations,
        seed=run_spec.seed,
        smoke_steps=run_spec.smoke_steps,
        profile_step=False,
        gpu_utilization_trace=gpu_trace_path,
        json_output=None,
        precision=ablations["precision"],
        fixed_feature_maps=ablations["fixed_feature_maps"],
        parallel_cuda_rollout=ablations["parallel_cuda_rollout"],
        compile_mode=ablations["compile_mode"],
        deferred_sync=ablations["deferred_sync"],
        carried_prior_scores=ablations["carried_prior_scores"],
        overlap_online_prior=ablations["overlap_online_prior"],
        combined_aux_policy_batch=ablations["combined_aux_policy_batch"],
        stop_check_interval=ablations["stop_check_interval"],
        prior_cache_enabled=ablations["prior_cache_enabled"],
        prior_cache_capacity=ablations["prior_cache_capacity"],
    )


def run_worker(
    run_spec: RunSpec,
    *,
    benchmark_runner: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
    cuda_validator: Callable[[str], dict[str, Any]] = validate_cuda_environment,
    gpu_trace_path: str | None = None,
) -> WorkerResult:
    """Execute one benchmark worker in-process for tests or the hidden CLI."""
    cuda_validator("cuda")
    if benchmark_runner is None:
        from benchmarks.optimized_s3gfn_benchmark import run_benchmark

        benchmark_runner = run_benchmark
    benchmark = benchmark_runner(
        build_benchmark_namespace(run_spec, gpu_trace_path=gpu_trace_path)
    )
    return WorkerResult(
        run_id=run_spec.run_id,
        stage=run_spec.stage,
        benchmark=benchmark,
    )


def parse_worker_stdout(stdout: str) -> WorkerResult:
    """Parse one worker's JSON payload."""
    payload = json.loads(stdout)
    return WorkerResult(
        run_id=payload["run_id"],
        stage=payload["stage"],
        benchmark=payload["benchmark"],
    )


def build_worker_command(run_spec: RunSpec, *, run_dir: Path) -> list[str]:
    """Build the fresh subprocess command for one run."""
    ablations = dict(
        _ABLATION_FLAGS if run_spec.ablations is None else run_spec.ablations
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--run-id",
        run_spec.run_id,
        "--stage",
        run_spec.stage,
        "--implementation",
        run_spec.implementation,
        "--profile",
        run_spec.profile,
        "--seed",
        str(run_spec.seed),
        "--batch-size",
        str(run_spec.batch_size),
        "--sequence-length",
        str(run_spec.sequence_length),
        "--max-length",
        str(run_spec.max_length),
        "--warmup",
        str(run_spec.warmup),
        "--iterations",
        str(run_spec.iterations),
        "--smoke-steps",
        str(run_spec.smoke_steps),
        "--model-name",
        run_spec.model_name,
        "--tokenizer-name",
        run_spec.tokenizer_name,
        "--hidden-size",
        str(run_spec.hidden_size),
        "--vocabulary-size",
        str(run_spec.vocabulary_size),
        "--precision",
        str(ablations["precision"]),
        "--compile-mode",
        str(ablations["compile_mode"]),
        "--stop-check-interval",
        str(ablations["stop_check_interval"]),
        "--prior-cache-capacity",
        str(ablations["prior_cache_capacity"]),
    ]
    for flag in (
        "fixed-feature-maps",
        "parallel-cuda-rollout",
        "deferred-sync",
        "carried-prior-scores",
        "overlap-online-prior",
        "combined-aux-policy-batch",
    ):
        option_name = flag.replace("-", "_")
        if ablations[option_name]:
            command.append(f"--{flag}")
    if not ablations["prior_cache_enabled"]:
        command.append("--no-prior-cache")
    if run_spec.real_model:
        command.append("--real-model")
    if run_spec.cache_dir is not None:
        command.extend(["--cache-dir", run_spec.cache_dir])
    if run_spec.gpu_utilization_trace:
        command.extend(
            [
                "--gpu-utilization-trace-path",
                str(run_dir / "nvidia_smi_dmon.log"),
            ]
        )
    return command


def _select_implementation_metrics(
    benchmark: dict[str, Any], implementation: str
) -> dict[str, Any]:
    """Select the stage-specific benchmark payload from one worker report."""
    implementations = benchmark.get("implementations", {})
    if implementation not in implementations:
        raise KeyError(f"Benchmark output is missing {implementation!r} metrics.")
    return implementations[implementation]


def execute_run(
    run_spec: RunSpec,
    *,
    output_dir: Path,
    subprocess_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Execute one run in a fresh subprocess and preserve raw artifacts."""
    run_dir = output_dir / "runs" / run_spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_worker_command(run_spec, run_dir=run_dir)
    started_at = _utc_now()
    finished_at = _utc_now()
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    record: dict[str, Any] = {
        "run_id": run_spec.run_id,
        "stage": run_spec.stage,
        "implementation": run_spec.implementation,
        "profile": run_spec.profile,
        "seed": run_spec.seed,
        "batch_size": run_spec.batch_size,
        "sequence_length": run_spec.sequence_length,
        "max_length": run_spec.max_length,
        "warmup": run_spec.warmup,
        "iterations": run_spec.iterations,
        "smoke_steps": run_spec.smoke_steps,
        "real_model": run_spec.real_model,
        "model_name": run_spec.model_name,
        "tokenizer_name": run_spec.tokenizer_name,
        "cache_dir": run_spec.cache_dir,
        "started_at": started_at,
        "finished_at": finished_at,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stage_metadata": STAGE_REGISTRY[run_spec.stage],
        "ablations": dict(
            _ABLATION_FLAGS if run_spec.ablations is None else run_spec.ablations
        ),
        "command": " ".join(shlex.quote(part) for part in command),
    }
    try:
        completed = subprocess_runner(
            command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        stdout = getattr(error, "stdout", "") or ""
        stderr = getattr(error, "stderr", "") or str(error)
        stdout_path.write_text(str(stdout), encoding="utf-8")
        stderr_path.write_text(str(stderr), encoding="utf-8")
        record.update(
            {
                "finished_at": _utc_now(),
                "returncode": None,
                "status": "failure",
                "error": f"Worker process failed to start or complete: {error}",
            }
        )
        return record
    finished_at = _utc_now()
    record["finished_at"] = finished_at
    record["returncode"] = completed.returncode
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    try:
        worker_result = parse_worker_stdout(completed.stdout)
        raw_json_path = run_dir / "raw_result.json"
        atomic_write_json(
            raw_json_path,
            {
                "run_id": worker_result.run_id,
                "stage": worker_result.stage,
                "benchmark": worker_result.benchmark,
            },
        )
        benchmark = worker_result.benchmark
        record["benchmark_path"] = str(raw_json_path)
        record["benchmark"] = _select_implementation_metrics(
            benchmark,
            run_spec.implementation,
        )
        record["quality"] = benchmark.get("quality", {})
        record["status"] = "success" if completed.returncode == 0 else "failure"
        if completed.returncode != 0:
            record["error"] = "Worker exited nonzero despite JSON output."
    except (json.JSONDecodeError, KeyError) as error:
        record["status"] = "failure"
        record["error"] = f"Malformed worker output: {error}"
    if completed.returncode != 0 and record["status"] == "success":
        record["status"] = "failure"
    return record


def _metric_median(record: dict[str, Any], *path: str) -> float | None:
    """Read one nested median-like numeric field from a run record."""
    current: Any = record.get("benchmark")
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def _metric_values(records: Sequence[dict[str, Any]], *path: str) -> list[float]:
    """Collect one nested numeric field from all matching run records."""
    values = []
    for record in records:
        value = _metric_median(record, *path)
        if value is not None:
            values.append(value)
    return values


def _training_quality(record: dict[str, Any]) -> dict[str, Any]:
    """Return the optional training quality payload from one run record."""
    benchmark = record.get("benchmark")
    if not isinstance(benchmark, dict):
        return {}
    training_step = benchmark.get("training_step")
    if not isinstance(training_step, dict):
        return {}
    quality = training_step.get("quality")
    return quality if isinstance(quality, dict) else {}


_SPEEDUP_METRICS = {
    "generation_wall_ms": ("generation", "wall_ms", "median"),
    "prior_wall_ms": ("prior", "wall_ms", "median"),
    "training_step_wall_ms": ("training_step", "wall_ms", "median"),
}


def _speedups_against(
    candidate_records: Sequence[dict[str, Any]],
    baseline_index: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    """Calculate speedups only for matched successful samples."""
    pairs = []
    for candidate in candidate_records:
        key = (
            str(candidate["profile"]),
            int(candidate["batch_size"]),
            int(candidate["seed"]),
        )
        baseline = baseline_index.get(key)
        if baseline is not None:
            pairs.append((baseline, candidate))

    speedups: dict[str, Any] = {"matched_samples": len(pairs)}
    for name, path in _SPEEDUP_METRICS.items():
        ratios = []
        for baseline, candidate in pairs:
            baseline_value = _metric_median(baseline, *path)
            candidate_value = _metric_median(candidate, *path)
            if (
                baseline_value is not None
                and candidate_value is not None
                and candidate_value > 0.0
            ):
                ratios.append(baseline_value / candidate_value)
        speedups[name] = _summarize(ratios)
    return speedups


def aggregate_run_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate successful runs by stage/profile/batch size."""
    successes = [record for record in records if record.get("status") == "success"]
    failures = [record for record in records if record.get("status") != "success"]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in successes:
        key = (record["stage"], record["profile"], int(record["batch_size"]))
        grouped.setdefault(key, []).append(record)

    stage_indices: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {}
    for record in successes:
        key = (
            str(record["profile"]),
            int(record["batch_size"]),
            int(record["seed"]),
        )
        stage_indices.setdefault(str(record["stage"]), {})[key] = record

    summaries: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for (stage, profile, batch_size), group_records in sorted(grouped.items()):
        stage_metadata = STAGE_REGISTRY[stage]
        quality_checked = [
            record.get("quality", {}).get("checked")
            for record in group_records
            if isinstance(record.get("quality"), dict)
        ]
        generation_matches = [
            record["quality"]["generation_matches"]
            for record in group_records
            if record.get("quality", {}).get("checked") is True
            and "generation_matches" in record["quality"]
        ]
        fidelity_matches = [
            record["quality"]["fidelity_matches"]
            for record in group_records
            if record.get("quality", {}).get("checked") is True
            and "fidelity_matches" in record["quality"]
        ]
        phase_names = sorted(
            {
                name
                for record in group_records
                for name in record["benchmark"]
                .get("training_step", {})
                .get("phases", {})
                .get("phases_ms", {})
            }
        )
        phase_summary = {
            name: _summarize(
                [
                    float(
                        record["benchmark"]["training_step"]["phases"]["phases_ms"][
                            name
                        ]["median"]
                    )
                    for record in group_records
                    if name
                    in record["benchmark"]
                    .get("training_step", {})
                    .get("phases", {})
                    .get("phases_ms", {})
                ]
            )
            for name in phase_names
        }
        quality_metric_names = (
            "validity_rate",
            "uniqueness_rate",
            "synthesizable_rate",
            "candidate_yield",
            "acquisition_mean",
            "acquisition_max",
        )
        quality_metrics = {
            name: _summarize(
                _metric_values(
                    group_records,
                    "training_step",
                    "quality",
                    name,
                    "median",
                )
            )
            for name in quality_metric_names
        }
        fidelity_names_set: set[str] = set()
        for record in group_records:
            frequencies = _training_quality(record).get("fidelity_frequencies")
            if isinstance(frequencies, dict):
                fidelity_names_set.update(str(fidelity) for fidelity in frequencies)
        fidelity_names = sorted(fidelity_names_set)
        quality_metrics["fidelity_frequencies"] = {
            fidelity: _summarize(
                _metric_values(
                    group_records,
                    "training_step",
                    "quality",
                    "fidelity_frequencies",
                    fidelity,
                    "median",
                )
            )
            for fidelity in fidelity_names
        }
        cuda_utilization = {
            "sm_average_percent": _summarize(
                _metric_values(
                    group_records,
                    "gpu_utilization",
                    "sm_average_percent",
                )
            ),
            "memory_average_percent": _summarize(
                _metric_values(
                    group_records,
                    "gpu_utilization",
                    "memory_average_percent",
                )
            ),
        }
        optimization_metrics = {
            "precision_dtypes": sorted(
                {
                    str(
                        record["benchmark"]
                        .get("optimization", {})
                        .get("precision_dtype")
                    )
                    for record in group_records
                    if record["benchmark"]
                    .get("optimization", {})
                    .get("precision_dtype")
                    is not None
                }
            ),
            "feature_map_redraw_count": _summarize(
                _metric_values(
                    group_records,
                    "optimization",
                    "feature_map_redraw_count",
                )
            ),
            "compile_cold_start_s": _summarize(
                _metric_values(
                    group_records,
                    "optimization",
                    "compile_cold_start_s",
                )
            ),
            "compile_graph_count": _summarize(
                _metric_values(
                    group_records,
                    "optimization",
                    "compile_graph_count",
                )
            ),
            "compile_graph_breaks": _summarize(
                _metric_values(
                    group_records,
                    "optimization",
                    "compile_graph_breaks",
                )
            ),
            "compile_recompilations": _summarize(
                _metric_values(
                    group_records,
                    "optimization",
                    "compile_recompilations",
                )
            ),
        }
        summary = {
            "stage": stage,
            "profile": profile,
            "batch_size": batch_size,
            "successful_runs": len(group_records),
            "failed_runs": len(
                [
                    record
                    for record in failures
                    if record["stage"] == stage
                    and record["profile"] == profile
                    and int(record["batch_size"]) == batch_size
                ]
            ),
            "seeds": sorted(int(record["seed"]) for record in group_records),
            "model": {
                "real_model": bool(group_records[0]["real_model"]),
                "model_name": group_records[0]["model_name"],
                "tokenizer_name": group_records[0]["tokenizer_name"],
                "cache_dir": group_records[0].get("cache_dir"),
                "sequence_length": int(group_records[0]["sequence_length"]),
                "max_length": int(group_records[0]["max_length"]),
            },
            "compile": stage_metadata["compile"],
            "ablations": stage_metadata["ablations"],
            "parent_stage": stage_metadata.get("parent"),
            "quality": {
                "successful_runs": len(group_records),
                "failed_runs": len(
                    [
                        record
                        for record in failures
                        if record["stage"] == stage
                        and record["profile"] == profile
                        and int(record["batch_size"]) == batch_size
                    ]
                ),
                "checked_runs": sum(1 for value in quality_checked if value is True),
                "generation_match_rate": (
                    None
                    if not generation_matches
                    else sum(1 for value in generation_matches if value)
                    / len(generation_matches)
                ),
                "fidelity_match_rate": (
                    None
                    if not fidelity_matches
                    else sum(1 for value in fidelity_matches if value)
                    / len(fidelity_matches)
                ),
            },
            "generation": {
                "wall_ms": _summarize(
                    _metric_values(group_records, "generation", "wall_ms", "median")
                ),
                "throughput_per_s": _summarize(
                    _metric_values(
                        group_records,
                        "generation",
                        "throughput_per_s",
                        "median",
                    )
                ),
                "peak_allocated_bytes": _summarize(
                    _metric_values(
                        group_records,
                        "generation",
                        "peak_allocated_bytes",
                        "median",
                    )
                ),
                "peak_reserved_bytes": _summarize(
                    _metric_values(
                        group_records,
                        "generation",
                        "peak_reserved_bytes",
                        "median",
                    )
                ),
            },
            "prior": {
                "wall_ms": _summarize(
                    _metric_values(group_records, "prior", "wall_ms", "median")
                ),
                "throughput_per_s": _summarize(
                    _metric_values(
                        group_records,
                        "prior",
                        "throughput_per_s",
                        "median",
                    )
                ),
                "peak_allocated_bytes": _summarize(
                    _metric_values(
                        group_records,
                        "prior",
                        "peak_allocated_bytes",
                        "median",
                    )
                ),
                "peak_reserved_bytes": _summarize(
                    _metric_values(
                        group_records,
                        "prior",
                        "peak_reserved_bytes",
                        "median",
                    )
                ),
            },
            "training_step": {
                "wall_ms": _summarize(
                    _metric_values(
                        group_records,
                        "training_step",
                        "wall_ms",
                        "median",
                    )
                ),
                "throughput_per_s": _summarize(
                    _metric_values(
                        group_records,
                        "training_step",
                        "throughput_per_s",
                        "median",
                    )
                ),
                "peak_allocated_bytes": _summarize(
                    _metric_values(
                        group_records,
                        "training_step",
                        "peak_allocated_bytes",
                        "median",
                    )
                ),
                "peak_reserved_bytes": _summarize(
                    _metric_values(
                        group_records,
                        "training_step",
                        "peak_reserved_bytes",
                        "median",
                    )
                ),
            },
            "throughput": {
                "generated_trajectories_per_s": _summarize(
                    _metric_values(
                        group_records,
                        "training_step",
                        "generated_trajectories_per_s",
                        "median",
                    )
                ),
                "valid_trajectories_per_s": _summarize(
                    _metric_values(
                        group_records,
                        "training_step",
                        "valid_trajectories_per_s",
                        "median",
                    )
                ),
            },
            "phase": phase_summary,
            "quality_metrics": quality_metrics,
            "cuda_utilization": cuda_utilization,
            "optimization_metrics": optimization_metrics,
            "prior_cache": {
                "hits": _summarize(
                    [
                        float(record["benchmark"]["prior_cache"]["hits"])
                        for record in group_records
                    ]
                ),
                "misses": _summarize(
                    [
                        float(record["benchmark"]["prior_cache"]["misses"])
                        for record in group_records
                    ]
                ),
            },
            "speedups_vs_reference": None,
            "speedups_vs_parent": None,
        }
        if stage != "reference":
            summary["speedups_vs_reference"] = _speedups_against(
                group_records,
                stage_indices.get("reference", {}),
            )
        parent_stage = stage_metadata.get("parent")
        if parent_stage is not None:
            summary["speedups_vs_parent"] = _speedups_against(
                group_records,
                stage_indices.get(parent_stage, {}),
            )
        summaries.append(summary)
        reference_speedups = summary["speedups_vs_reference"]
        parent_speedups = summary["speedups_vs_parent"]
        ablations = summary["ablations"]
        seed_values = summary["seeds"]
        csv_row = {
            "stage": stage,
            "stage_family": stage_metadata["family"],
            "stage_index": stage_metadata["stage_index"],
            "parent_stage": summary["parent_stage"],
            "profile": profile,
            "batch_size": batch_size,
            "seed": (
                seed_values[0]
                if len(seed_values) == 1
                else ",".join(str(seed) for seed in seed_values)
            ),
            "successful_runs": len(group_records),
            "failed_runs": summary["failed_runs"],
            "precision": ablations["precision"],
            "compile_mode": ablations["compile_mode"],
            "ablations_json": json.dumps(ablations, sort_keys=True),
            "generation_wall_ms_median": summary["generation"]["wall_ms"]["median"],
            "generation_wall_ms_p95": summary["generation"]["wall_ms"]["p95"],
            "generation_trajectories_per_s_median": summary["generation"][
                "throughput_per_s"
            ]["median"],
            "prior_wall_ms_median": summary["prior"]["wall_ms"]["median"],
            "training_step_wall_ms_median": summary["training_step"]["wall_ms"][
                "median"
            ],
            "training_step_wall_ms_p95": summary["training_step"]["wall_ms"]["p95"],
            "training_step_trajectories_per_s_median": summary["training_step"][
                "throughput_per_s"
            ]["median"],
            "generated_trajectories_per_s_median": summary["throughput"][
                "generated_trajectories_per_s"
            ]["median"],
            "valid_trajectories_per_s_median": summary["throughput"][
                "valid_trajectories_per_s"
            ]["median"],
            "validity_rate_median": summary["quality_metrics"]["validity_rate"][
                "median"
            ],
            "uniqueness_rate_median": summary["quality_metrics"]["uniqueness_rate"][
                "median"
            ],
            "synthesizable_rate_median": summary["quality_metrics"][
                "synthesizable_rate"
            ]["median"],
            "candidate_yield_median": summary["quality_metrics"]["candidate_yield"][
                "median"
            ],
            "acquisition_mean_median": summary["quality_metrics"]["acquisition_mean"][
                "median"
            ],
            "acquisition_max_median": summary["quality_metrics"]["acquisition_max"][
                "median"
            ],
            "generation_speedup_vs_reference": (
                None
                if reference_speedups is None
                else reference_speedups["generation_wall_ms"]["median"]
            ),
            "prior_speedup_vs_reference": (
                None
                if reference_speedups is None
                else reference_speedups["prior_wall_ms"]["median"]
            ),
            "training_step_speedup_vs_reference": (
                None
                if reference_speedups is None
                else reference_speedups["training_step_wall_ms"]["median"]
            ),
            "generation_speedup_vs_parent": (
                None
                if parent_speedups is None
                else parent_speedups["generation_wall_ms"]["median"]
            ),
            "prior_speedup_vs_parent": (
                None
                if parent_speedups is None
                else parent_speedups["prior_wall_ms"]["median"]
            ),
            "training_step_speedup_vs_parent": (
                None
                if parent_speedups is None
                else parent_speedups["training_step_wall_ms"]["median"]
            ),
            "peak_allocated_bytes_median": summary["training_step"][
                "peak_allocated_bytes"
            ]["median"],
            "peak_reserved_bytes_median": summary["training_step"][
                "peak_reserved_bytes"
            ]["median"],
            "gpu_sm_average_percent_median": summary["cuda_utilization"][
                "sm_average_percent"
            ]["median"],
            "gpu_memory_average_percent_median": summary["cuda_utilization"][
                "memory_average_percent"
            ]["median"],
            "feature_map_redraw_count_median": summary["optimization_metrics"][
                "feature_map_redraw_count"
            ]["median"],
            "compile_cold_start_s_median": summary["optimization_metrics"][
                "compile_cold_start_s"
            ]["median"],
            "compile_graph_count_median": summary["optimization_metrics"][
                "compile_graph_count"
            ]["median"],
            "compile_graph_breaks_median": summary["optimization_metrics"][
                "compile_graph_breaks"
            ]["median"],
            "compile_recompilations_median": summary["optimization_metrics"][
                "compile_recompilations"
            ]["median"],
            "quality_generation_match_rate": summary["quality"][
                "generation_match_rate"
            ],
            "quality_fidelity_match_rate": summary["quality"]["fidelity_match_rate"],
            "status": "success",
        }
        for phase_name, phase_values in phase_summary.items():
            csv_row[f"phase_{phase_name}_median_ms"] = phase_values["median"]
            csv_row[f"phase_{phase_name}_p95_ms"] = phase_values["p95"]
        for fidelity, values in quality_metrics["fidelity_frequencies"].items():
            csv_row[f"fidelity_{fidelity}_frequency_median"] = values["median"]
        csv_rows.append(csv_row)
    return {
        "totals": {
            "records": len(records),
            "successful": len(successes),
            "failed": len(failures),
        },
        "groups": summaries,
        "csv_rows": csv_rows,
    }


def run_experiments(
    args: argparse.Namespace,
    *,
    subprocess_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Run the full experiment matrix and persist manifest, checkpoint, and summaries."""
    validate_cuda_environment(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.jsonl"
    checkpoint_path = output_dir / CHECKPOINT_NAME
    run_specs = plan_runs(args)
    manifest = build_manifest(args, run_specs)
    atomic_write_json(output_dir / "manifest.json", manifest)

    checkpoint = (
        load_checkpoint(checkpoint_path) if args.resume else {"completed_runs": {}}
    )
    if not args.resume and runs_path.exists():
        runs_path.unlink()

    all_records: list[dict[str, Any]] = []
    if args.resume and runs_path.exists():
        all_records.extend(
            json.loads(line)
            for line in runs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    completed_runs = checkpoint.get("completed_runs", {})
    for run_spec in run_specs:
        if args.resume and run_spec.run_id in completed_runs:
            continue
        record = execute_run(
            run_spec,
            output_dir=output_dir,
            subprocess_runner=subprocess_runner,
        )
        append_jsonl(runs_path, record)
        all_records.append(record)
        completed_runs[run_spec.run_id] = {
            "status": record["status"],
            "finished_at": record["finished_at"],
        }
        checkpoint["completed_runs"] = completed_runs
        update_checkpoint(checkpoint_path, checkpoint)

    summary = aggregate_run_records(all_records)
    atomic_write_json(output_dir / "summary.json", summary)
    _write_summary_csv(output_dir / "summary.csv", summary["csv_rows"])
    failures = [record for record in all_records if record.get("status") != "success"]
    atomic_write_json(output_dir / "failures.json", failures)
    return 1 if failures else 0


def _build_worker_parser() -> argparse.ArgumentParser:
    """Create the hidden worker parser used by subprocess runs."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-worker", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--implementation", choices=("reference", "optimized"), required=True
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--smoke-steps", type=int, required=True)
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--vocabulary-size", type=int, required=True)
    parser.add_argument(
        "--precision",
        choices=("fp32", "cuda_auto"),
        default="fp32",
    )
    parser.add_argument(
        "--fixed-feature-maps",
        action="store_true",
    )
    parser.add_argument(
        "--parallel-cuda-rollout",
        action="store_true",
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "eager",
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
            "max-autotune",
        ),
        default="eager",
    )
    parser.add_argument("--deferred-sync", action="store_true")
    parser.add_argument("--carried-prior-scores", action="store_true")
    parser.add_argument("--overlap-online-prior", action="store_true")
    parser.add_argument("--combined-aux-policy-batch", action="store_true")
    parser.add_argument("--stop-check-interval", type=int, default=1)
    parser.add_argument("--no-prior-cache", action="store_true")
    parser.add_argument("--prior-cache-capacity", type=int, default=8192)
    parser.add_argument("--gpu-utilization-trace-path")
    return parser


def _run_internal_worker(argv: Sequence[str]) -> int:
    """Execute the hidden worker CLI and emit one JSON payload to stdout."""
    parser = _build_worker_parser()
    args = parser.parse_args(argv)
    worker_result = run_worker(
        RunSpec(
            run_id=args.run_id,
            stage=args.stage,
            implementation=args.implementation,
            profile=args.profile,
            seed=args.seed,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            max_length=args.max_length,
            warmup=args.warmup,
            iterations=args.iterations,
            smoke_steps=args.smoke_steps,
            real_model=args.real_model,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            cache_dir=args.cache_dir,
            hidden_size=args.hidden_size,
            vocabulary_size=args.vocabulary_size,
            gpu_utilization_trace=bool(args.gpu_utilization_trace_path),
            ablations={
                "precision": args.precision,
                "fixed_feature_maps": args.fixed_feature_maps,
                "parallel_cuda_rollout": args.parallel_cuda_rollout,
                "compile_mode": args.compile_mode,
                "deferred_sync": args.deferred_sync,
                "carried_prior_scores": args.carried_prior_scores,
                "overlap_online_prior": args.overlap_online_prior,
                "combined_aux_policy_batch": args.combined_aux_policy_batch,
                "stop_check_interval": args.stop_check_interval,
                "prior_cache_enabled": not args.no_prior_cache,
                "prior_cache_capacity": args.prior_cache_capacity,
            },
        ),
        gpu_trace_path=args.gpu_utilization_trace_path,
    )
    print(
        json.dumps(
            {
                "run_id": worker_result.run_id,
                "stage": worker_result.stage,
                "benchmark": worker_result.benchmark,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level orchestration CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(STAGE_REGISTRY),
        choices=tuple(STAGE_REGISTRY),
    )
    parser.add_argument(
        "--profiles",
        "--profile",
        dest="profiles",
        nargs="+",
        default=["quick"],
        choices=tuple(PROFILE_PRESETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fake-model", action="store_true")
    parser.add_argument(
        "--model-name",
        default="ibm-research/GP-MoLFormer-Uniq",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="ibm-research/MoLFormer-XL-both-10pct",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--vocabulary-size", type=int, default=64)
    parser.add_argument("--gpu-utilization-trace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--internal-worker" in argv:
        return _run_internal_worker(argv)
    args = build_parser().parse_args(argv)
    try:
        return run_experiments(args)
    except CUDAValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
