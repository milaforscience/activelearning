"""Tests for the optimized S3-GFN CUDA experiment runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks import run_optimized_s3gfn_cuda_experiments as experiments


def _make_args(tmp_path: Path, *, resume: bool = False) -> object:
    return experiments.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--stages",
            "reference",
            "optimized-baseline",
            "--profiles",
            "quick",
            "--seeds",
            "0",
            "--batch-sizes",
            "8",
            "--fake-model",
            *(["--resume"] if resume else []),
        ]
    )


def _benchmark_payload(
    *, implementation: str, generation_ms: float
) -> dict[str, object]:
    return {
        "implementations": {
            implementation: {
                "generation": {
                    "wall_ms": {"median": generation_ms},
                    "throughput_per_s": {"median": 1000.0 / generation_ms},
                    "peak_allocated_bytes": {"median": 1024.0},
                    "peak_reserved_bytes": {"median": 2048.0},
                },
                "prior": {
                    "wall_ms": {"median": generation_ms / 2.0},
                    "throughput_per_s": {"median": 2000.0 / generation_ms},
                    "peak_allocated_bytes": {"median": 512.0},
                    "peak_reserved_bytes": {"median": 1536.0},
                },
                "training_step": {
                    "wall_ms": {"median": generation_ms * 2.0},
                    "throughput_per_s": {"median": 500.0 / generation_ms},
                    "peak_allocated_bytes": {"median": 3072.0},
                    "peak_reserved_bytes": {"median": 4096.0},
                    "phases": {
                        "phases_ms": {
                            "generation": {"median": generation_ms / 3.0},
                            "preparation": {"median": generation_ms / 4.0},
                        },
                        "forward_counts": {},
                        "forward_times_ms": {},
                        "forward_phase_counts": {},
                        "forward_phase_times_ms": {},
                        "raw_samples": [],
                    },
                    "quality": {
                        "validity_rate": {"median": 0.75},
                        "uniqueness_rate": {"median": 0.5},
                        "synthesizable_rate": {"median": 1.0},
                        "candidate_yield": {"median": 0.375},
                        "acquisition_mean": {"median": 2.5},
                        "acquisition_max": {"median": 4.0},
                        "fidelity_frequencies": {
                            "1": {"median": 0.5},
                            "2": {"median": 0.5},
                        },
                    },
                },
                "policy_vectorization": {
                    "batched": {"wall_ms": {"median": generation_ms / 5.0}},
                    "per_sequence": {"wall_ms": {"median": generation_ms / 2.5}},
                    "speedup_per_sequence_over_batched": 2.0,
                },
                "prior_cache": {"hits": 2, "misses": 1},
                "gpu_utilization": {
                    "sm_average_percent": 80.0,
                    "memory_average_percent": 40.0,
                },
                "optimization": {
                    "precision_dtype": "bfloat16",
                    "feature_map_redraw_count": 0,
                    "compile_cold_start_s": 1.5,
                    "compile_graph_count": 2,
                    "compile_graph_breaks": 1,
                    "compile_recompilations": 3,
                },
            }
        },
        "quality": {"checked": False},
    }


def test_build_benchmark_namespace_forwards_seed() -> None:
    run_spec = experiments.RunSpec(
        run_id="optimized-baseline__quick__seed7__batch8",
        stage="optimized-baseline",
        implementation="optimized",
        profile="quick",
        seed=7,
        batch_size=8,
        sequence_length=64,
        max_length=64,
        warmup=1,
        iterations=3,
        smoke_steps=0,
        real_model=False,
        model_name="model",
        tokenizer_name="tokenizer",
        cache_dir="cache",
        hidden_size=16,
        vocabulary_size=32,
        gpu_utilization_trace=False,
    )

    namespace = experiments.build_benchmark_namespace(run_spec, gpu_trace_path=None)

    assert namespace.seed == 7
    assert namespace.cache_dir == "cache"


def _completed_process_for_run(command: list[str]) -> subprocess.CompletedProcess[str]:
    stage = command[command.index("--stage") + 1]
    implementation = command[command.index("--implementation") + 1]
    payload = {
        "run_id": command[command.index("--run-id") + 1],
        "stage": stage,
        "benchmark": _benchmark_payload(
            implementation=implementation,
            generation_ms=12.0 if implementation == "reference" else 6.0,
        ),
    }
    if implementation == "optimized":
        payload["benchmark"]["quality"] = {
            "checked": True,
            "generation_matches": True,
            "fidelity_matches": False,
        }
    return subprocess.CompletedProcess(
        args=command,
        returncode=0,
        stdout=json.dumps(payload),
        stderr="worker stderr\n",
    )


def test_execute_run_preserves_stdout_stderr_and_raw_json(tmp_path: Path) -> None:
    run_spec = experiments.RunSpec(
        run_id="reference__quick__seed0__batch8",
        stage="reference",
        implementation="reference",
        profile="quick",
        seed=0,
        batch_size=8,
        sequence_length=64,
        max_length=64,
        warmup=1,
        iterations=3,
        smoke_steps=0,
        real_model=False,
        model_name="model",
        tokenizer_name="tokenizer",
        cache_dir=None,
        hidden_size=16,
        vocabulary_size=32,
        gpu_utilization_trace=False,
    )

    record = experiments.execute_run(
        run_spec,
        output_dir=tmp_path,
        subprocess_runner=lambda *args, **kwargs: _completed_process_for_run(args[0]),
    )

    assert record["status"] == "success"
    assert (
        json.loads(Path(record["benchmark_path"]).read_text(encoding="utf-8"))["stage"]
        == "reference"
    )
    assert Path(record["stdout_path"]).read_text(encoding="utf-8").startswith("{")
    assert Path(record["stderr_path"]).read_text(encoding="utf-8") == "worker stderr\n"
    assert not (Path(record["benchmark_path"]).parent / ".raw_result.json.tmp").exists()


def test_run_experiments_writes_manifest_and_resume_skips_completed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiments,
        "validate_cuda_environment",
        lambda device: {"device": device},
    )
    first_args = _make_args(tmp_path)
    call_count = {"value": 0}

    def fake_runner(*args, **kwargs):
        call_count["value"] += 1
        return _completed_process_for_run(args[0])

    assert experiments.run_experiments(first_args, subprocess_runner=fake_runner) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads(
        (tmp_path / experiments.CHECKPOINT_NAME).read_text(encoding="utf-8")
    )
    assert (
        manifest["stage_registry"]["optimized-baseline"]["implementation"]
        == "optimized"
    )
    assert manifest["stage_registry"]["optimized-baseline"]["ablations"] == {
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
    assert sorted(checkpoint["completed_runs"]) == [
        "optimized-baseline__quick__seed0__batch8",
        "reference__quick__seed0__batch8",
    ]
    assert call_count["value"] == 2

    resume_args = _make_args(tmp_path, resume=True)

    def unexpected_runner(*args, **kwargs):
        raise AssertionError("resume should skip completed runs")

    assert (
        experiments.run_experiments(resume_args, subprocess_runner=unexpected_runner)
        == 0
    )
    assert len((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_run_experiments_records_failures_for_malformed_worker_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiments,
        "validate_cuda_environment",
        lambda device: {"device": device},
    )
    args = _make_args(tmp_path)

    def fake_runner(*args, **kwargs):
        command = args[0]
        implementation = command[command.index("--implementation") + 1]
        if implementation == "optimized":
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="not-json",
                stderr="boom\n",
            )
        return _completed_process_for_run(command)

    exit_code = experiments.run_experiments(args, subprocess_runner=fake_runner)

    failures = json.loads((tmp_path / "failures.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert len(failures) == 1
    assert "Malformed worker output" in failures[0]["error"]
    assert summary["totals"] == {"records": 2, "successful": 1, "failed": 1}


def test_aggregate_run_records_only_uses_matching_successful_samples() -> None:
    records = [
        {
            "run_id": "reference_seed0",
            "stage": "reference",
            "profile": "quick",
            "batch_size": 8,
            "seed": 0,
            "status": "success",
            "real_model": False,
            "model_name": "model",
            "tokenizer_name": "tokenizer",
            "sequence_length": 64,
            "max_length": 64,
            "benchmark": _benchmark_payload(
                implementation="reference",
                generation_ms=12.0,
            )["implementations"]["reference"],
        },
        {
            "run_id": "reference_seed1",
            "stage": "reference",
            "profile": "quick",
            "batch_size": 8,
            "seed": 1,
            "status": "success",
            "real_model": False,
            "model_name": "model",
            "tokenizer_name": "tokenizer",
            "sequence_length": 64,
            "max_length": 64,
            "benchmark": _benchmark_payload(
                implementation="reference",
                generation_ms=10.0,
            )["implementations"]["reference"],
        },
        {
            "run_id": "optimized_seed0",
            "stage": "optimized-baseline",
            "profile": "quick",
            "batch_size": 8,
            "seed": 0,
            "status": "success",
            "real_model": False,
            "model_name": "model",
            "tokenizer_name": "tokenizer",
            "sequence_length": 64,
            "max_length": 64,
            "benchmark": _benchmark_payload(
                implementation="optimized",
                generation_ms=6.0,
            )["implementations"]["optimized"],
        },
        {
            "run_id": "optimized_seed1_failed",
            "stage": "optimized-baseline",
            "profile": "quick",
            "batch_size": 8,
            "seed": 1,
            "status": "failure",
            "real_model": False,
            "model_name": "model",
            "tokenizer_name": "tokenizer",
            "sequence_length": 64,
            "max_length": 64,
        },
    ]

    summary = experiments.aggregate_run_records(records)
    optimized_group = next(
        group for group in summary["groups"] if group["stage"] == "optimized-baseline"
    )

    assert optimized_group["speedups_vs_reference"]["matched_samples"] == 1
    assert (
        optimized_group["speedups_vs_reference"]["generation_wall_ms"]["median"] == 2.0
    )


def test_aggregate_run_records_preserves_quality_and_cache_metadata() -> None:
    records = [
        {
            "run_id": "optimized_seed0",
            "stage": "optimized-baseline",
            "profile": "quick",
            "batch_size": 8,
            "seed": 0,
            "status": "success",
            "real_model": False,
            "model_name": "model",
            "tokenizer_name": "tokenizer",
            "cache_dir": "cache",
            "sequence_length": 64,
            "max_length": 64,
            "quality": {
                "checked": True,
                "generation_matches": True,
                "fidelity_matches": False,
            },
            "benchmark": _benchmark_payload(
                implementation="optimized",
                generation_ms=6.0,
            )["implementations"]["optimized"],
        }
    ]

    summary = experiments.aggregate_run_records(records)
    group = summary["groups"][0]

    assert group["model"]["cache_dir"] == "cache"
    assert group["ablations"]["prior_cache_enabled"] is True
    assert group["quality"]["checked_runs"] == 1
    assert group["quality"]["generation_match_rate"] == 1.0
    assert group["quality"]["fidelity_match_rate"] == 0.0
    assert group["quality_metrics"]["validity_rate"]["median"] == 0.75
    assert group["quality_metrics"]["uniqueness_rate"]["median"] == 0.5
    assert group["quality_metrics"]["synthesizable_rate"]["median"] == 1.0
    assert group["quality_metrics"]["candidate_yield"]["median"] == 0.375
    assert group["quality_metrics"]["acquisition_mean"]["median"] == 2.5
    assert group["quality_metrics"]["fidelity_frequencies"]["1"]["median"] == 0.5
    assert group["cuda_utilization"]["sm_average_percent"]["median"] == 80.0
    assert group["optimization_metrics"]["compile_cold_start_s"]["median"] == 1.5
    assert group["optimization_metrics"]["compile_graph_breaks"]["median"] == 1.0


def test_atomic_write_json_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"

    experiments.atomic_write_json(path, {"value": 1})
    experiments.atomic_write_json(path, {"value": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}
    assert not (tmp_path / ".payload.json.tmp").exists()


def test_validate_cuda_environment_raises_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiments.torch.cuda, "is_available", lambda: False)

    with pytest.raises(experiments.CUDAValidationError):
        experiments.validate_cuda_environment()
