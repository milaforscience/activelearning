"""Tests for the optimized S3-GFN benchmark entrypoint."""

from __future__ import annotations

import argparse

import pytest

from benchmarks import optimized_s3gfn_benchmark as benchmark


def test_run_benchmark_forwards_seed_to_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        benchmark,
        "_load_fake_templates",
        lambda **kwargs: {"templates": kwargs},
    )
    monkeypatch.setattr(
        benchmark,
        "_collect_environment_metadata",
        lambda device: {"device": str(device)},
    )

    def fake_benchmark_implementation(
        templates,
        *,
        implementation,
        device,
        batch_size,
        sequence_length,
        max_length,
        warmup,
        iterations,
        smoke_steps,
        seed,
    ):
        captured[implementation] = {
            "templates": templates,
            "device": str(device),
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "max_length": max_length,
            "warmup": warmup,
            "iterations": iterations,
            "smoke_steps": smoke_steps,
            "seed": seed,
        }
        return (
            {
                "generation": {"wall_ms": {"median": 1.0}},
                "prior": {"wall_ms": {"median": 1.0}},
                "training_step": {
                    "wall_ms": {"median": 1.0},
                    "phases": {
                        "phases_ms": {},
                        "forward_counts": {},
                        "forward_times_ms": {},
                        "forward_phase_counts": {},
                        "forward_phase_times_ms": {},
                        "raw_samples": [],
                    },
                },
                "policy_vectorization": {"speedup_per_sequence_over_batched": 1.0},
                "prior_cache": {"hits": 0, "misses": 0, "enabled": False},
                "smoke_test": None,
            },
            {"input_ids": [[1, 2]], "fidelity_indices": [0], "smiles": ["C"]},
        )

    monkeypatch.setattr(
        benchmark, "_benchmark_implementation", fake_benchmark_implementation
    )

    args = benchmark.build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--implementation",
            "comparison",
            "--seed",
            "11",
            "--cache-dir",
            "cache-dir",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ]
    )

    result = benchmark.run_benchmark(args)

    assert captured["reference"]["seed"] == 11
    assert captured["optimized"]["seed"] == 11
    assert result["config"]["seed"] == 11
    assert result["config"]["cache_dir"] == "cache-dir"


def test_run_benchmark_rejects_cuda_without_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)

    args = benchmark.build_parser().parse_args(
        ["--device", "cuda", "--warmup", "0", "--iterations", "1"]
    )

    with pytest.raises(ValueError, match="requires CUDA availability"):
        benchmark.run_benchmark(args)


def test_build_parser_accepts_seed_and_cache_dir() -> None:
    args = benchmark.build_parser().parse_args(
        ["--seed", "5", "--cache-dir", "cache", "--warmup", "0", "--iterations", "1"]
    )

    assert isinstance(args, argparse.Namespace)
    assert args.seed == 5
    assert args.cache_dir == "cache"
