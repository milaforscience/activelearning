"""Tests for the optimized S3-GFN sampler configuration."""

from __future__ import annotations

from pydantic import TypeAdapter
import pytest

from activelearning.sampler.config import OptimizedS3GFNSamplerConfig, SamplerConfig


def test_optimized_s3gfn_config_round_trips_through_sampler_union() -> None:
    config = TypeAdapter(SamplerConfig).validate_python(
        {
            "type": "OptimizedS3GFNSampler",
            "n_samples": 4,
            "fidelities": [1, 2],
            "prior_cache_capacity": 4096,
        }
    )

    assert isinstance(config, OptimizedS3GFNSamplerConfig)
    assert config.fidelities == [1, 2]
    assert config.prior_cache_capacity == 4096


def test_optimized_s3gfn_config_defaults_match_stage0_constructor_surface() -> None:
    config = OptimizedS3GFNSamplerConfig(n_samples=4)

    assert config.precision == "fp32"
    assert config.fixed_feature_maps is False
    assert config.parallel_cuda_rollout is False
    assert config.compile_mode == "eager"
    assert config.deferred_sync is False
    assert config.carried_prior_scores is False
    assert config.overlap_online_prior is False
    assert config.combined_aux_policy_batch is False
    assert config.stop_check_interval == 1
    assert config.prior_cache_enabled is True
    assert config.prior_cache_capacity == 8192


def test_optimized_s3gfn_config_builds_sampler_lazily() -> None:
    sampler = OptimizedS3GFNSamplerConfig(
        n_samples=3,
        fidelities=[1, 2],
        batch_size=5,
        replay_batch_size=4,
    ).build()

    assert type(sampler).__name__ == "OptimizedS3GFNSampler"
    assert sampler.n_samples == 3
    assert sampler.fidelities == (1, 2)
    assert sampler.batch_size == 5
    assert sampler.replay_batch_size == 4


def test_optimized_s3gfn_config_build_passes_through_future_constructor_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _PlaceholderOptimizedSampler:
        def __init__(
            self,
            *,
            n_samples: int,
            fidelities: list[int],
            precision: str,
            compile_mode: str,
            fixed_feature_maps: bool,
            parallel_cuda_rollout: bool,
            deferred_sync: bool,
            carried_prior_scores: bool,
            overlap_online_prior: bool,
            combined_aux_policy_batch: bool,
            stop_check_interval: int,
            prior_cache_enabled: bool,
            prior_cache_capacity: int,
        ) -> None:
            captured.update(
                {
                    "n_samples": n_samples,
                    "fidelities": fidelities,
                    "precision": precision,
                    "compile_mode": compile_mode,
                    "fixed_feature_maps": fixed_feature_maps,
                    "parallel_cuda_rollout": parallel_cuda_rollout,
                    "deferred_sync": deferred_sync,
                    "carried_prior_scores": carried_prior_scores,
                    "overlap_online_prior": overlap_online_prior,
                    "combined_aux_policy_batch": combined_aux_policy_batch,
                    "stop_check_interval": stop_check_interval,
                    "prior_cache_enabled": prior_cache_enabled,
                    "prior_cache_capacity": prior_cache_capacity,
                }
            )

    import activelearning.sampler.optimized_s3gfn.sampler as optimized_sampler_module

    monkeypatch.setattr(
        optimized_sampler_module,
        "OptimizedS3GFNSampler",
        _PlaceholderOptimizedSampler,
    )

    config = OptimizedS3GFNSamplerConfig(
        n_samples=4,
        fidelities=[1, 2],
        precision="cuda_auto",
        fixed_feature_maps=True,
        parallel_cuda_rollout=True,
        compile_mode="max-autotune-no-cudagraphs",
        deferred_sync=True,
        carried_prior_scores=True,
        overlap_online_prior=True,
        combined_aux_policy_batch=True,
        stop_check_interval=3,
        prior_cache_enabled=False,
        prior_cache_capacity=256,
    )

    sampler = config.build()

    assert isinstance(sampler, _PlaceholderOptimizedSampler)
    assert captured == {
        "n_samples": 4,
        "fidelities": [1, 2],
        "precision": "cuda_auto",
        "compile_mode": "max-autotune-no-cudagraphs",
        "fixed_feature_maps": True,
        "parallel_cuda_rollout": True,
        "deferred_sync": True,
        "carried_prior_scores": True,
        "overlap_online_prior": True,
        "combined_aux_policy_batch": True,
        "stop_check_interval": 3,
        "prior_cache_enabled": False,
        "prior_cache_capacity": 256,
    }
