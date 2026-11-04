"""Tests for standalone sampler configuration models."""

from activelearning.sampler.config import (
    ExactGridSamplerConfig,
    HypercubeSamplerConfig,
    PoolFileSamplerConfig,
)
from activelearning.utils.types import DEFAULT_FIDELITY


def test_hypercube_config_build_uses_default_fidelity() -> None:
    """An omitted fidelity setting builds a single-fidelity sampler."""
    sampler = HypercubeSamplerConfig(
        bounds=[(0.0, 1.0)],
        num_samples=2,
    ).build()

    assert sampler._fidelity_levels == [DEFAULT_FIDELITY]


def test_pool_file_config_build_uses_default_fidelity(tmp_path) -> None:
    """An omitted fidelity setting builds a single-fidelity sampler."""
    sampler = PoolFileSamplerConfig(
        candidate_pool_file=tmp_path / "candidates.txt",
        num_samples=2,
    ).build()

    assert sampler._fidelity_levels == [DEFAULT_FIDELITY]


def test_exact_grid_config_build_uses_default_fidelity() -> None:
    """An omitted fidelity setting builds a single-fidelity sampler."""
    sampler = ExactGridSamplerConfig(
        bounds=[(0.0, 1.0)],
        points_per_dimension=[2],
    ).build()

    assert sampler._fidelity_levels == [DEFAULT_FIDELITY]
