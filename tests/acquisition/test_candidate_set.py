"""Tests for CandidateSetSpec implementations.

Covers HypercubeCandidateSetSpec, TrainDataCandidateSetSpec, and
TensorCandidateSetSpec against both single-fidelity and multi-fidelity
fitted surrogates.
"""

import pytest
import torch

from activelearning.acquisition.botorch.candidate_set import (
    HypercubeCandidateSetSpec,
    TensorCandidateSetSpec,
    TrainDataCandidateSetSpec,
)
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.utils.types import Candidate, Observation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_fidelity_observations() -> list[Observation]:
    return [
        Observation(x=[1.0, 2.0], y=5.0),
        Observation(x=[3.0, 4.0], y=7.0),
        Observation(x=[5.0, 6.0], y=9.0),
    ]


@pytest.fixture()
def multi_fidelity_observations() -> list[Observation]:
    return [
        Observation(x=[1.0, 2.0], y=5.0, fidelity=0),
        Observation(x=[3.0, 4.0], y=7.0, fidelity=1),
        Observation(x=[5.0, 6.0], y=9.0, fidelity=1),
        Observation(x=[1.0, 2.0], y=4.5, fidelity=0),
    ]


@pytest.fixture()
def fitted_sf_surrogate(
    single_fidelity_observations: list[Observation],
) -> BoTorchGPSurrogate:
    surrogate = BoTorchGPSurrogate()
    surrogate.fit(single_fidelity_observations)
    return surrogate


@pytest.fixture()
def fitted_mf_surrogate(
    multi_fidelity_observations: list[Observation],
) -> BoTorchGPSurrogate:
    surrogate = BoTorchGPSurrogate()
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(multi_fidelity_observations)
    return surrogate


# ---------------------------------------------------------------------------
# HypercubeCandidateSetSpec
# ---------------------------------------------------------------------------


class TestHypercubeCandidateSetSpec:
    """Tests for HypercubeCandidateSetSpec."""

    BOUNDS_2D = [(0.0, 5.0), (0.0, 7.0)]
    N = 20

    # --- construction validation -------------------------------------------

    def test_empty_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="bounds must not be empty"):
            HypercubeCandidateSetSpec(bounds=[], n_points=10)

    def test_invalid_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="lower >= upper"):
            HypercubeCandidateSetSpec(bounds=[(5.0, 1.0)], n_points=10)

    def test_equal_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="lower >= upper"):
            HypercubeCandidateSetSpec(bounds=[(3.0, 3.0)], n_points=10)

    def test_nonpositive_n_points_raises(self) -> None:
        with pytest.raises(ValueError, match="n_points must be > 0"):
            HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=0)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy must be"):
            HypercubeCandidateSetSpec(
                bounds=self.BOUNDS_2D,
                n_points=10,
                strategy="sobol",  # type: ignore[arg-type]
            )

    # --- single-fidelity builds --------------------------------------------

    def test_uniform_sf_shape(self, fitted_sf_surrogate: BoTorchGPSurrogate) -> None:
        """Output shape is (n_points, feature_dims) with no fidelity column."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_sf_surrogate)
        assert result.shape == (self.N, 2)

    def test_lhs_sf_shape(self, fitted_sf_surrogate: BoTorchGPSurrogate) -> None:
        spec = HypercubeCandidateSetSpec(
            bounds=self.BOUNDS_2D, n_points=self.N, strategy="lhs"
        )
        result = spec.build(fitted_sf_surrogate)
        assert result.shape == (self.N, 2)

    def test_sf_no_fidelity_column_when_not_passed(
        self, fitted_sf_surrogate: BoTorchGPSurrogate
    ) -> None:
        """When target_fidelity_value is not passed, no fidelity column is appended."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_sf_surrogate)
        assert result.dtype == torch.float64
        assert result.ndim == 2
        assert result.shape[1] == 2

    # --- multi-fidelity builds ---------------------------------------------

    def test_mf_shape_includes_fidelity_column(
        self, fitted_mf_surrogate: BoTorchGPSurrogate
    ) -> None:
        """When target_fidelity_value is passed, the output has an extra fidelity column."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_mf_surrogate, target_fidelity_value=1.0)
        assert result.shape == (self.N, 3)  # 2 features + 1 fidelity

    def test_mf_target_fidelity_value_high(
        self, fitted_mf_surrogate: BoTorchGPSurrogate
    ) -> None:
        """The passed target_fidelity_value is set correctly in the fidelity column."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_mf_surrogate, target_fidelity_value=1.0)
        fid_column = result[:, -1]
        assert torch.allclose(
            fid_column, torch.full((self.N,), 1.0, dtype=torch.float64)
        )

    def test_mf_target_fidelity_value_low(
        self, fitted_mf_surrogate: BoTorchGPSurrogate
    ) -> None:
        """A lower target_fidelity_value is correctly reflected in the fidelity column."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_mf_surrogate, target_fidelity_value=0.5)
        fid_column = result[:, -1]
        assert torch.allclose(
            fid_column, torch.full((self.N,), 0.5, dtype=torch.float64)
        )

    # --- dtype & two-run independence --------------------------------------

    def test_output_dtype_is_float64(
        self, fitted_sf_surrogate: BoTorchGPSurrogate
    ) -> None:
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_sf_surrogate)
        assert result.dtype == torch.float64

    def test_two_builds_differ(self, fitted_sf_surrogate: BoTorchGPSurrogate) -> None:
        """Each call to build() samples fresh points (stochastic)."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        r1 = spec.build(fitted_sf_surrogate)
        r2 = spec.build(fitted_sf_surrogate)
        assert not torch.allclose(r1, r2)

    # --- update is a no-op ---------------------------------------------------

    def test_update_is_noop(
        self, single_fidelity_observations: list[Observation]
    ) -> None:
        """update() should be accepted without error."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        spec.update(single_fidelity_observations)  # should not raise


# ---------------------------------------------------------------------------
# TrainDataCandidateSetSpec
# ---------------------------------------------------------------------------


class TestTrainDataCandidateSetSpec:
    """Tests for TrainDataCandidateSetSpec."""

    def test_sf_shape_and_dtype(
        self,
        fitted_sf_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        """Output shape and dtype match the encoded candidates."""
        spec = TrainDataCandidateSetSpec()
        spec.update(single_fidelity_observations)
        result = spec.build(fitted_sf_surrogate)
        assert result.shape == (len(single_fidelity_observations), 2)
        assert result.dtype == torch.float64

    def test_mf_shape_includes_fidelity_column(
        self,
        fitted_mf_surrogate: BoTorchGPSurrogate,
        multi_fidelity_observations: list[Observation],
    ) -> None:
        """In MF mode the encoded output includes the fidelity column."""
        spec = TrainDataCandidateSetSpec()
        spec.update(multi_fidelity_observations)
        result = spec.build(fitted_mf_surrogate)
        assert result.shape == (
            len(multi_fidelity_observations),
            3,
        )  # 2 features + 1 fidelity
        assert result.dtype == torch.float64

    def test_matches_encode_candidates(
        self,
        fitted_sf_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        """Output equals encoding the same candidates directly via the surrogate."""
        spec = TrainDataCandidateSetSpec()
        spec.update(single_fidelity_observations)
        result = spec.build(fitted_sf_surrogate)
        candidates = [
            Candidate(x=obs.x, fidelity=obs.fidelity)
            for obs in single_fidelity_observations
        ]
        expected = fitted_sf_surrogate.encode_candidates(candidates)
        assert torch.allclose(result, expected)

    def test_raises_before_update(
        self, fitted_sf_surrogate: BoTorchGPSurrogate
    ) -> None:
        """build() raises RuntimeError when update() has not been called."""
        spec = TrainDataCandidateSetSpec()
        with pytest.raises(RuntimeError, match="no cached candidates"):
            spec.build(fitted_sf_surrogate)

    def test_update_replaces_cached_candidates(
        self,
        fitted_sf_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        """A second call to update() replaces the previously cached observations."""
        spec = TrainDataCandidateSetSpec()
        spec.update(single_fidelity_observations)
        new_obs = [Observation(x=[0.1, 0.2], y=1.0)]
        spec.update(new_obs)
        result = spec.build(fitted_sf_surrogate)
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# TensorCandidateSetSpec
# ---------------------------------------------------------------------------


class TestTensorCandidateSetSpec:
    """Tests for TensorCandidateSetSpec."""

    def test_returns_same_tensor(self, fitted_sf_surrogate: BoTorchGPSurrogate) -> None:
        """build() returns the exact tensor object passed at construction."""
        tensor = torch.rand(10, 2, dtype=torch.float64)
        spec = TensorCandidateSetSpec(tensor)
        result = spec.build(fitted_sf_surrogate)
        assert result is tensor

    def test_surrogate_not_used(self) -> None:
        """build() does not require a fitted surrogate — tensor is returned as-is."""
        tensor = torch.zeros(5, 3, dtype=torch.float64)
        spec = TensorCandidateSetSpec(tensor)
        # Pass an unfitted surrogate to confirm it's never accessed
        surrogate = BoTorchGPSurrogate()
        result = spec.build(surrogate)
        assert torch.equal(result, tensor)

    def test_update_is_noop(
        self, single_fidelity_observations: list[Observation]
    ) -> None:
        """update() should be accepted without error."""
        tensor = torch.zeros(5, 2, dtype=torch.float64)
        spec = TensorCandidateSetSpec(tensor)
        spec.update(single_fidelity_observations)  # should not raise

    def test_1d_tensor_raises(self) -> None:
        """A 1-D tensor is rejected at construction time."""
        with pytest.raises(ValueError, match="2-D"):
            TensorCandidateSetSpec(torch.rand(10))

    def test_3d_tensor_raises(self) -> None:
        """A 3-D tensor is rejected at construction time."""
        with pytest.raises(ValueError, match="2-D"):
            TensorCandidateSetSpec(torch.rand(4, 10, 2))
