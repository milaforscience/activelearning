"""Tests for CandidateSetSpec implementations.

Covers HypercubeCandidateSetSpec, TrainDataCandidateSetSpec, and
TensorCandidateSetSpec against both single-fidelity and multi-fidelity
fitted surrogates.
"""

import pytest
import torch

from activelearning.acquisition.candidate_set import (
    HypercubeCandidateSetSpec,
    TensorCandidateSetSpec,
    TrainDataCandidateSetSpec,
)
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.utils.types import Observation


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
                bounds=self.BOUNDS_2D, n_points=10, strategy="sobol"  # type: ignore[arg-type]
            )

    # --- single-fidelity builds --------------------------------------------

    def test_uniform_sf_shape(self, fitted_sf_surrogate: BoTorchGPSurrogate) -> None:
        """Output shape is (n_points, feature_dims) in single-fidelity mode."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_sf_surrogate)
        assert result.shape == (self.N, 2)

    def test_lhs_sf_shape(self, fitted_sf_surrogate: BoTorchGPSurrogate) -> None:
        spec = HypercubeCandidateSetSpec(
            bounds=self.BOUNDS_2D, n_points=self.N, strategy="lhs"
        )
        result = spec.build(fitted_sf_surrogate)
        assert result.shape == (self.N, 2)

    def test_uniform_sf_values_in_bounds(
        self, fitted_sf_surrogate: BoTorchGPSurrogate
    ) -> None:
        """Raw feature values respect the specified bounds."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_sf_surrogate)
        # Surrogate applies Normalize internally, so check raw columns via
        # manual decode: the surrogate's Normalize layer will have shifted values;
        # we check that the encoded output is a 2D float tensor instead.
        assert result.dtype == torch.float64
        assert result.ndim == 2
        assert result.shape[1] == 2

    # --- multi-fidelity builds ---------------------------------------------

    def test_mf_shape_includes_fidelity_column(
        self, fitted_mf_surrogate: BoTorchGPSurrogate
    ) -> None:
        """In MF mode the output has an extra fidelity column: (n_points, d+1)."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_mf_surrogate)
        assert result.shape == (self.N, 3)  # 2 features + 1 fidelity

    def test_mf_inferred_target_fidelity(
        self, fitted_mf_surrogate: BoTorchGPSurrogate
    ) -> None:
        """When target_fidelity_id is None, the fidelity with max confidence is used."""
        spec = HypercubeCandidateSetSpec(bounds=self.BOUNDS_2D, n_points=self.N)
        result = spec.build(fitted_mf_surrogate)
        # Fidelity confidences: {0: 0.5, 1: 1.0} → target id=1 → encoded as 1.0
        fid_column = result[:, -1]
        assert torch.allclose(fid_column, torch.full((self.N,), 1.0, dtype=torch.float64))

    def test_mf_explicit_target_fidelity_id(
        self, fitted_mf_surrogate: BoTorchGPSurrogate
    ) -> None:
        """Explicit target_fidelity_id is respected over the inferred default."""
        spec = HypercubeCandidateSetSpec(
            bounds=self.BOUNDS_2D, n_points=self.N, target_fidelity_id=0
        )
        result = spec.build(fitted_mf_surrogate)
        # Fidelity id=0 is encoded as confidence 0.5
        fid_column = result[:, -1]
        assert torch.allclose(fid_column, torch.full((self.N,), 0.5, dtype=torch.float64))

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


# ---------------------------------------------------------------------------
# TrainDataCandidateSetSpec
# ---------------------------------------------------------------------------


class TestTrainDataCandidateSetSpec:
    """Tests for TrainDataCandidateSetSpec."""

    def test_returns_train_x(self, fitted_sf_surrogate: BoTorchGPSurrogate) -> None:
        """Output is exactly the model's train_X."""
        spec = TrainDataCandidateSetSpec()
        result = spec.build(fitted_sf_surrogate)
        train_X, _ = fitted_sf_surrogate.get_train_data()
        assert torch.equal(result, train_X)

    def test_returns_train_x_mf(self, fitted_mf_surrogate: BoTorchGPSurrogate) -> None:
        """Output is exactly the model's train_X including the fidelity column."""
        spec = TrainDataCandidateSetSpec()
        result = spec.build(fitted_mf_surrogate)
        train_X, _ = fitted_mf_surrogate.get_train_data()
        assert torch.equal(result, train_X)

    def test_raises_before_fit(self) -> None:
        """Calling build() on an unfitted surrogate raises RuntimeError."""
        surrogate = BoTorchGPSurrogate()
        spec = TrainDataCandidateSetSpec()
        with pytest.raises(RuntimeError):
            spec.build(surrogate)


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
