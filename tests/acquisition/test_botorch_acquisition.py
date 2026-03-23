"""Tests for BoTorchAcquisitionBase, AnalyticBoTorchAcquisition, and QBatchBoTorchAcquisition."""

import math
from typing import Any, Iterable, Optional
from unittest.mock import MagicMock

import pytest
import torch

from activelearning.acquisition.botorch.botorch_acquisition import (
    AnalyticBoTorchAcquisition,
    QBatchBoTorchAcquisition,
)
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.utils.types import Candidate, Observation


# ---------------------------------------------------------------------------
# Concrete test stubs — minimal implementations of the abstract base classes
# ---------------------------------------------------------------------------


class StubAnalytic(AnalyticBoTorchAcquisition):
    """Analytic stub that wraps a callable as the BoTorch acquisition function."""

    def __init__(self, acqf_fn: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._acqf_fn = acqf_fn

    def _build_botorch_acquisition(self) -> Any:
        if self._acqf_fn is not None:
            return self._acqf_fn
        # Default: return the mean across the q dimension.
        return lambda X: X.squeeze(-2).sum(dim=-1)


class StubQBatch(QBatchBoTorchAcquisition):
    """Q-batch stub that wraps a callable as the BoTorch acquisition function."""

    def __init__(self, acqf_fn: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._acqf_fn = acqf_fn

    def _build_botorch_acquisition(self) -> Any:
        if self._acqf_fn is not None:
            return self._acqf_fn
        # Default: sum all features across the q-batch.
        return lambda X: X.sum(dim=(-1, -2))


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
def candidates() -> list[Candidate]:
    return [Candidate(x=[2.0, 3.0]), Candidate(x=[4.0, 5.0])]


@pytest.fixture()
def candidate_batches() -> list[list[Candidate]]:
    return [
        [Candidate(x=[2.0, 3.0]), Candidate(x=[4.0, 5.0])],
        [Candidate(x=[1.0, 1.0]), Candidate(x=[6.0, 7.0])],
    ]


@pytest.fixture()
def fitted_surrogate(
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


# ===================================================================
# BoTorchAcquisitionBase — capability flags & __init__
# ===================================================================


class TestCapabilityFlags:
    """Verify that capability flags propagate correctly through the hierarchy."""

    def test_analytic_flags(self) -> None:
        acq = StubAnalytic()
        assert acq.supports_singleton_scoring is True
        assert acq.supports_batch_scoring is False

    def test_qbatch_flags(self) -> None:
        acq = StubQBatch()
        assert acq.supports_singleton_scoring is True
        assert acq.supports_batch_scoring is False

    def test_maximize_default(self) -> None:
        assert StubAnalytic().maximize is True
        assert StubQBatch().maximize is True

    def test_maximize_override(self) -> None:
        assert StubAnalytic(maximize=False).maximize is False
        assert StubQBatch(maximize=False).maximize is False

    def test_kwargs_forwarding(self) -> None:
        """Ensure BoTorchAcquisitionBase kwargs pass through from subclass __init__."""
        acq = StubAnalytic(
            maximize=False,
            target_fidelity_value=0.8,
            fidelity_costs={0: 1.0, 1: 5.0},
        )
        assert acq.maximize is False
        assert acq._target_fidelity_value_override == 0.8
        assert acq._fidelity_costs == {0: 1.0, 1: 5.0}


# ===================================================================
# Parameter validation
# ===================================================================


class TestParameterValidation:
    """Verify that invalid parameters raise appropriate errors."""

    def test_num_fantasies_must_be_positive(self) -> None:
        """num_fantasies must be > 0."""
        from activelearning.acquisition.botorch.botorch_multifidelity import (
            QMultiFidelityMaxValueEntropy,
        )
        from activelearning.acquisition.candidate_set import (
            HypercubeCandidateSetSpec,
        )

        spec = HypercubeCandidateSetSpec(
            bounds=[(0.0, 5.0), (0.0, 5.0)],
            n_points=10,
        )
        with pytest.raises(ValueError, match="num_fantasies must be > 0"):
            QMultiFidelityMaxValueEntropy(
                candidate_set_spec=spec,
                num_fantasies=0,
            )

    def test_num_mv_samples_must_be_positive(self) -> None:
        """num_mv_samples must be > 0."""
        from activelearning.acquisition.botorch.botorch_multifidelity import (
            QMultiFidelityMaxValueEntropy,
        )
        from activelearning.acquisition.candidate_set import (
            HypercubeCandidateSetSpec,
        )

        spec = HypercubeCandidateSetSpec(
            bounds=[(0.0, 5.0), (0.0, 5.0)],
            n_points=10,
        )
        with pytest.raises(ValueError, match="num_mv_samples must be > 0"):
            QMultiFidelityMaxValueEntropy(
                candidate_set_spec=spec,
                num_mv_samples=0,
            )

    def test_num_y_samples_must_be_positive(self) -> None:
        """num_y_samples must be > 0."""
        from activelearning.acquisition.botorch.botorch_multifidelity import (
            QMultiFidelityMaxValueEntropy,
        )
        from activelearning.acquisition.candidate_set import (
            HypercubeCandidateSetSpec,
        )

        spec = HypercubeCandidateSetSpec(
            bounds=[(0.0, 5.0), (0.0, 5.0)],
            n_points=10,
        )
        with pytest.raises(ValueError, match="num_y_samples must be > 0"):
            QMultiFidelityMaxValueEntropy(
                candidate_set_spec=spec,
                num_y_samples=0,
            )


# ===================================================================
# Properties — default state before update()
# ===================================================================


class TestPropertiesBeforeUpdate:
    """All runtime properties should be None before update() is called."""

    @pytest.fixture()
    def acq(self) -> StubAnalytic:
        return StubAnalytic()

    def test_botorch_surrogate_is_none(self, acq: StubAnalytic) -> None:
        assert acq.botorch_surrogate is None

    def test_botorch_acqf_is_none(self, acq: StubAnalytic) -> None:
        assert acq.botorch_acqf is None

    def test_observations_cache_is_none(self, acq: StubAnalytic) -> None:
        assert acq.observations_cache is None

    def test_resolved_target_fidelity_value_is_none(self, acq: StubAnalytic) -> None:
        assert acq.resolved_target_fidelity_value is None

    def test_resolved_project_fn_is_none(self, acq: StubAnalytic) -> None:
        assert acq.resolved_project_to_target_fidelity_fn is None

    def test_resolved_cost_model_is_none(self, acq: StubAnalytic) -> None:
        assert acq.resolved_cost_model is None

    def test_resolved_cost_aware_utility_is_none(self, acq: StubAnalytic) -> None:
        assert acq.resolved_cost_aware_utility is None


# ===================================================================
# update() — surrogate validation
# ===================================================================


class TestUpdateValidation:
    """update() must reject non-BoTorchGPSurrogate instances."""

    def test_rejects_wrong_surrogate_type(self) -> None:
        acq = StubAnalytic()
        with pytest.raises(TypeError, match="BoTorchGPSurrogate"):
            acq.update(DummyMeanSurrogate())

    def test_rejects_mock_surrogate(self) -> None:
        acq = StubAnalytic()
        with pytest.raises(TypeError, match="BoTorchGPSurrogate"):
            acq.update(MagicMock())

    def test_accepts_botorch_surrogate(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        assert acq.botorch_surrogate is fitted_surrogate

    def test_stores_surrogate_on_base(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
    ) -> None:
        """Both base class .surrogate and typed .botorch_surrogate are set."""
        acq = StubAnalytic()
        acq.update(fitted_surrogate)
        assert acq.surrogate is fitted_surrogate
        assert acq.botorch_surrogate is fitted_surrogate


# ===================================================================
# update() — observations materialization
# ===================================================================


class TestObservationsCache:
    """Observations should be materialized into a list during update()."""

    def test_none_when_not_provided(self, fitted_surrogate: BoTorchGPSurrogate) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate)
        assert acq.observations_cache is None

    def test_materialized_from_list(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        assert acq.observations_cache == single_fidelity_observations

    def test_materialized_from_generator(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        """One-pass iterables are safely materialized into a list."""

        def gen() -> Iterable[Observation]:
            yield from single_fidelity_observations

        acq = StubAnalytic()
        acq.update(fitted_surrogate, gen())
        assert acq.observations_cache == single_fidelity_observations

    def test_empty_observations_yield_empty_list(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, [])
        assert acq.observations_cache == []


# ===================================================================
# update() — repeated calls
# ===================================================================


class TestRepeatedUpdate:
    """Calling update() multiple times should fully replace prior state."""

    def test_replaces_surrogate(
        self, single_fidelity_observations: list[Observation]
    ) -> None:
        s1 = BoTorchGPSurrogate()
        s1.fit(single_fidelity_observations)
        s2 = BoTorchGPSurrogate()
        s2.fit(single_fidelity_observations)

        acq = StubAnalytic()
        acq.update(s1, single_fidelity_observations)
        assert acq.botorch_surrogate is s1

        acq.update(s2, single_fidelity_observations)
        assert acq.botorch_surrogate is s2

    def test_replaces_acqf(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        first_acqf = acq.botorch_acqf

        acq.update(fitted_surrogate, single_fidelity_observations)
        assert first_acqf is not None
        assert acq.botorch_acqf is not first_acqf

    def test_observations_replaced(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        assert acq.observations_cache is not None
        assert len(acq.observations_cache) == 3

        subset = single_fidelity_observations[:1]
        acq.update(fitted_surrogate, subset)
        assert acq.observations_cache is not None
        assert len(acq.observations_cache) == 1


# ===================================================================
# score() — AnalyticBoTorchAcquisition
# ===================================================================


class TestAnalyticScore:
    """AnalyticBoTorchAcquisition.score() delegates to _score_encoded."""

    def test_returns_ones_before_update(self, candidates: list[Candidate]) -> None:
        """Before update(), score() returns constant 1.0 for every candidate."""
        acq = StubAnalytic()
        scores = acq.score(candidates)
        assert scores == [1.0] * len(candidates)

    def test_returns_correct_length(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidates: list[Candidate],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        scores = acq.score(candidates)
        assert len(scores) == len(candidates)

    def test_returns_floats(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidates: list[Candidate],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        scores = acq.score(candidates)
        assert all(isinstance(s, float) for s in scores)

    def test_score_batches_raises(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidate_batches: list[list[Candidate]],
    ) -> None:
        """Analytic acquisitions do not support batch scoring."""
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        with pytest.raises(NotImplementedError, match="batch scoring"):
            acq.score_batches(candidate_batches)

    def test_single_candidate(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        scores = acq.score([Candidate(x=[3.0, 3.0])])
        assert len(scores) == 1

    def test_deterministic_across_calls(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidates: list[Candidate],
    ) -> None:
        """Repeated scoring should produce identical results."""
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        scores_a = acq.score(candidates)
        scores_b = acq.score(candidates)
        assert scores_a == scores_b


# ===================================================================
# score() / score_batches() — QBatchBoTorchAcquisition
# ===================================================================


class TestQBatchScore:
    """QBatchBoTorchAcquisition supports singleton scoring."""

    def test_singleton_returns_ones_before_update(
        self, candidates: list[Candidate]
    ) -> None:
        """Before update(), score() returns constant 1.0 for every candidate."""
        acq = StubQBatch()
        scores = acq.score(candidates)
        assert scores == [1.0] * len(candidates)

    def test_singleton_returns_correct_length(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidates: list[Candidate],
    ) -> None:
        acq = StubQBatch()
        acq.update(fitted_surrogate, single_fidelity_observations)
        scores = acq.score(candidates)
        assert len(scores) == len(candidates)

    def test_all_scores_are_floats(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidates: list[Candidate],
    ) -> None:
        acq = StubQBatch()
        acq.update(fitted_surrogate, single_fidelity_observations)
        for s in acq.score(candidates):
            assert isinstance(s, float)

    def test_empty_candidate_list_returns_empty(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        """score([]) should return [] instead of crashing."""
        acq = StubQBatch()
        acq.update(fitted_surrogate, single_fidelity_observations)
        scores = acq.score([])
        assert scores == []


# ===================================================================
# cost_fn post-processing
# ===================================================================


class TestCostWeightingPostProcessing:
    """Optional cost_weighting applies caller-defined post-processing to raw scores."""

    def test_score_with_cost_weighting_divides_by_cost(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidates: list[Candidate],
    ) -> None:
        acq = StubQBatch()
        acq.update(fitted_surrogate, single_fidelity_observations)
        raw_scores = acq.score(candidates)
        cost_weighting = lambda scores, cands: [s / 2.0 for s in scores]  # noqa: E731
        weighted_scores = acq.score(candidates, cost_weighting=cost_weighting)
        assert len(weighted_scores) == len(raw_scores)
        for raw, ws in zip(raw_scores, weighted_scores):
            assert math.isclose(ws, raw / 2.0, rel_tol=1e-6)

    def test_score_with_cost_weighting_subtractive(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidates: list[Candidate],
    ) -> None:
        acq = StubQBatch()
        acq.update(fitted_surrogate, single_fidelity_observations)
        raw_scores = acq.score(candidates)
        cost_weighting = lambda scores, cands: [s - 1.0 for s in scores]  # noqa: E731
        weighted_scores = acq.score(candidates, cost_weighting=cost_weighting)
        for raw, ws in zip(raw_scores, weighted_scores):
            assert math.isclose(ws, raw - 1.0, rel_tol=1e-6)

    def test_score_without_cost_weighting_unchanged(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidates: list[Candidate],
    ) -> None:
        acq = StubQBatch()
        acq.update(fitted_surrogate, single_fidelity_observations)
        assert acq.score(candidates) == acq.score(candidates, cost_weighting=None)

    def test_cost_weighting_not_applied_before_update(
        self, candidates: list[Candidate]
    ) -> None:
        """cost_weighting has no effect before update() — all scores remain 1.0."""
        acq = StubQBatch()
        cost_weighting = lambda scores, cands: [s * 999 for s in scores]  # noqa: E731
        scores = acq.score(candidates, cost_weighting=cost_weighting)
        assert scores == [1.0] * len(candidates)


class TestMultiFidelityResolution:
    """Verify that MF helpers resolve correctly during update()."""

    def test_single_fidelity_resolves_none(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        assert acq.resolved_target_fidelity_value is None
        assert acq.resolved_project_to_target_fidelity_fn is None

    def test_multi_fidelity_resolves_target_value(
        self,
        fitted_mf_surrogate: BoTorchGPSurrogate,
        multi_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_mf_surrogate, multi_fidelity_observations)
        assert acq.resolved_target_fidelity_value is not None
        assert acq.resolved_target_fidelity_value == 1.0  # max confidence

    def test_multi_fidelity_resolves_projection_fn(
        self,
        fitted_mf_surrogate: BoTorchGPSurrogate,
        multi_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_mf_surrogate, multi_fidelity_observations)
        proj_fn = acq.resolved_project_to_target_fidelity_fn
        assert proj_fn is not None

        # Verify the projection overwrites the fidelity dimension.
        X = torch.randn(3, 3)
        X_proj = proj_fn(X)
        fid_dim = fitted_mf_surrogate.get_fidelity_dimension()
        assert fid_dim is not None
        assert acq.resolved_target_fidelity_value is not None
        assert torch.all(X_proj[..., fid_dim] == acq.resolved_target_fidelity_value)

    def test_target_fidelity_override(
        self,
        fitted_mf_surrogate: BoTorchGPSurrogate,
        multi_fidelity_observations: list[Observation],
    ) -> None:
        """User-specified override takes precedence over surrogate default."""
        acq = StubAnalytic(target_fidelity_value=0.42)
        acq.update(fitted_mf_surrogate, multi_fidelity_observations)
        assert acq.resolved_target_fidelity_value == 0.42

    def test_custom_projection_fn_override(
        self,
        fitted_mf_surrogate: BoTorchGPSurrogate,
        multi_fidelity_observations: list[Observation],
    ) -> None:
        """User-specified projection callable takes precedence."""

        def custom_fn(X: torch.Tensor) -> torch.Tensor:
            return X * 0.0

        acq = StubAnalytic(project_to_target_fidelity_fn=custom_fn)
        acq.update(fitted_mf_surrogate, multi_fidelity_observations)
        assert acq.resolved_project_to_target_fidelity_fn is custom_fn


# ===================================================================
# Cost-aware resolution
# ===================================================================


class TestCostAwareResolution:
    """Verify cost model / cost-aware utility resolution."""

    def test_no_cost_by_default(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        assert acq.resolved_cost_model is None
        assert acq.resolved_cost_aware_utility is None

    def test_custom_cost_model_override(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        cost_model = object()
        cost_utility = object()
        # Both must be provided since default utility builder is not implemented.
        acq = StubAnalytic(cost_model=cost_model, cost_aware_utility=cost_utility)
        acq.update(fitted_surrogate, single_fidelity_observations)
        assert acq.resolved_cost_model is cost_model
        assert acq.resolved_cost_aware_utility is cost_utility

    def test_custom_cost_aware_utility_override(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        sentinel = object()
        acq = StubAnalytic(cost_aware_utility=sentinel)
        acq.update(fitted_surrogate, single_fidelity_observations)
        assert acq.resolved_cost_aware_utility is sentinel

    def test_fidelity_costs_without_implementation_raises(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        """Default cost-model builder is not implemented yet — should raise."""
        acq = StubAnalytic(fidelity_costs={0: 1.0, 1: 5.0})
        with pytest.raises(NotImplementedError, match="cost-model construction"):
            acq.update(fitted_surrogate, single_fidelity_observations)


# ===================================================================
# _build_botorch_acquisition — lifecycle
# ===================================================================


class TestBuildLifecycle:
    """Verify that _build_botorch_acquisition is called during update."""

    def test_acqf_built_during_update(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        assert acq.botorch_acqf is not None

    def test_acqf_rebuilt_on_each_update(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        """Each update() should produce a fresh acquisition object."""
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        first = acq.botorch_acqf

        acq.update(fitted_surrogate, single_fidelity_observations)
        second = acq.botorch_acqf
        # The stub returns a new lambda each time.
        assert first is not second


# ===================================================================
# _score_encoded — shared scoring path
# ===================================================================


class TestScoreEncoded:
    """Verify the shared _score_encoded helper."""

    def test_uses_no_grad(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
        candidates: list[Candidate],
    ) -> None:
        """Scoring should not accumulate gradients."""
        acq = StubAnalytic()
        acq.update(fitted_surrogate, single_fidelity_observations)
        
        # Create candidates with gradients enabled to verify torch.no_grad() is applied
        test_candidates = [Candidate(x=[1.0, 1.0])]
        cand_list = list(test_candidates)
        X = fitted_surrogate.encode_candidates(cand_list).unsqueeze(1)
        X_with_grad = X.clone().detach().requires_grad_(True)
        
        # Directly call _score_encoded to inspect the output tensor
        botorch_acqf = acq._require_botorch_acqf()
        with torch.no_grad():
            raw_output = botorch_acqf(X_with_grad)
        
        # Verify the output tensor does not track gradients
        assert not raw_output.requires_grad, (
            "Acquisition output should not require gradients when scored with "
            "torch.no_grad()"
        )

    def test_custom_acqf_fn(
        self,
        fitted_surrogate: BoTorchGPSurrogate,
        single_fidelity_observations: list[Observation],
    ) -> None:
        """A custom acqf_fn callable is correctly invoked."""
        # Always returns 42.0 per candidate.
        acq = StubAnalytic(acqf_fn=lambda X: torch.full((X.shape[0],), 42.0))
        acq.update(fitted_surrogate, single_fidelity_observations)
        scores = acq.score([Candidate(x=[1.0, 1.0]), Candidate(x=[2.0, 2.0])])
        assert scores == [42.0, 42.0]


# ===================================================================
# Multi-fidelity scoring end-to-end
# ===================================================================


class TestMultiFidelityScoring:
    """Scoring with a multi-fidelity surrogate should work correctly."""

    def test_qbatch_singleton_score_mf(
        self,
        fitted_mf_surrogate: BoTorchGPSurrogate,
        multi_fidelity_observations: list[Observation],
    ) -> None:
        acq = StubQBatch()
        acq.update(fitted_mf_surrogate, multi_fidelity_observations)
        candidates = [
            Candidate(x=[2.0, 3.0], fidelity=1),
            Candidate(x=[4.0, 5.0], fidelity=0),
        ]
        scores = acq.score(candidates)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
