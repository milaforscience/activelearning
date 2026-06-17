"""Tests for GFlowNetGridSampler."""

import shutil
import torch
import pytest
from omegaconf import OmegaConf
from typing import Sequence

from activelearning.active_learning import active_learning
from activelearning.budget.budget import Budget
from activelearning.dataset.list_dataset import ListDataset
from activelearning.oracle.augmented_function_oracle import BraninOracle
from activelearning.runtime import RuntimeContext
from activelearning.sampler.gflownet.grid_sampler import GFlowNetGridSampler
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.utils.types import Candidate
from tests.sampler.gflownet.conftest import _make_minimal_gflownet_conf


# ---------------------------------------------------------------------------
# Acquisition stubs
# ---------------------------------------------------------------------------


class _ConstantAcquisition:
    """Returns a fixed scalar for every candidate."""

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        return [1.0] * len(candidates)


class _CountingAcquisition:
    """Returns the sum of absolute coordinates as a proxy score."""

    def score(self, candidates: Sequence[Candidate]) -> list[float]:
        return [sum(abs(v) for v in c.x) for c in candidates]

    def update(self, surrogate, observations) -> None:  # noqa: ARG002
        pass


# ---------------------------------------------------------------------------
# Selector wrapper: assigns a fidelity to each selected candidate
# ---------------------------------------------------------------------------


class _FidelityAssigningSelector:
    """Delegates to an inner selector, then stamps a fixed fidelity on results."""

    def __init__(self, inner, fidelity: int = 0) -> None:
        self.inner = inner
        self.fidelity = fidelity

    def __call__(self, samples, **kwargs):
        selected = self.inner(samples, **kwargs)
        return [Candidate(x=c.x, fidelity=self.fidelity) for c in selected]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conf_non_grid():
    """Config with a non-Grid env (ContinuousCube) — used to test validation."""
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="gfn_test_")
    conf, _ = _make_minimal_gflownet_conf(n_train_steps=5, grid_length=5, n_dim=2)
    env_conf = OmegaConf.to_container(conf.env, resolve=True)
    env_conf["_target_"] = "gflownet.envs.cube.ContinuousCube"
    conf = OmegaConf.merge(conf, {"env": env_conf})
    yield conf, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestGFlowNetGridSamplerInstantiation:
    def test_stores_n_samples(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetGridSampler(n_samples=5, conf=conf)
        assert sampler.n_samples == 5

    def test_raises_for_non_grid_env(self, conf_non_grid):
        """GFlowNetGridSampler must reject non-Grid envs at construction time."""
        conf, _ = conf_non_grid
        with pytest.raises(ValueError, match="Grid environment"):
            GFlowNetGridSampler(n_samples=3, conf=conf)

    def test_domain_bounds_stored(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        bounds = [[-5.0, 10.0], [0.0, 15.0]]
        sampler = GFlowNetGridSampler(n_samples=3, conf=conf, domain_bounds=bounds)
        assert sampler.domain_bounds == bounds

    def test_domain_bounds_none_by_default(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetGridSampler(n_samples=3, conf=conf)
        assert sampler.domain_bounds is None

    def test_domain_bounds_wrong_ndim_raises(self, gflownet_conf_2d):
        """domain_bounds length must match conf.env.n_dim."""
        conf, _ = gflownet_conf_2d  # n_dim=2
        with pytest.raises(ValueError, match="n_dim"):
            GFlowNetGridSampler(n_samples=3, conf=conf, domain_bounds=[[0.0, 1.0]])

    def test_domain_bounds_lo_ge_hi_raises(self, gflownet_conf_2d):
        """Each domain_bounds pair must have lo < hi."""
        conf, _ = gflownet_conf_2d
        with pytest.raises(ValueError, match="lo.*>=.*hi|hi.*lo"):
            GFlowNetGridSampler(
                n_samples=3,
                conf=conf,
                domain_bounds=[[5.0, 0.0], [0.0, 1.0]],
            )


# ---------------------------------------------------------------------------
# domain_bounds coordinate mapping
# ---------------------------------------------------------------------------


class TestGFlowNetGridSamplerOutputBounds:
    def test_candidates_within_domain_bounds(self, gflownet_conf_2d):
        """With domain_bounds, candidates must lie within the specified per-dim ranges."""
        conf, _ = gflownet_conf_2d
        bounds = [[-5.0, 10.0], [0.0, 15.0]]
        sampler = GFlowNetGridSampler(n_samples=8, conf=conf, domain_bounds=bounds)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        for c in candidates:
            x1, x2 = c.x
            assert -5.0 <= x1 <= 10.0, f"x1={x1} outside [-5, 10]"
            assert 0.0 <= x2 <= 15.0, f"x2={x2} outside [0, 15]"

    def test_domain_bounds_different_per_dim(self, gflownet_conf_2d):
        """Dimensions must map to different ranges when bounds differ."""
        conf, _ = gflownet_conf_2d
        # Use clearly non-overlapping ranges so we can tell them apart.
        bounds = [[100.0, 200.0], [0.0, 1.0]]
        sampler = GFlowNetGridSampler(n_samples=8, conf=conf, domain_bounds=bounds)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        for c in candidates:
            x1, x2 = c.x
            assert 100.0 <= x1 <= 200.0, f"x1={x1} outside [100, 200]"
            assert 0.0 <= x2 <= 1.0, f"x2={x2} outside [0, 1]"

    def test_no_domain_bounds_uses_cell_min_max(self, gflownet_conf_2d):
        """Without domain_bounds, candidates stay within cell_min/cell_max (0.0–1.0)."""
        conf, _ = gflownet_conf_2d  # cell_min=0.0, cell_max=1.0
        sampler = GFlowNetGridSampler(n_samples=5, conf=conf)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        for c in candidates:
            for v in c.x:
                assert 0.0 <= v <= 1.0, f"coordinate {v} outside [0, 1]"

    def test_proxy_receives_domain_bounds_coordinates_during_training(
        self, gflownet_conf_2d
    ):
        """Acquisition must receive domain_bounds coordinates during training.

        Regression test for the original domain_bounds bug where the GFlowNet
        trained on the default cell_min/cell_max coordinate range, but candidates
        were post-hoc rescaled to domain_bounds — meaning the training reward was
        evaluated at the wrong domain, making the learned policy incompatible with
        the acquisition function.

        If the fix is correct, every call to acquisition.score() during training
        must receive coordinates that lie within domain_bounds, not within the
        default cell range.
        """
        conf, _ = gflownet_conf_2d  # default cell_min=0.0, cell_max=1.0
        # Use bounds that don't overlap with the default [0, 1] range so we can
        # unambiguously detect if training uses the wrong cell coordinates.
        bounds = [[100.0, 200.0], [500.0, 600.0]]
        received_coords: list[tuple] = []

        class _RecordingAcquisition:
            def score(self, candidates):
                received_coords.extend(c.x for c in candidates)
                return [1.0] * len(candidates)

        sampler = GFlowNetGridSampler(n_samples=4, conf=conf, domain_bounds=bounds)
        sampler.sample(acquisition=_RecordingAcquisition())

        assert len(received_coords) > 0, "Acquisition was never called during training"
        for x1, x2 in received_coords:
            assert 100.0 <= x1 <= 200.0, (
                f"x1={x1} is outside domain_bounds [100, 200]: "
                "proxy is being called with wrong-domain (cell_min/cell_max) coordinates"
            )
            assert 500.0 <= x2 <= 600.0, (
                f"x2={x2} is outside domain_bounds [500, 600]: "
                "proxy is being called with wrong-domain (cell_min/cell_max) coordinates"
            )

    def test_6d_hartmann_domain_candidates_in_bounds(self, gflownet_conf_6d):
        """6-D Hartmann domain: all candidate coordinates must lie within [0, 1]."""
        conf, _ = gflownet_conf_6d
        bounds = [[0.0, 1.0]] * 6
        sampler = GFlowNetGridSampler(n_samples=8, conf=conf, domain_bounds=bounds)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        assert all(len(c.x) == 6 for c in candidates), (
            "Candidates must be 6-dimensional"
        )
        for c in candidates:
            for dim, v in enumerate(c.x):
                assert 0.0 <= v <= 1.0, f"dim {dim}: {v} outside [0, 1]"

    def test_6d_asymmetric_domain_bounds(self, gflownet_conf_6d):
        """6-D grid with distinct per-dimension bounds maps each dimension independently."""
        conf, _ = gflownet_conf_6d
        # Use clearly non-overlapping ranges for each dim so mapping errors are obvious.
        bounds = [
            [0.0, 1.0],  # dim 0
            [10.0, 20.0],  # dim 1
            [100.0, 200.0],  # dim 2
            [-5.0, -1.0],  # dim 3
            [0.5, 1.5],  # dim 4
            [50.0, 60.0],  # dim 5
        ]
        sampler = GFlowNetGridSampler(n_samples=8, conf=conf, domain_bounds=bounds)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        for c in candidates:
            for dim, (v, (lo, hi)) in enumerate(zip(c.x, bounds)):
                assert lo <= v <= hi, f"dim {dim}: {v} outside [{lo}, {hi}]"

    def test_6d_proxy_receives_domain_bounds_coordinates(self, gflownet_conf_6d):
        """In 6-D, training rewards must be evaluated within domain_bounds coordinates.

        Regression test: verifies that the per-dimension mapping applies correctly
        for higher-dimensional problems, not just the 2-D case.
        """
        conf, _ = gflownet_conf_6d  # default cell_min=0.0, cell_max=1.0
        # Each dim in a non-overlapping range that doesn't intersect [0, 1].
        bounds = [
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0],
            [40.0, 50.0],
            [50.0, 60.0],
            [60.0, 70.0],
        ]
        received_coords: list[tuple] = []

        class _RecordingAcquisition:
            def score(self, candidates):
                received_coords.extend(c.x for c in candidates)
                return [1.0] * len(candidates)

        sampler = GFlowNetGridSampler(n_samples=4, conf=conf, domain_bounds=bounds)
        sampler.sample(acquisition=_RecordingAcquisition())

        assert len(received_coords) > 0, "Acquisition was never called during training"
        for coords in received_coords:
            for dim, (v, (lo, hi)) in enumerate(zip(coords, bounds)):
                assert lo <= v <= hi, (
                    f"dim {dim}: v={v} is outside domain_bounds [{lo}, {hi}]: "
                    "proxy received wrong-domain coordinates during training"
                )


# ---------------------------------------------------------------------------
# Sample output validation
# ---------------------------------------------------------------------------


class TestGFlowNetGridSamplerSample:
    def test_sample_returns_correct_count(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetGridSampler(n_samples=6, conf=conf)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        assert len(candidates) == 6

    def test_sample_returns_candidate_objects(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetGridSampler(n_samples=4, conf=conf)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_sample_within_cell_range(self, gflownet_conf_2d):
        """Candidates should lie within [cell_min, cell_max] (the native grid range)."""
        conf, _ = gflownet_conf_2d
        # gflownet_conf_2d uses cell_min=0.0, cell_max=1.0
        sampler = GFlowNetGridSampler(n_samples=5, conf=conf)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        for c in candidates:
            for v in c.x:
                assert 0.0 <= v <= 1.0, (
                    f"coordinate {v} outside [cell_min, cell_max]=[0, 1]"
                )

    def test_sample_raises_without_acquisition(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetGridSampler(n_samples=3, conf=conf)
        with pytest.raises(ValueError, match="requires an acquisition"):
            sampler.sample(acquisition=None)


# ---------------------------------------------------------------------------
# End-to-end active-learning loop integration
# ---------------------------------------------------------------------------


class TestGFlowNetGridSamplerActivelearningLoop:
    @pytest.fixture(autouse=True)
    def _setup(self, gflownet_conf_2d):
        self.conf, self.tmpdir = gflownet_conf_2d

    def test_active_learning_loop_runs_at_least_one_round(self):
        """Full AL loop completes at least one round and returns observations."""
        sampler = GFlowNetGridSampler(n_samples=20, conf=self.conf)
        oracle = BraninOracle(
            fidelity_costs={0: 1.0},
            fidelity_confidences={0: 1.0},
        )
        _, cost, num_rounds = active_learning(
            dataset=ListDataset(),
            surrogate=DummyMeanSurrogate(),
            acquisition=_CountingAcquisition(),
            sampler=sampler,
            selector=_FidelityAssigningSelector(
                inner=TopKAcquisitionSelector(num_samples=3), fidelity=0
            ),
            oracle=oracle,
            budget=Budget(available_budget=5.0, schedule=lambda r: 3.0),
        )
        assert num_rounds >= 1
        assert cost > 0.0

    def test_active_learning_loop_observations_have_correct_dimensionality(self):
        """All oracle observations carry 2-D coordinates."""
        sampler = GFlowNetGridSampler(n_samples=20, conf=self.conf)
        oracle = BraninOracle(
            fidelity_costs={0: 1.0},
            fidelity_confidences={0: 1.0},
        )
        dataset, _, _ = active_learning(
            dataset=ListDataset(),
            surrogate=DummyMeanSurrogate(),
            acquisition=_CountingAcquisition(),
            sampler=sampler,
            selector=_FidelityAssigningSelector(
                inner=TopKAcquisitionSelector(num_samples=3), fidelity=0
            ),
            oracle=oracle,
            budget=Budget(available_budget=5.0, schedule=lambda r: 3.0),
        )
        for obs in dataset.get_observations_iterable():
            assert len(obs.x) == 2
            assert torch.isfinite(torch.tensor(obs.y))

    def test_runtime_logger_end_called_once_by_al_loop(self):
        """logger.end() must be called exactly once — by the AL loop, not the sampler."""
        from unittest.mock import Mock

        sampler = GFlowNetGridSampler(n_samples=20, conf=self.conf)
        runtime_logger = Mock()
        runtime_context = RuntimeContext(logger=runtime_logger)
        oracle = BraninOracle(
            fidelity_costs={0: 1.0},
            fidelity_confidences={0: 1.0},
        )
        active_learning(
            dataset=ListDataset(),
            surrogate=DummyMeanSurrogate(),
            acquisition=_CountingAcquisition(),
            sampler=sampler,
            selector=_FidelityAssigningSelector(
                inner=TopKAcquisitionSelector(num_samples=3), fidelity=0
            ),
            oracle=oracle,
            budget=Budget(available_budget=5.0, schedule=lambda r: 3.0),
            runtime_context=runtime_context,
        )
        runtime_logger.end.assert_called_once()
