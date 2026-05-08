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
from activelearning.sampler.fidelity_policy import (
    FixedFidelityPolicy,
    FixedFidelityPolicyConfig,
    JointSamplingFidelityPolicy,
)
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
def conf_2d_with_bounds():
    """2-D grid config plus [0, 1]^2 output bounds."""
    conf, tmpdir = _make_minimal_gflownet_conf(
        n_train_steps=5, grid_length=5, n_dim=2, cell_min=0.0, cell_max=1.0
    )
    yield conf, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


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

    def test_stores_fixed_output_policy(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetGridSampler(
            n_samples=5,
            conf=conf,
            fidelity_policy=FixedFidelityPolicy(value=2),
        )
        assert sampler.fidelity_policy == FixedFidelityPolicy(value=2)

    def test_stores_output_bounds(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        bounds = [(-1.0, 1.0), (0.0, 2.0)]
        sampler = GFlowNetGridSampler(n_samples=3, conf=conf, output_bounds=bounds)
        assert sampler._out_lb is not None
        assert sampler._out_ub is not None

    def test_no_output_bounds_leaves_rescaling_disabled(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetGridSampler(n_samples=3, conf=conf, output_bounds=None)
        assert sampler._out_lb is None
        assert sampler._out_ub is None

    def test_raises_for_non_grid_env(self, conf_non_grid):
        """GFlowNetGridSampler must reject non-Grid envs at construction time."""
        conf, _ = conf_non_grid
        with pytest.raises(ValueError, match="Grid environment"):
            GFlowNetGridSampler(n_samples=3, conf=conf)

    def test_reward_fidelity_is_rejected_for_joint_sampling(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        with pytest.raises(ValueError, match="reward_fidelity"):
            GFlowNetGridSampler(
                n_samples=3,
                conf=conf,
                fidelity_policy=JointSamplingFidelityPolicy(n_fidelities=2),
                reward_fidelity=1,
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

    def test_sample_applies_fixed_output_policy(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetGridSampler(
            n_samples=4,
            conf=conf,
            fidelity_policy=FixedFidelityPolicy(value=3),
        )
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        assert {candidate.fidelity for candidate in candidates} == {3}

    def test_sample_within_output_bounds(self, conf_2d_with_bounds):
        conf, _ = conf_2d_with_bounds
        bounds = [(-3.0, 7.0), (2.0, 10.0)]
        sampler = GFlowNetGridSampler(n_samples=5, conf=conf, output_bounds=bounds)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        for c in candidates:
            assert bounds[0][0] <= c.x[0] <= bounds[0][1], (
                f"x[0]={c.x[0]} out of {bounds[0]}"
            )
            assert bounds[1][0] <= c.x[1] <= bounds[1][1], (
                f"x[1]={c.x[1]} out of {bounds[1]}"
            )

    def test_sample_within_native_grid_range_when_no_output_bounds(
        self, conf_2d_with_bounds
    ):
        conf, _ = conf_2d_with_bounds
        # cell_min=0, cell_max=1 from fixture
        sampler = GFlowNetGridSampler(n_samples=5, conf=conf, output_bounds=None)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        for c in candidates:
            for v in c.x:
                assert 0.0 <= v <= 1.0, f"coordinate {v} outside [0, 1]"

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
    def _setup(self, conf_2d_with_bounds):
        self.conf, self.tmpdir = conf_2d_with_bounds

    def test_active_learning_loop_runs_at_least_one_round(self):
        """Full AL loop completes at least one round and returns observations."""
        sampler = GFlowNetGridSampler(
            n_samples=20,
            conf=self.conf,
            output_bounds=[(-5.0, 10.0), (0.0, 15.0)],
        )
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
        sampler = GFlowNetGridSampler(
            n_samples=20,
            conf=self.conf,
            output_bounds=[(-5.0, 10.0), (0.0, 15.0)],
        )
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

        sampler = GFlowNetGridSampler(
            n_samples=20,
            conf=self.conf,
            output_bounds=[(-5.0, 10.0), (0.0, 15.0)],
        )
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


class TestGFlowNetGridSamplerConfigRoundTrip:
    """Verify that GFlowNetGridSamplerConfig forwards the shared fidelity policy."""

    def test_config_build_passes_fixed_policy_to_sampler(self, gflownet_conf_2d):
        from activelearning.sampler.config import GFlowNetGridSamplerConfig

        conf_dict, _ = gflownet_conf_2d
        conf_raw = OmegaConf.to_container(conf_dict, resolve=True)
        cfg = GFlowNetGridSamplerConfig(
            n_samples=3,
            fidelity_policy=FixedFidelityPolicyConfig(value=2),
            conf=conf_raw,
            output_bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        sampler = cfg.build()
        assert sampler.fidelity_policy == FixedFidelityPolicy(value=2)
