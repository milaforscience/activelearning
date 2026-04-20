"""Tests for GFlowNetSampler."""

from unittest.mock import Mock

import pytest
import torch

from activelearning.runtime import RuntimeContext
from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler
from activelearning.utils.types import Candidate


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _ConstantAcquisition:
    """Returns a fixed scalar for every candidate (for training stability)."""

    def score(self, candidates):
        return [1.0] * len(candidates)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sampler(gflownet_conf_2d) -> GFlowNetSampler:
    conf, _ = gflownet_conf_2d
    return GFlowNetSampler(n_samples=4, conf=conf)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestGFlowNetSamplerInstantiation:
    def test_instantiation_stores_n_samples(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=7, conf=conf)
        assert sampler.n_samples == 7

    def test_instantiation_stores_n_fidelities(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf, n_fidelities=2)
        assert sampler.n_fidelities == 2

    def test_default_n_fidelities_is_one(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf)
        assert sampler.n_fidelities == 1


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


class TestGFlowNetSamplerHelpers:
    def test_device_str_returns_string(self, sampler):
        sampler.bind_runtime_context(RuntimeContext(device=torch.device("cpu")))
        assert sampler._device_str() == "cpu"

    def test_float_precision_float32(self, sampler):
        sampler.bind_runtime_context(RuntimeContext(dtype=torch.float32))
        assert sampler._float_precision() == 32

    def test_float_precision_float64(self, sampler):
        sampler.bind_runtime_context(RuntimeContext(dtype=torch.float64))
        assert sampler._float_precision() == 64


# ---------------------------------------------------------------------------
# _states_to_candidates: guard clauses
# (conversion logic is tested in test_utils.py via proxy_states_to_candidates)
# ---------------------------------------------------------------------------


class TestStatesToCandidatesGuards:
    def test_empty_list_returns_empty(self, sampler):
        assert sampler._states_to_candidates([], Mock()) == []

    def test_empty_tensor_returns_empty(self, sampler):
        assert sampler._states_to_candidates(torch.empty(0, 2), Mock()) == []

    def test_non_list_non_tensor_returns_empty(self, sampler):
        assert sampler._states_to_candidates("not_a_state", Mock()) == []

    def test_none_returns_empty(self, sampler):
        assert sampler._states_to_candidates(None, Mock()) == []


# ---------------------------------------------------------------------------
# sample(): errors
# ---------------------------------------------------------------------------


class TestGFlowNetSamplerSampleErrors:
    def test_sample_raises_without_acquisition(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf)
        with pytest.raises(ValueError, match="requires an acquisition"):
            sampler.sample(acquisition=None)


# ---------------------------------------------------------------------------
# sample(): smoke tests (require actual GFlowNet training)
# ---------------------------------------------------------------------------


class TestGFlowNetSamplerSmokeTest:
    def test_sample_returns_candidates(self, gflownet_conf_2d):
        """sample() returns the requested number of Candidate objects."""
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=4, conf=conf)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        assert len(candidates) == 4
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_sample_returns_2d_coordinates(self, gflownet_conf_2d):
        """Each candidate has 2 coordinates (matching the 2-D grid env)."""
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        assert all(len(c.x) == 2 for c in candidates)


# ---------------------------------------------------------------------------
# Runtime logger integration
# ---------------------------------------------------------------------------


class TestGFlowNetSamplerRuntimeLogger:
    def test_runtime_logger_receives_metrics(self, gflownet_conf_2d):
        """Binding a runtime logger causes metrics to be forwarded during training."""
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf)
        runtime_logger = Mock()
        sampler.bind_runtime_context(RuntimeContext(logger=runtime_logger))

        sampler.sample(acquisition=_ConstantAcquisition())

        runtime_logger.log_metric.assert_called()
        # log_step must NOT be called by the wrapper — it belongs to the AL loop.
        runtime_logger.log_step.assert_not_called()

    def test_runtime_logger_end_not_called_by_sampler(self, gflownet_conf_2d):
        """The sampler must not call logger.end(); that belongs to the AL loop."""
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf)
        runtime_logger = Mock()
        sampler.bind_runtime_context(RuntimeContext(logger=runtime_logger))

        sampler.sample(acquisition=_ConstantAcquisition())

        runtime_logger.end.assert_not_called()
