"""Tests for GFlowNetSampler."""

from unittest.mock import Mock, patch

import pytest
import torch

from activelearning.runtime import RuntimeContext
from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler
from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapper,
    MultiFidelityGFlowNetEnvWrapperFidFirst,
    MultiFidelityGFlowNetEnvWrapperFidLast,
)
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

    def test_default_fidelity_action_is_any(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf)
        assert sampler.fidelity_action == "any"

    def test_instantiation_stores_fixed_fidelity(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf, fixed_fidelity=1)
        assert sampler.fixed_fidelity == 1

    @pytest.mark.parametrize("action", ["any", "first", "last"])
    def test_fidelity_action_is_stored(self, gflownet_conf_2d, action):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf, fidelity_action=action)
        assert sampler.fidelity_action == action

    def test_fixed_fidelity_requires_single_fidelity(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        with pytest.raises(ValueError, match="fixed_fidelity"):
            GFlowNetSampler(n_samples=3, conf=conf, n_fidelities=2, fixed_fidelity=1)

    @pytest.mark.parametrize("action", ["first", "last"])
    def test_warns_when_fidelity_action_set_with_single_fidelity(
        self, gflownet_conf_2d, action, caplog
    ):
        import logging

        conf, _ = gflownet_conf_2d
        with caplog.at_level(
            logging.WARNING,
            logger="activelearning.sampler.gflownet.gflownet_sampler",
        ):
            GFlowNetSampler(
                n_samples=3, conf=conf, n_fidelities=1, fidelity_action=action
            )
        assert any("fidelity_action" in r.message for r in caplog.records)

    def test_no_warning_when_fidelity_action_any_with_single_fidelity(
        self, gflownet_conf_2d, caplog
    ):
        import logging

        conf, _ = gflownet_conf_2d
        with caplog.at_level(
            logging.WARNING,
            logger="activelearning.sampler.gflownet.gflownet_sampler",
        ):
            GFlowNetSampler(
                n_samples=3, conf=conf, n_fidelities=1, fidelity_action="any"
            )
        assert not caplog.records

    def test_no_warning_when_fidelity_action_set_with_multi_fidelity(
        self, gflownet_conf_2d, caplog
    ):
        import logging

        conf, _ = gflownet_conf_2d
        with caplog.at_level(
            logging.WARNING,
            logger="activelearning.sampler.gflownet.gflownet_sampler",
        ):
            GFlowNetSampler(
                n_samples=3, conf=conf, n_fidelities=2, fidelity_action="first"
            )
        assert not caplog.records

    def test_active_learning_round_reads_from_runtime_context(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf)
        sampler.bind_runtime_context(RuntimeContext(active_learning_round=4))
        assert sampler.active_learning_round == 4


# ---------------------------------------------------------------------------
# fidelity_action → wrapper class selection
# ---------------------------------------------------------------------------


class TestGFlowNetSamplerFidelityActionWrapperSelection:
    """Verify that _build_agent selects the right wrapper class for each action.

    These tests patch ``build_multi_fidelity_env_wrapper`` and
    ``gflownet_from_config`` so no actual GFlowNet training occurs.
    """

    _EXPECTED_WRAPPER = {
        "any": MultiFidelityGFlowNetEnvWrapper,
        "first": MultiFidelityGFlowNetEnvWrapperFidFirst,
        "last": MultiFidelityGFlowNetEnvWrapperFidLast,
    }

    @pytest.mark.parametrize("action", ["any", "first", "last"])
    def test_correct_wrapper_class_is_instantiated(self, gflownet_conf_2d, action):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(
            n_samples=2, conf=conf, n_fidelities=2, fidelity_action=action
        )
        sampler.bind_runtime_context(RuntimeContext())

        wrapper_target = self._EXPECTED_WRAPPER[action]
        captured: list = []

        def _fake_build(fidelity_action, env_base_maker, n_fidelities, **kwargs):
            instance = wrapper_target(
                env_base_maker=env_base_maker, n_fidelities=n_fidelities
            )
            captured.append(instance)
            return instance

        mock_agent = Mock()
        mock_agent.proxy = Mock()
        mock_agent.env = Mock()
        mock_agent.logger = Mock()

        with (
            patch(
                "activelearning.sampler.gflownet.gflownet_sampler.build_multi_fidelity_env_wrapper",
                side_effect=_fake_build,
            ),
            patch(
                "activelearning.sampler.gflownet.gflownet_sampler.gflownet_from_config",
                return_value=mock_agent,
            ),
        ):
            sampler._build_agent(acquisition=Mock())

        assert len(captured) == 1
        assert isinstance(captured[0], wrapper_target)

    def test_invalid_fidelity_action_raises(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(
            n_samples=2,
            conf=conf,
            n_fidelities=2,
            fidelity_action="bad",  # type: ignore[arg-type]
        )
        sampler.bind_runtime_context(RuntimeContext())
        with pytest.raises(ValueError, match="fidelity_action"):
            sampler._build_agent(acquisition=Mock())

    def test_round_index_is_passed_to_proxy(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=2, conf=conf)
        sampler.bind_runtime_context(RuntimeContext(active_learning_round=3))

        mock_agent = Mock()
        mock_agent.proxy = Mock()
        mock_agent.env = Mock()
        mock_agent.logger = Mock()

        with patch(
            "activelearning.sampler.gflownet.gflownet_sampler.gflownet_from_config",
            return_value=mock_agent,
        ):
            sampler._build_agent(acquisition=Mock())

        mock_agent.proxy.set_round_index.assert_called_once_with(3)


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

    def test_sample_applies_fixed_fidelity(self, gflownet_conf_2d):
        conf, _ = gflownet_conf_2d
        sampler = GFlowNetSampler(n_samples=3, conf=conf, fixed_fidelity=1)
        candidates = sampler.sample(acquisition=_ConstantAcquisition())
        assert all(candidate.fidelity == 1 for candidate in candidates)


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


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestGFlowNetSamplerConfigRoundTrip:
    """Verify that GFlowNetSamplerConfig correctly passes fidelity_action to the sampler."""

    def test_default_fidelity_action_in_config_is_any(self):
        from activelearning.sampler.config import GFlowNetSamplerConfig

        cfg = GFlowNetSamplerConfig(n_samples=3)
        assert cfg.fidelity_action == "any"

    @pytest.mark.parametrize("action", ["any", "first", "last"])
    def test_config_stores_fidelity_action(self, action):
        from activelearning.sampler.config import GFlowNetSamplerConfig

        cfg = GFlowNetSamplerConfig(n_samples=3, fidelity_action=action)
        assert cfg.fidelity_action == action

    @pytest.mark.parametrize("action", ["any", "first", "last"])
    def test_config_build_passes_fidelity_action_to_sampler(
        self, gflownet_conf_2d, action
    ):
        from activelearning.sampler.config import GFlowNetSamplerConfig

        conf_dict, _ = gflownet_conf_2d
        # Convert the OmegaConf DictConfig to a plain dict for the `conf` field.
        from omegaconf import OmegaConf

        conf_raw = OmegaConf.to_container(conf_dict, resolve=True)
        cfg = GFlowNetSamplerConfig(n_samples=3, fidelity_action=action, conf=conf_raw)
        sampler = cfg.build()
        assert sampler.fidelity_action == action

    def test_config_build_passes_fixed_fidelity_to_sampler(self, gflownet_conf_2d):
        from activelearning.sampler.config import GFlowNetSamplerConfig

        conf_dict, _ = gflownet_conf_2d
        from omegaconf import OmegaConf

        conf_raw = OmegaConf.to_container(conf_dict, resolve=True)
        cfg = GFlowNetSamplerConfig(n_samples=3, fixed_fidelity=1, conf=conf_raw)
        sampler = cfg.build()
        assert sampler.fixed_fidelity == 1
