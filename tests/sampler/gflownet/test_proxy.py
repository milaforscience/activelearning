"""Unit tests for AcquisitionProxy."""

import pytest
import torch

from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapperBase,
)
from activelearning.sampler.gflownet.proxy import AcquisitionProxy
from activelearning.utils.types import Candidate


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _make_proxy(**kwargs) -> AcquisitionProxy:
    """Build an AcquisitionProxy with sensible test defaults."""
    defaults = {"device": "cpu", "float_precision": 32, "reward_function": "identity"}
    return AcquisitionProxy(**{**defaults, **kwargs})


class _ConstantAcquisition:
    """Returns a fixed score for every candidate."""

    def __init__(self, value: float = 2.0) -> None:
        self.value = value

    def score(self, candidates: list[Candidate]) -> list[float]:
        return [self.value] * len(candidates)


class _CoordSumAcquisition:
    """Returns the sum of each candidate's coordinates."""

    def score(self, candidates: list[Candidate]) -> list[float]:
        return [sum(c.x) for c in candidates]


class _FidelityCapturingAcquisition:
    """Records all candidate fidelities it receives."""

    def __init__(self) -> None:
        self.seen: list[int | None] = []

    def score(self, candidates: list[Candidate]) -> list[float]:
        self.seen.extend(c.fidelity for c in candidates)
        return [1.0] * len(candidates)


class _StringCapturingAcquisition:
    """Records candidate proxy values to verify string inputs are preserved."""

    def __init__(self) -> None:
        self.seen: list[object] = []

    def score(self, candidates: list[Candidate]) -> list[float]:
        self.seen.extend(c.x for c in candidates)
        return [1.0] * len(candidates)


class _StubMFEnv(MultiFidelityGFlowNetEnvWrapperBase):
    """Minimal MF env stub; bypasses parent __init__."""

    def __init__(self, idx_base_env: int = 0, idx_fidelity: int = 1) -> None:
        self.idx_base_env = idx_base_env
        self.idx_fidelity = idx_fidelity


def _make_mf_states(
    coords: list[list[float]],
    fidelities: list[int],
    idx_base: int = 0,
    idx_fid: int = 1,
) -> list[dict]:
    return [
        {
            idx_base: torch.tensor(c, dtype=torch.float32),
            idx_fid: torch.tensor([f], dtype=torch.float32),
        }
        for c, f in zip(coords, fidelities)
    ]


# ---------------------------------------------------------------------------
# Instantiation & setup
# ---------------------------------------------------------------------------


class TestAcquisitionProxyInit:
    def test_acquisition_defaults_to_none(self):
        proxy = _make_proxy()
        assert proxy.acquisition is None

    def test_env_defaults_to_none(self):
        proxy = _make_proxy()
        assert proxy._env is None

    def test_reward_scaling_defaults_to_identity(self):
        proxy = _make_proxy()
        assert proxy._round_reward_scale() == pytest.approx(1.0)

    def test_bad_reward_beta_raises(self):
        with pytest.raises(ValueError, match="reward_beta"):
            _make_proxy(reward_beta=0.0)

    def test_bad_reward_rho_raises(self):
        with pytest.raises(ValueError, match="reward_rho"):
            _make_proxy(reward_rho=0.0)

    def test_set_acquisition_replaces(self):
        proxy = _make_proxy()
        acq = _ConstantAcquisition()
        proxy.set_acquisition(acq)
        assert proxy.acquisition is acq

    def test_set_round_index_updates_reward_scale(self):
        proxy = _make_proxy(reward_beta=0.5, reward_rho=2.0)
        proxy.set_round_index(3)
        assert proxy._round_reward_scale() == pytest.approx(16.0)

    def test_set_round_index_rejects_negative(self):
        proxy = _make_proxy()
        with pytest.raises(ValueError, match="round_index"):
            proxy.set_round_index(-1)

    def test_setup_stores_env(self):
        proxy = _make_proxy()
        env = object()
        proxy.setup(env)
        assert proxy._env is env

    def test_setup_none_clears_env(self):
        proxy = _make_proxy()
        proxy.setup(object())
        proxy.setup(None)
        assert proxy._env is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestAcquisitionProxyErrors:
    def test_raises_when_no_acquisition(self):
        proxy = _make_proxy()
        with pytest.raises(RuntimeError, match="no acquisition function"):
            proxy(torch.tensor([[0.1, 0.2]]))

    def test_raises_after_set_to_none(self):
        proxy = _make_proxy()
        proxy.set_acquisition(_ConstantAcquisition())
        proxy.set_acquisition(None)
        with pytest.raises(RuntimeError):
            proxy(torch.tensor([[0.1, 0.2]]))


# ---------------------------------------------------------------------------
# Single-fidelity __call__
# ---------------------------------------------------------------------------


class TestAcquisitionProxyCallSingleFidelity:
    @pytest.fixture(autouse=True)
    def _proxy(self):
        self.proxy = _make_proxy(float_precision=32)
        self.proxy.set_acquisition(_ConstantAcquisition(value=3.0))

    def test_returns_tensor(self):
        result = self.proxy(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        assert torch.is_tensor(result)

    def test_output_length_matches_input(self):
        result = self.proxy(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        assert result.shape == (3,)

    def test_constant_acquisition_values(self):
        result = self.proxy(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        assert torch.allclose(result, torch.tensor([3.0, 3.0]))

    def test_applies_paper_reward_scaling(self):
        proxy = _make_proxy(float_precision=32, reward_beta=0.25, reward_rho=2.0)
        proxy.set_acquisition(_ConstantAcquisition(value=3.0))
        proxy.set_round_index(2)
        result = proxy(torch.tensor([[0.1, 0.2]]))
        assert result.item() == pytest.approx(48.0)

    def test_acquisition_receives_correct_coords(self):
        acq = _CoordSumAcquisition()
        self.proxy.set_acquisition(acq)
        states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = self.proxy(states)
        assert result[0].item() == pytest.approx(3.0, abs=1e-5)
        assert result[1].item() == pytest.approx(7.0, abs=1e-5)

    def test_output_dtype_matches_float_precision_32(self):
        result = self.proxy(torch.tensor([[0.1, 0.2]]))
        assert result.dtype == torch.float32

    def test_output_dtype_matches_float_precision_64(self):
        proxy = _make_proxy(float_precision=64)
        proxy.set_acquisition(_ConstantAcquisition())
        result = proxy(torch.tensor([[0.1, 0.2]]))
        assert result.dtype == torch.float64

    def test_list_of_sequences_input(self):
        result = self.proxy([[0.1, 0.2], [0.3, 0.4]])
        assert result.shape == (2,)

    def test_list_of_strings_preserves_candidate_proxy_value(self):
        acq = _StringCapturingAcquisition()
        self.proxy.set_acquisition(acq)
        self.proxy(["[C][=O][N]"])
        assert acq.seen == ["[C][=O][N]"]


# ---------------------------------------------------------------------------
# Multi-fidelity __call__
# ---------------------------------------------------------------------------


class TestAcquisitionProxyCallMultiFidelity:
    @pytest.fixture(autouse=True)
    def _proxy(self):
        self.env = _StubMFEnv(idx_base_env=0, idx_fidelity=1)
        self.proxy = _make_proxy()
        self.proxy.setup(self.env)
        self.proxy.set_acquisition(_ConstantAcquisition(value=1.0))

    def test_returns_correct_number_of_values(self):
        states = _make_mf_states([[0.1, 0.2], [0.3, 0.4]], fidelities=[0, 1])
        result = self.proxy(states)
        assert result.shape == (2,)

    def test_acquisition_receives_fidelity(self):
        acq = _FidelityCapturingAcquisition()
        self.proxy.set_acquisition(acq)
        states = _make_mf_states([[0.1, 0.2], [0.3, 0.4]], fidelities=[0, 2])
        self.proxy(states)
        assert acq.seen == [0, 2]

    def test_acquisition_receives_base_coords(self):
        acq = _CoordSumAcquisition()
        self.proxy.set_acquisition(acq)
        states = _make_mf_states([[1.0, 2.0]], fidelities=[0])
        result = self.proxy(states)
        assert result[0].item() == pytest.approx(3.0, abs=1e-5)

    def test_fidelity_first_layout(self):
        """Works with idx_fidelity=0, idx_base_env=1 (FidFirst wrapper)."""
        env = _StubMFEnv(idx_base_env=1, idx_fidelity=0)
        self.proxy.setup(env)
        acq = _FidelityCapturingAcquisition()
        self.proxy.set_acquisition(acq)
        states = _make_mf_states([[0.5, 0.6]], fidelities=[2], idx_base=1, idx_fid=0)
        self.proxy(states)
        assert acq.seen == [2]
