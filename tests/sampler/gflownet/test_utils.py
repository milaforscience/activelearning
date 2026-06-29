"""Unit tests for activelearning.sampler.gflownet.utils."""

import pytest
import torch

from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapperBase,
)
from activelearning.sampler.gflownet.utils import proxy_states_to_candidates
from activelearning.utils.types import Candidate


# ---------------------------------------------------------------------------
# Stub: minimal MultiFidelityGFlowNetEnvWrapperBase without real __init__
# ---------------------------------------------------------------------------


class _StubMFEnv(MultiFidelityGFlowNetEnvWrapperBase):
    """Bypasses the parent __init__; exposes only what proxy_states_to_candidates needs."""

    idx_base_env = 0
    idx_fidelity = 1

    def __init__(
        self,
        proxy_return: list,
        idx_base_env: int = 0,
        idx_fidelity: int = 1,
    ) -> None:
        self.idx_base_env = idx_base_env
        self.idx_fidelity = idx_fidelity
        self._proxy_return = proxy_return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mf_proxy(
    coords: list[list[float]],
    fidelity_indices: list[int],
    idx_base: int = 0,
    idx_fid: int = 1,
) -> list[dict]:
    """Build the dict-list structure that a multi-fidelity env.states2proxy returns."""
    return [
        {
            idx_base: torch.tensor(c, dtype=torch.float32),
            idx_fid: torch.tensor([f], dtype=torch.float32),
        }
        for c, f in zip(coords, fidelity_indices)
    ]


def _plain_env():
    """A plain mock env that is NOT a MultiFidelityGFlowNetEnvWrapperBase."""
    from unittest.mock import Mock

    return Mock(spec=[])  # no attributes → isinstance check fails cleanly


# ---------------------------------------------------------------------------
# Guard clause
# ---------------------------------------------------------------------------


class TestProxyStatesToCandidatesGuards:
    def test_empty_list(self):
        assert proxy_states_to_candidates([], _plain_env()) == []

    def test_empty_tensor(self):
        assert proxy_states_to_candidates(torch.empty(0, 2), _plain_env()) == []

    def test_none(self):
        with pytest.raises(TypeError):
            proxy_states_to_candidates(None, _plain_env())

    def test_non_list_non_tensor(self):
        with pytest.raises(TypeError):
            proxy_states_to_candidates("invalid", _plain_env())


# ---------------------------------------------------------------------------
# Single-fidelity paths
# ---------------------------------------------------------------------------


class TestProxyStatesToCandidatesSingleFidelity:
    def test_2d_tensor(self):
        coords = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        result = proxy_states_to_candidates(coords, _plain_env())
        assert len(result) == 2
        assert result[0].x == pytest.approx((0.1, 0.2), abs=1e-6)
        assert result[1].x == pytest.approx((0.3, 0.4), abs=1e-6)

    def test_list_of_1d_tensors(self):
        coords = [torch.tensor([0.5, 0.6]), torch.tensor([0.7, 0.8])]
        result = proxy_states_to_candidates(coords, _plain_env())
        assert len(result) == 2
        assert result[0].x == pytest.approx((0.5, 0.6), abs=1e-6)
        assert result[1].x == pytest.approx((0.7, 0.8), abs=1e-6)

    def test_list_of_plain_sequences(self):
        result = proxy_states_to_candidates([[0.1, 0.9], [0.2, 0.8]], _plain_env())
        assert len(result) == 2
        assert result[0].x == pytest.approx((0.1, 0.9), abs=1e-6)

    def test_no_fidelity_set(self):
        result = proxy_states_to_candidates(torch.tensor([[1.0, 2.0]]), _plain_env())
        assert result[0].fidelity is None

    def test_returns_candidate_objects(self):
        result = proxy_states_to_candidates(torch.tensor([[0.0, 1.0]]), _plain_env())
        assert all(isinstance(c, Candidate) for c in result)

    def test_coordinates_are_python_floats(self):
        """Coordinates must be Python floats (float64) regardless of input dtype."""
        result = proxy_states_to_candidates(
            torch.tensor([[1.0, 2.0]], dtype=torch.float32), _plain_env()
        )
        assert all(isinstance(v, float) for v in result[0].x)


# ---------------------------------------------------------------------------
# Multi-fidelity path
# ---------------------------------------------------------------------------


class TestProxyStatesToCandidatesMultiFidelity:
    def test_extracts_coords_and_fidelity(self):
        # Raw fidelities from real Choice env are 1-based (1..N).
        proxy = _make_mf_proxy([[0.1, 0.2], [0.3, 0.4]], fidelity_indices=[1, 2])
        env = _StubMFEnv(proxy_return=proxy, idx_base_env=0, idx_fidelity=1)
        result = proxy_states_to_candidates(proxy, env)
        assert len(result) == 2
        assert result[0].x == pytest.approx((0.1, 0.2), abs=1e-6)
        assert result[0].fidelity == 1
        assert result[1].x == pytest.approx((0.3, 0.4), abs=1e-6)
        assert result[1].fidelity == 2

    def test_fidelity_first_layout(self):
        """Works when idx_fidelity=0 and idx_base_env=1 (FidFirst wrapper)."""
        proxy = _make_mf_proxy(
            [[0.5, 0.6]], fidelity_indices=[2], idx_base=1, idx_fid=0
        )
        env = _StubMFEnv(proxy_return=proxy, idx_base_env=1, idx_fidelity=0)
        result = proxy_states_to_candidates(proxy, env)
        assert result[0].x == pytest.approx((0.5, 0.6), abs=1e-6)
        assert result[0].fidelity == 2

    def test_non_tensor_fid_raw(self):
        """fid_raw may be a plain Python sequence, not a tensor."""
        proxy = [{0: torch.tensor([0.1, 0.2]), 1: [3]}]
        env = _StubMFEnv(proxy_return=proxy, idx_base_env=0, idx_fidelity=1)
        result = proxy_states_to_candidates(proxy, env)
        assert result[0].fidelity == 3

    def test_returns_candidate_objects(self):
        proxy = _make_mf_proxy([[0.0, 1.0]], fidelity_indices=[1])
        env = _StubMFEnv(proxy_return=proxy)
        result = proxy_states_to_candidates(proxy, env)
        assert all(isinstance(c, Candidate) for c in result)

    def test_coordinates_are_python_floats(self):
        proxy = _make_mf_proxy([[1.0, 2.0]], fidelity_indices=[1])
        env = _StubMFEnv(proxy_return=proxy)
        result = proxy_states_to_candidates(proxy, env)
        assert all(isinstance(v, float) for v in result[0].x)

    def test_empty_states_proxy_returns_empty(self):
        env = _StubMFEnv(proxy_return=[])
        assert proxy_states_to_candidates([], env) == []

    # -----------------------------------------------------------------------
    # fidelity_map: mapping 1-based raw Choice indices to domain values
    # -----------------------------------------------------------------------

    def test_fidelity_map_translates_raw_index(self):
        """fidelity_map=[1,2,3] maps raw 1-based index: 1→1, 2→2, 3→3.

        The GFlowNet Choice env produces 1-based states (1..N), so
        fidelity_map[raw_index - 1] is used internally.
        """
        # Raw fidelities [1, 2, 3] simulate real Choice env output.
        proxy = _make_mf_proxy(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], fidelity_indices=[1, 2, 3]
        )
        env = _StubMFEnv(proxy_return=proxy, idx_base_env=0, idx_fidelity=1)
        result = proxy_states_to_candidates(proxy, env, fidelity_map=[1, 2, 3])
        assert result[0].fidelity == 1
        assert result[1].fidelity == 2
        assert result[2].fidelity == 3

    def test_fidelity_map_non_contiguous_values(self):
        """fidelity_map can contain arbitrary non-contiguous integers.

        Raw indices 1, 2 (Choice env) are mapped to fidelity_map[0], fidelity_map[1].
        """
        proxy = _make_mf_proxy([[0.1, 0.2], [0.3, 0.4]], fidelity_indices=[1, 2])
        env = _StubMFEnv(proxy_return=proxy, idx_base_env=0, idx_fidelity=1)
        result = proxy_states_to_candidates(proxy, env, fidelity_map=[5, 10])
        assert result[0].fidelity == 5
        assert result[1].fidelity == 10

    def test_fidelity_map_values_not_starting_from_zero(self):
        """fidelity_map values not starting from 0 or 1 are stamped correctly.

        Raw 1-based indices from Choice env are remapped via fidelity_map.
        """
        proxy = _make_mf_proxy(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], fidelity_indices=[1, 2, 3]
        )
        env = _StubMFEnv(proxy_return=proxy, idx_base_env=0, idx_fidelity=1)
        fidelity_map = [7, 42, 100]
        result = proxy_states_to_candidates(proxy, env, fidelity_map=fidelity_map)
        assert result[0].fidelity == 7
        assert result[1].fidelity == 42
        assert result[2].fidelity == 100

    def test_fidelity_map_none_returns_raw_index(self):
        """When fidelity_map=None, the raw 1-based index is returned unchanged."""
        proxy = _make_mf_proxy([[0.1, 0.2], [0.3, 0.4]], fidelity_indices=[1, 2])
        env = _StubMFEnv(proxy_return=proxy, idx_base_env=0, idx_fidelity=1)
        result = proxy_states_to_candidates(proxy, env, fidelity_map=None)
        assert result[0].fidelity == 1
        assert result[1].fidelity == 2

    def test_fidelity_map_ignored_for_single_fidelity(self):
        """fidelity_map has no effect on single-fidelity envs (fidelity stays None)."""
        coords = torch.tensor([[0.1, 0.2]])
        result = proxy_states_to_candidates(coords, _plain_env(), fidelity_map=[99])
        assert result[0].fidelity is None
