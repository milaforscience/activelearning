"""Tests for PendingAwareGreedyFantasySelector."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from activelearning.selector.pending_aware_greedy_selector import (
    PendingAwareGreedyFantasySelector,
)
from activelearning.utils.types import Candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidates(n: int) -> list[Candidate]:
    return [Candidate(x=[float(i)]) for i in range(n)]


def make_acquisition(scores: list[float], supports_pending: bool = True) -> Any:
    """Return a mock acquisition whose score_with_pending returns fixed scores."""
    acq = MagicMock()
    acq.supports_pending_scoring = supports_pending
    acq.score_with_pending.side_effect = lambda candidates, pending: [
        scores[int(c.x[0])] for c in candidates
    ]
    return acq


def uniform_cost_fn(candidates: Any) -> list[float]:
    return [1.0] * len(list(candidates))


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_k_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            PendingAwareGreedyFantasySelector(k=0)

    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            PendingAwareGreedyFantasySelector(k=-1)

    def test_valid_k_constructs(self) -> None:
        sel = PendingAwareGreedyFantasySelector(k=3)
        assert sel.k == 3


# ---------------------------------------------------------------------------
# Input validation inside select()
# ---------------------------------------------------------------------------


class TestSelectValidation:
    def test_missing_acquisition_raises(self) -> None:
        sel = PendingAwareGreedyFantasySelector(k=2)
        with pytest.raises(ValueError, match="requires an acquisition"):
            sel.select(make_candidates(3))

    def test_acquisition_without_pending_support_raises(self) -> None:
        sel = PendingAwareGreedyFantasySelector(k=2)
        acq = make_acquisition([1.0, 2.0, 3.0], supports_pending=False)
        with pytest.raises(ValueError, match="supports_pending_scoring"):
            sel.select(make_candidates(3), acquisition=acq)

    def test_cost_fn_without_budget_raises(self) -> None:
        sel = PendingAwareGreedyFantasySelector(k=2)
        acq = make_acquisition([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Both cost_fn and round_budget"):
            sel.select(make_candidates(3), acquisition=acq, cost_fn=uniform_cost_fn)

    def test_budget_without_cost_fn_raises(self) -> None:
        sel = PendingAwareGreedyFantasySelector(k=2)
        acq = make_acquisition([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Both cost_fn and round_budget"):
            sel.select(make_candidates(3), acquisition=acq, round_budget=5.0)

    def test_empty_candidates_returns_empty(self) -> None:
        sel = PendingAwareGreedyFantasySelector(k=2)
        acq = make_acquisition([])
        assert sel.select([], acquisition=acq) == []


# ---------------------------------------------------------------------------
# Core greedy selection behaviour
# ---------------------------------------------------------------------------


class TestGreedySelection:
    def test_selects_k_candidates(self) -> None:
        """Selector returns exactly k candidates from a larger pool."""
        candidates = make_candidates(5)
        scores = [1.0, 5.0, 3.0, 4.0, 2.0]
        acq = make_acquisition(scores)
        sel = PendingAwareGreedyFantasySelector(k=3)
        result = sel.select(candidates, acquisition=acq)
        assert len(result) == 3

    def test_selects_best_first(self) -> None:
        """First selected candidate is the globally highest scorer."""
        candidates = make_candidates(4)
        scores = [1.0, 4.0, 2.0, 3.0]
        acq = make_acquisition(scores)
        sel = PendingAwareGreedyFantasySelector(k=1)
        result = sel.select(candidates, acquisition=acq)
        assert result[0] == candidates[1]  # score 4.0

    def test_returns_all_when_k_geq_pool_size(self) -> None:
        candidates = make_candidates(3)
        acq = make_acquisition([1.0, 2.0, 3.0])
        sel = PendingAwareGreedyFantasySelector(k=10)
        result = sel.select(candidates, acquisition=acq)
        assert len(result) == 3

    def test_no_duplicate_selections(self) -> None:
        candidates = make_candidates(4)
        acq = make_acquisition([1.0, 2.0, 3.0, 4.0])
        sel = PendingAwareGreedyFantasySelector(k=4)
        result = sel.select(candidates, acquisition=acq)
        assert len(result) == len(set(id(c) for c in result))

    def test_pending_grows_each_step(self) -> None:
        """score_with_pending is called with a growing pending set."""
        candidates = make_candidates(3)
        scores = [3.0, 2.0, 1.0]
        acq = make_acquisition(scores)
        sel = PendingAwareGreedyFantasySelector(k=3)
        sel.select(candidates, acquisition=acq)

        calls = acq.score_with_pending.call_args_list
        assert len(calls) == 3
        # First call: no pending
        assert list(calls[0].args[1]) == []
        # Second call: 1 pending
        assert len(calls[1].args[1]) == 1
        # Third call: 2 pending
        assert len(calls[2].args[1]) == 2


# ---------------------------------------------------------------------------
# Budget-constrained selection
# ---------------------------------------------------------------------------


class TestBudgetConstrainedSelection:
    def test_respects_budget(self) -> None:
        """With unit costs and budget=2, exactly 2 candidates are selected."""
        candidates = make_candidates(5)
        acq = make_acquisition([5.0, 4.0, 3.0, 2.0, 1.0])
        sel = PendingAwareGreedyFantasySelector(k=5)
        result = sel.select(
            candidates,
            acquisition=acq,
            cost_fn=uniform_cost_fn,
            round_budget=2.0,
        )
        assert len(result) == 2

    def test_returns_empty_when_all_unaffordable(self) -> None:
        candidates = make_candidates(3)
        acq = make_acquisition([3.0, 2.0, 1.0])
        sel = PendingAwareGreedyFantasySelector(k=3)
        result = sel.select(
            candidates,
            acquisition=acq,
            cost_fn=lambda _: [10.0, 10.0, 10.0],
            round_budget=5.0,
        )
        assert result == []

    def test_selects_affordable_subset(self) -> None:
        """With heterogeneous costs, skips candidates that would exceed budget."""
        candidates = make_candidates(4)
        # candidate 3 has score 4.0 but cost 10 — should be skipped
        acq = make_acquisition([1.0, 2.0, 3.0, 4.0])
        costs = [1.0, 1.0, 1.0, 10.0]

        def cost_fn(cands: Any) -> list[float]:
            return [costs[int(c.x[0])] for c in cands]

        sel = PendingAwareGreedyFantasySelector(k=3)
        result = sel.select(
            candidates,
            acquisition=acq,
            cost_fn=cost_fn,
            round_budget=3.0,
        )
        assert len(result) == 3
        assert candidates[3] not in result

    def test_negative_cost_raises(self) -> None:
        """Negative costs are rejected with a ValueError."""
        candidates = make_candidates(3)
        acq = make_acquisition([3.0, 2.0, 1.0])
        sel = PendingAwareGreedyFantasySelector(k=3)
        with pytest.raises(ValueError, match="negative cost"):
            sel.select(
                candidates,
                acquisition=acq,
                cost_fn=lambda _: [1.0, -0.5, 1.0],
                round_budget=5.0,
            )
