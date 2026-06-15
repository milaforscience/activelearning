"""Tests for budget schedule implementations.

Covers correctness of the sigmoid schedule and overflow safety for
extreme-but-valid steepness values.
"""

import pytest

from activelearning.budget.budget import Budget
from activelearning.budget.budget_schedule import sigmoid_iteration_schedule


class TestSigmoidIterationSchedule:
    def test_allocations_sum_to_total_budget(self):
        schedule = sigmoid_iteration_schedule(
            total_budget=100.0,
            num_iterations=10,
            midpoint_fraction=0.5,
            steepness=10.0,
        )
        total = sum(schedule(i) for i in range(10))
        assert total == pytest.approx(100.0, rel=1e-9)

    def test_all_allocations_non_negative(self):
        schedule = sigmoid_iteration_schedule(
            total_budget=50.0,
            num_iterations=20,
            midpoint_fraction=0.5,
            steepness=5.0,
        )
        for i in range(20):
            assert schedule(i) >= 0.0

    def test_out_of_range_rounds_return_zero(self):
        schedule = sigmoid_iteration_schedule(
            total_budget=10.0,
            num_iterations=5,
            midpoint_fraction=0.5,
            steepness=10.0,
        )
        assert schedule(-1) == 0.0
        assert schedule(5) == 0.0
        assert schedule(100) == 0.0

    def test_extreme_steepness_produces_zero_rounds(self):
        """Very large steepness produces zero-budget rounds (caught by validate_schedule)."""
        schedule = sigmoid_iteration_schedule(
            total_budget=100.0,
            num_iterations=10,
            midpoint_fraction=0.5,
            steepness=1e6,
        )
        # Some rounds get zero — validate_schedule will catch this at runtime
        allocations = [schedule(i) for i in range(10)]
        assert any(a == 0.0 for a in allocations)
        assert sum(allocations) == pytest.approx(100.0, rel=1e-6)

        budget = Budget(available_budget=100.0, schedule=schedule)
        with pytest.raises(ValueError, match="less than the minimum oracle query cost"):
            budget.validate_schedule(min_query_cost=1.0)

    def test_near_zero_steepness_falls_back_to_uniform(self):
        """When steepness is so small that all CDF increments are zero,
        allocations must fall back to uniform rather than raising ZeroDivisionError."""
        schedule = sigmoid_iteration_schedule(
            total_budget=100.0,
            num_iterations=5,
            midpoint_fraction=0.5,
            steepness=1e-300,  # Effectively flat sigmoid → zero increments in float32
        )
        total = sum(schedule(i) for i in range(5))
        assert total == pytest.approx(100.0, rel=1e-9)
        # Each round should receive an equal share
        for i in range(5):
            assert schedule(i) == pytest.approx(20.0)
        """Sigmoid schedule should front-load budget toward the midpoint and beyond."""
        schedule = sigmoid_iteration_schedule(
            total_budget=100.0,
            num_iterations=10,
            midpoint_fraction=0.5,
            steepness=10.0,
        )
        early = sum(schedule(i) for i in range(3))
        late = sum(schedule(i) for i in range(7, 10))
        assert late > early


class TestSigmoidParameterValidation:
    """Tests for sigmoid_iteration_schedule input validation."""

    def test_negative_total_budget_raises(self):
        with pytest.raises(ValueError, match="total_budget must be positive"):
            sigmoid_iteration_schedule(
                total_budget=-10.0,
                num_iterations=5,
                midpoint_fraction=0.5,
                steepness=5.0,
            )

    def test_zero_total_budget_raises(self):
        with pytest.raises(ValueError, match="total_budget must be positive"):
            sigmoid_iteration_schedule(
                total_budget=0.0,
                num_iterations=5,
                midpoint_fraction=0.5,
                steepness=5.0,
            )

    def test_zero_num_iterations_raises(self):
        with pytest.raises(ValueError, match="num_iterations must be positive"):
            sigmoid_iteration_schedule(
                total_budget=100.0,
                num_iterations=0,
                midpoint_fraction=0.5,
                steepness=5.0,
            )

    def test_midpoint_fraction_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="midpoint_fraction must be in"):
            sigmoid_iteration_schedule(
                total_budget=100.0,
                num_iterations=5,
                midpoint_fraction=0.0,
                steepness=5.0,
            )
        with pytest.raises(ValueError, match="midpoint_fraction must be in"):
            sigmoid_iteration_schedule(
                total_budget=100.0,
                num_iterations=5,
                midpoint_fraction=1.0,
                steepness=5.0,
            )

    def test_negative_steepness_raises(self):
        with pytest.raises(ValueError, match="steepness must be positive"):
            sigmoid_iteration_schedule(
                total_budget=100.0,
                num_iterations=5,
                midpoint_fraction=0.5,
                steepness=-1.0,
            )
