"""Tests for budget schedule implementations.

Covers correctness of the sigmoid schedule and overflow safety for
extreme-but-valid steepness values.
"""

import pytest

from activelearning.budget.schedule_config import sigmoid_iteration_schedule


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

    def test_extreme_steepness_does_not_raise(self):
        """Very large steepness must not raise OverflowError."""
        schedule = sigmoid_iteration_schedule(
            total_budget=100.0,
            num_iterations=10,
            midpoint_fraction=0.5,
            steepness=1e6,
        )
        total = sum(schedule(i) for i in range(10))
        assert total == pytest.approx(100.0, rel=1e-6)

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
