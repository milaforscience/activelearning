import logging

import pytest

from activelearning.budget.budget import Budget, _BUDGET_ATOL


@pytest.fixture
def constant_schedule():
    """Schedule that returns constant budget per round."""

    def schedule(round):
        return 10.0

    return schedule


@pytest.fixture
def linear_schedule():
    """Schedule with linearly increasing budget per round."""

    def schedule(round):
        return 5.0 * (round + 1)

    return schedule


@pytest.fixture
def exponential_schedule():
    """Schedule with exponentially increasing budget per round."""

    def schedule(round):
        return 2.0**round

    return schedule


def test_budget_initialization(constant_schedule):
    """Test that Budget initializes with correct attributes."""
    budget = Budget(available_budget=100.0, schedule=constant_schedule)
    assert budget.available_budget == 100.0
    assert budget.schedule == constant_schedule
    assert budget.max_rounds is None


def test_budget_initialization_with_max_rounds(constant_schedule):
    """Test that Budget stores an optional round limit."""
    budget = Budget(
        available_budget=100.0,
        schedule=constant_schedule,
        max_rounds=3,
    )
    assert budget.max_rounds == 3


@pytest.mark.parametrize("max_rounds", [0, -1])
def test_budget_rejects_non_positive_max_rounds(constant_schedule, max_rounds):
    """Test that a configured round limit must be positive."""
    with pytest.raises(ValueError, match="max_rounds must be a positive integer"):
        Budget(
            available_budget=100.0,
            schedule=constant_schedule,
            max_rounds=max_rounds,
        )


def test_get_round_budget_constant_schedule(constant_schedule):
    """Test round budget calculation with constant schedule."""
    budget = Budget(available_budget=100.0, schedule=constant_schedule)
    assert budget.get_round_budget(0) == 10.0
    assert budget.get_round_budget(5) == 10.0
    assert budget.get_round_budget(10) == 10.0


def test_get_round_budget_linear_schedule(linear_schedule):
    """Test round budget calculation with linear schedule."""
    budget = Budget(available_budget=100.0, schedule=linear_schedule)
    assert budget.get_round_budget(0) == 5.0
    assert budget.get_round_budget(1) == 10.0
    assert budget.get_round_budget(2) == 15.0


def test_get_round_budget_exponential_schedule(exponential_schedule):
    """Test round budget calculation with exponential schedule."""
    budget = Budget(available_budget=100.0, schedule=exponential_schedule)
    assert budget.get_round_budget(0) == 1.0
    assert budget.get_round_budget(1) == 2.0
    assert budget.get_round_budget(2) == 4.0
    assert budget.get_round_budget(3) == 8.0


def test_get_round_budget_capped_at_available(constant_schedule, caplog):
    """Test that round budget is capped at available_budget with warning."""
    budget = Budget(available_budget=5.0, schedule=constant_schedule)

    with caplog.at_level(logging.WARNING):
        round_budget = budget.get_round_budget(0)

    assert round_budget == 5.0  # Capped at available
    assert "exceeds available budget" in caplog.text
    assert "Capping at available budget" in caplog.text


def test_consume_success():
    """Test that consume correctly deducts cost from available budget."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)

    budget.consume(30.0)
    assert budget.available_budget == 70.0

    budget.consume(20.0)
    assert budget.available_budget == 50.0


def test_consume_exact_budget():
    """Test consuming exactly the available budget."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)

    budget.consume(100.0)
    assert budget.available_budget == 0.0


def test_consume_raises_when_exceeds_budget():
    """Test that consume raises ValueError when cost exceeds available budget."""
    budget = Budget(available_budget=50.0, schedule=lambda r: 10.0)

    with pytest.raises(ValueError, match="Cost .* exceeds available budget"):
        budget.consume(60.0)

    # Budget should remain unchanged after failed consume
    assert budget.available_budget == 50.0


def test_consume_multiple_calls_deplete_budget():
    """Test multiple consume calls correctly deplete budget."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)

    budget.consume(25.0)
    budget.consume(25.0)
    budget.consume(25.0)
    budget.consume(25.0)

    assert budget.available_budget == 0.0


def test_consume_after_partial_depletion():
    """Test that consume works correctly after partial budget consumption."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)

    budget.consume(70.0)
    assert budget.available_budget == 30.0

    # This should succeed
    budget.consume(30.0)
    assert budget.available_budget == 0.0

    # This should fail
    with pytest.raises(ValueError):
        budget.consume(1.0)


def test_zero_budget():
    """Test Budget with zero initial budget."""
    budget = Budget(available_budget=0.0, schedule=lambda r: 10.0)

    assert budget.available_budget == 0.0
    assert budget.get_round_budget(0) == 0.0

    with pytest.raises(ValueError):
        budget.consume(1.0)


def test_negative_schedule_value(caplog):
    """Test handling of negative values from schedule."""

    def negative_schedule(r):
        return -10.0

    budget = Budget(available_budget=100.0, schedule=negative_schedule)

    # Negative schedule should return as-is (no capping for negative)
    round_budget = budget.get_round_budget(0)
    assert round_budget == -10.0


def test_schedule_exceeds_after_consumption(caplog):
    """Test capping when schedule exceeds available budget after consumption."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 50.0)

    # First round: 50.0 allocated, no capping
    round_budget_0 = budget.get_round_budget(0)
    assert round_budget_0 == 50.0

    # Consume 80.0, leaving 20.0 available
    budget.consume(80.0)
    assert budget.available_budget == 20.0

    # Second round: schedule wants 50.0 but only 20.0 available
    with caplog.at_level(logging.WARNING):
        round_budget_1 = budget.get_round_budget(1)

    assert round_budget_1 == 20.0
    assert "exceeds available budget" in caplog.text


def test_can_afford_sufficient_budget():
    """Test can_afford returns True when budget is sufficient."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
    assert budget.can_afford(50.0) is True
    assert budget.can_afford(100.0) is True
    assert budget.can_afford(1.0) is True


def test_can_afford_insufficient_budget():
    """Test can_afford returns False when budget is insufficient."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
    assert budget.can_afford(101.0) is False
    assert budget.can_afford(200.0) is False


def test_can_afford_exact_budget():
    """Test can_afford returns True when cost equals available budget."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
    assert budget.can_afford(100.0) is True


def test_can_afford_zero_cost():
    """Test can_afford with zero cost."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
    assert budget.can_afford(0.0) is True


def test_can_afford_no_side_effects():
    """Test that can_afford does not modify available_budget."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
    initial_budget = budget.available_budget

    # Call can_afford multiple times
    budget.can_afford(50.0)
    budget.can_afford(150.0)
    budget.can_afford(100.0)

    # Budget should remain unchanged
    assert budget.available_budget == initial_budget


class TestValidateSchedule:
    """Tests for Budget.validate_schedule method."""

    def test_valid_schedule_passes(self):
        """Schedule with sufficient allocations should not raise."""
        budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
        # No exception expected
        budget.validate_schedule(min_query_cost=5.0)

    def test_underfunded_round_raises(self):
        """Schedule with a round below min_query_cost should raise."""

        # Schedule: round 0 gets 1.0, rounds 1-9 get 10.0 each (total 91)
        def stingy_start(r: int) -> float:
            if r >= 10:
                return 0.0
            return 1.0 if r == 0 else 10.0

        budget = Budget(available_budget=91.0, schedule=stingy_start)
        with pytest.raises(ValueError, match="less than the minimum oracle query cost"):
            budget.validate_schedule(min_query_cost=5.0)

    def test_all_rounds_underfunded_raises(self):
        """Schedule where every round is below min cost should raise."""

        # 100 rounds of 0.5 each, but schedule signals end after that
        def low_schedule(r: int) -> float:
            return 0.5 if r < 200 else 0.0

        budget = Budget(available_budget=100.0, schedule=low_schedule)
        with pytest.raises(ValueError, match="less than the minimum oracle query cost"):
            budget.validate_schedule(min_query_cost=1.0)

    def test_exact_min_cost_passes(self):
        """Schedule allocating exactly min_query_cost should pass."""
        budget = Budget(available_budget=50.0, schedule=lambda r: 5.0)
        # Should not raise — exactly at the threshold
        budget.validate_schedule(min_query_cost=5.0)

    def test_capped_schedule_need_not_cover_full_budget(self):
        """A capped run may intentionally leave part of the total budget unused."""

        def schedule(round_index: int) -> float:
            return 10.0 if round_index < 3 else 0.0

        budget = Budget(
            available_budget=100.0,
            schedule=schedule,
            max_rounds=3,
        )

        budget.validate_schedule(min_query_cost=5.0)

    def test_schedule_with_leading_zeros_are_underfunded(self):
        """Leading zero-budget rounds should fail when a positive min query cost is required."""
        allocations = [0.0, 0.0, 10.0, 20.0, 20.0]

        def delayed_start_schedule(round_index: int) -> float:
            return allocations[round_index] if round_index < len(allocations) else 0.0

        budget = Budget(available_budget=50.0, schedule=delayed_start_schedule)
        with pytest.raises(ValueError, match="less than the minimum oracle query cost"):
            budget.validate_schedule(min_query_cost=1.0)

    def test_schedule_with_leading_zeros_and_insufficient_total_raises(self):
        """Validation should fail when total allocatable budget cannot cover available budget."""
        allocations = [0.0, 0.0, 1.0, 5.0, 10.0, 50.0, 10.0, 5.0, 1.0, 0.0, 0.0]

        def sparse_schedule(round_index: int) -> float:
            return allocations[round_index] if round_index < len(allocations) else 0.0

        budget = Budget(available_budget=100.0, schedule=sparse_schedule)
        with pytest.raises(ValueError, match="cannot cover available_budget"):
            budget.validate_schedule(min_query_cost=0.0)


class TestBudgetInitialization:
    """Tests for Budget.__init__ edge cases."""

    def test_negative_budget_raises(self):
        """Negative available_budget should raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            Budget(available_budget=-1.0, schedule=lambda r: 10.0)

    def test_integer_budget_coerced_to_float(self):
        """Integer available_budget should be coerced to float."""
        budget = Budget(available_budget=100, schedule=lambda r: 10.0)
        assert isinstance(budget.available_budget, float)
        assert budget.available_budget == 100.0

    def test_large_budget_accepted(self):
        """Very large budget values should be accepted."""
        budget = Budget(available_budget=1e15, schedule=lambda r: 1e12)
        assert budget.available_budget == 1e15


class TestConsumeFloatingPoint:
    """Tests for floating-point tolerance in Budget.consume."""

    def test_consume_within_tolerance_succeeds(self):
        """Consuming slightly more than budget within _BUDGET_ATOL should succeed."""
        budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
        # Cost exceeds budget by less than _BUDGET_ATOL
        budget.consume(100.0 + _BUDGET_ATOL * 0.5)
        assert budget.available_budget == 0.0

    def test_consume_beyond_tolerance_raises(self):
        """Consuming more than budget beyond _BUDGET_ATOL should raise."""
        budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
        with pytest.raises(ValueError, match="exceeds available budget"):
            budget.consume(100.0 + _BUDGET_ATOL * 10)

    def test_consume_zero_cost(self):
        """Consuming zero cost should leave budget unchanged."""
        budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
        budget.consume(0.0)
        assert budget.available_budget == 100.0

    def test_consume_floors_at_zero(self):
        """Budget should never go negative after consume (floored at 0.0)."""
        budget = Budget(available_budget=10.0, schedule=lambda r: 10.0)
        # Tiny floating-point overshoot within tolerance
        budget.consume(10.0 + _BUDGET_ATOL * 0.1)
        assert budget.available_budget == 0.0

    def test_consume_accumulation_drift(self):
        """Many small consumes should not cause floating-point drift issues."""
        budget = Budget(available_budget=1.0, schedule=lambda r: 0.1)
        for _ in range(10):
            budget.consume(0.1)
        assert budget.available_budget == pytest.approx(0.0, abs=_BUDGET_ATOL)


class TestCanAffordFloatingPoint:
    """Tests for floating-point tolerance in Budget.can_afford."""

    def test_can_afford_within_tolerance(self):
        """Cost slightly above budget within _BUDGET_ATOL should be affordable."""
        budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
        assert budget.can_afford(100.0 + _BUDGET_ATOL * 0.5) is True

    def test_cannot_afford_beyond_tolerance(self):
        """Cost above budget beyond _BUDGET_ATOL should not be affordable."""
        budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
        assert budget.can_afford(100.0 + _BUDGET_ATOL * 10) is False

    def test_can_afford_after_consumption(self):
        """can_afford should reflect remaining budget after consumption."""
        budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
        budget.consume(60.0)
        assert budget.can_afford(40.0) is True
        assert budget.can_afford(41.0) is False

    def test_can_afford_zero_budget(self):
        """Zero remaining budget can only afford zero cost."""
        budget = Budget(available_budget=0.0, schedule=lambda r: 10.0)
        assert budget.can_afford(0.0) is True
        assert budget.can_afford(_BUDGET_ATOL * 0.5) is True
        assert budget.can_afford(_BUDGET_ATOL * 10) is False


class TestGetRoundBudgetEdgeCases:
    """Edge case tests for Budget.get_round_budget."""

    def test_get_round_budget_does_not_consume(self):
        """get_round_budget should not modify available_budget."""
        budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
        budget.get_round_budget(0)
        budget.get_round_budget(1)
        assert budget.available_budget == 100.0

    def test_get_round_budget_with_zero_schedule(self):
        """Schedule returning zero should return zero without warning."""
        budget = Budget(available_budget=100.0, schedule=lambda r: 0.0)
        assert budget.get_round_budget(0) == 0.0

    def test_get_round_budget_exact_available(self):
        """Schedule returning exactly available budget should not warn."""
        budget = Budget(available_budget=10.0, schedule=lambda r: 10.0)
        assert budget.get_round_budget(0) == 10.0
