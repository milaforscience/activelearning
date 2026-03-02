import logging

import pytest

from activelearning.budget.budget import Budget


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
