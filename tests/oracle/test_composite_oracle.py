import pytest

from activelearning.budget.budget import Budget
from activelearning.oracle.composite_oracle import CompositeOracle
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.utils.types import Candidate


@pytest.fixture
def budget():
    """Create a Budget with sufficient funds for testing."""

    def schedule(r):
        return 100.0

    return Budget(available_budget=1000.0, schedule=schedule)


@pytest.fixture
def cheap_oracle():
    """Create a cheaper oracle with lower costs."""

    def score_fn(x):
        return float(x) * 1.0

    return MultiFidelityOracle(
        fidelity_configs={
            0: {"cost_per_sample": 1.0, "score_fn": score_fn},
            1: {"cost_per_sample": 2.0, "score_fn": score_fn},
        }
    )


@pytest.fixture
def expensive_oracle():
    """Create a more expensive oracle with higher costs."""

    def score_fn(x):
        return float(x) * 2.0

    return MultiFidelityOracle(
        fidelity_configs={
            0: {"cost_per_sample": 3.0, "score_fn": score_fn},
            1: {"cost_per_sample": 5.0, "score_fn": score_fn},
            2: {"cost_per_sample": 7.0, "score_fn": score_fn},
        }
    )


def test_initialization_with_single_oracle(cheap_oracle):
    """Test CompositeOracle initializes with single sub-oracle."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle])
    assert len(composite._sub_oracles) == 1
    assert composite._sub_oracles[0] == cheap_oracle


def test_initialization_with_multiple_oracles(cheap_oracle, expensive_oracle):
    """Test CompositeOracle initializes with multiple sub-oracles."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle, expensive_oracle])
    assert len(composite._sub_oracles) == 2


def test_get_supported_fidelities_single_oracle(cheap_oracle):
    """Test get_supported_fidelities with single oracle."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle])
    assert composite.get_supported_fidelities() == [0, 1]


def test_get_supported_fidelities_multiple_oracles(cheap_oracle, expensive_oracle):
    """Test get_supported_fidelities combines all fidelities from sub-oracles."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle, expensive_oracle])
    # cheap: [0, 1], expensive: [0, 1, 2] -> combined: [0, 1, 2]
    assert composite.get_supported_fidelities() == [0, 1, 2]


def test_get_costs_single_fidelity(cheap_oracle, expensive_oracle):
    """Test get_costs returns cheapest cost for each candidate."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle, expensive_oracle])
    candidates = [Candidate(x=i, fidelity=0) for i in range(3)]
    costs = composite.get_costs(candidates)

    # cheap oracle has cost 1.0 for fidelity 0, expensive has 3.0
    assert costs == [1.0, 1.0, 1.0]


def test_get_costs_mixed_fidelities(cheap_oracle, expensive_oracle):
    """Test get_costs with mixed fidelities selects cheapest for each."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle, expensive_oracle])
    candidates = [
        Candidate(x=1, fidelity=0),  # cheap=1.0, expensive=3.0 -> 1.0
        Candidate(x=2, fidelity=1),  # cheap=2.0, expensive=5.0 -> 2.0
        Candidate(x=3, fidelity=2),  # only expensive=7.0 -> 7.0
    ]
    costs = composite.get_costs(candidates)

    assert costs == [1.0, 2.0, 7.0]


def test_get_costs_unsupported_fidelity(cheap_oracle):
    """Test get_costs raises error for unsupported fidelity."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle])
    candidates = [Candidate(x=1, fidelity=999)]

    with pytest.raises(ValueError, match="No oracle supports fidelity 999"):
        composite.get_costs(candidates)


def test_query_routes_to_cheapest_oracle(cheap_oracle, expensive_oracle, budget):
    """Test query routes candidates to cheapest oracle."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle, expensive_oracle])
    candidates = [Candidate(x=i, fidelity=0) for i in [10, 20, 30]]

    observations = composite.query(candidates)

    # Should use cheap oracle (score_fn = x * 1.0)
    assert len(observations) == 3
    for i, obs in enumerate(observations):
        assert obs.x == [10, 20, 30][i]
        assert obs.y == float([10, 20, 30][i]) * 1.0
        assert obs.fidelity == 0


def test_query_mixed_fidelities_routes_correctly(
    cheap_oracle, expensive_oracle, budget
):
    """Test query routes mixed fidelity candidates to correct oracles."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle, expensive_oracle])
    candidates = [
        Candidate(x=10, fidelity=0),  # -> cheap oracle
        Candidate(x=20, fidelity=1),  # -> cheap oracle
        Candidate(x=30, fidelity=2),  # -> expensive oracle (only one supporting fid 2)
    ]

    observations = composite.query(candidates)

    assert len(observations) == 3
    # First two from cheap oracle (score = x * 1.0)
    assert observations[0].y == 10.0
    assert observations[1].y == 20.0
    # Last one from expensive oracle (score = x * 2.0)
    assert observations[2].y == 60.0


def test_query_consumes_budget(cheap_oracle, expensive_oracle):
    """Test that explicit budget consumption works correctly."""

    def schedule(r):
        return 50.0

    budget = Budget(available_budget=100.0, schedule=schedule)
    composite = CompositeOracle(sub_oracles=[cheap_oracle, expensive_oracle])
    candidates = [
        Candidate(x=1, fidelity=0),  # cheap: 1.0
        Candidate(x=2, fidelity=1),  # cheap: 2.0
        Candidate(x=3, fidelity=2),  # expensive: 7.0
    ]  # Total: 10.0

    costs = composite.get_costs(candidates)
    budget.consume(sum(costs))
    composite.query(candidates)

    assert budget.available_budget == 90.0


def test_query_raises_when_budget_insufficient(cheap_oracle):
    """Test that budget consumption raises ValueError when budget is insufficient."""

    def schedule(r):
        return 10.0

    budget = Budget(available_budget=5.0, schedule=schedule)
    composite = CompositeOracle(sub_oracles=[cheap_oracle])
    candidates = [Candidate(x=i, fidelity=0) for i in range(10)]  # Total cost: 10.0

    costs = composite.get_costs(candidates)
    with pytest.raises(ValueError, match="Cost .* exceeds available budget"):
        budget.consume(sum(costs))


def test_query_preserves_candidate_order(cheap_oracle, expensive_oracle, budget):
    """Test that query preserves the order of candidates."""
    composite = CompositeOracle(sub_oracles=[cheap_oracle, expensive_oracle])
    candidates = [
        Candidate(x=5, fidelity=2),  # expensive only
        Candidate(x=10, fidelity=0),  # cheap
        Candidate(x=15, fidelity=1),  # cheap
        Candidate(x=20, fidelity=2),  # expensive only
        Candidate(x=25, fidelity=0),  # cheap
    ]

    observations = composite.query(candidates)

    # Verify order matches input candidates
    assert observations[0].x == 5
    assert observations[1].x == 10
    assert observations[2].x == 15
    assert observations[3].x == 20
    assert observations[4].x == 25


def test_cheapest_oracle_selection_with_equal_costs():
    """Test that when costs are equal, first oracle is selected."""

    def score_fn_1(x):
        return float(x) * 1.0

    def score_fn_2(x):
        return float(x) * 2.0

    oracle1 = MultiFidelityOracle(
        fidelity_configs={0: {"cost_per_sample": 2.0, "score_fn": score_fn_1}}
    )
    oracle2 = MultiFidelityOracle(
        fidelity_configs={0: {"cost_per_sample": 2.0, "score_fn": score_fn_2}}
    )

    composite = CompositeOracle(sub_oracles=[oracle1, oracle2])
    candidates = [Candidate(x=10, fidelity=0)]

    observations = composite.query(candidates)

    # Should use oracle1 (first in list when costs are equal)
    assert observations[0].y == 10.0  # score_fn_1: x * 1.0


def test_empty_candidates(cheap_oracle):
    """Test query with empty candidates list."""

    def schedule(r):
        return 100.0

    budget = Budget(available_budget=1000.0, schedule=schedule)
    composite = CompositeOracle(sub_oracles=[cheap_oracle])
    candidates = []

    observations = composite.query(candidates)

    assert observations == []
    assert budget.available_budget == 1000.0  # No budget consumed


def test_composite_of_composite_oracles():
    """Test that CompositeOracle works with nested composite oracles."""

    def score_fn_1(x):
        return float(x) * 1.0

    def score_fn_2(x):
        return float(x) * 2.0

    def score_fn_3(x):
        return float(x) * 3.0

    def score_fn_4(x):
        return float(x) * 4.0

    # Create base oracles with different costs and fidelities
    oracle_a = MultiFidelityOracle(
        fidelity_configs={
            0: {"cost_per_sample": 1.0, "score_fn": score_fn_1},
            1: {"cost_per_sample": 2.0, "score_fn": score_fn_1},
        }
    )

    oracle_b = MultiFidelityOracle(
        fidelity_configs={
            0: {"cost_per_sample": 3.0, "score_fn": score_fn_2},
            2: {"cost_per_sample": 5.0, "score_fn": score_fn_2},
        }
    )

    oracle_c = MultiFidelityOracle(
        fidelity_configs={
            1: {"cost_per_sample": 1.5, "score_fn": score_fn_3},
            2: {"cost_per_sample": 4.0, "score_fn": score_fn_3},
        }
    )

    oracle_d = MultiFidelityOracle(
        fidelity_configs={
            3: {"cost_per_sample": 10.0, "score_fn": score_fn_4},
        }
    )

    # Create first level composite oracles
    composite_1 = CompositeOracle(
        sub_oracles=[oracle_a, oracle_b]
    )  # Supports fid 0, 1, 2
    composite_2 = CompositeOracle(
        sub_oracles=[oracle_c, oracle_d]
    )  # Supports fid 1, 2, 3

    # Create top-level composite of composites
    top_composite = CompositeOracle(sub_oracles=[composite_1, composite_2])

    # Test fidelity support propagates through nesting
    assert set(top_composite.get_supported_fidelities()) == {0, 1, 2, 3}

    # Test cost calculation through nesting
    candidates = [
        Candidate(x=10, fidelity=0),  # Should route to oracle_a (cost 1.0)
        Candidate(
            x=20, fidelity=1
        ),  # Should route to oracle_c (cost 1.5) - cheaper than oracle_a
        Candidate(
            x=30, fidelity=2
        ),  # Should route to oracle_c (cost 4.0) - cheaper than oracle_b
        Candidate(
            x=40, fidelity=3
        ),  # Should route to oracle_d (cost 10.0) - only option
    ]

    costs = top_composite.get_costs(candidates)
    assert costs == [1.0, 1.5, 4.0, 10.0]
    assert sum(costs) == 16.5

    # Test query through nesting
    observations = top_composite.query(candidates)

    assert len(observations) == 4
    assert observations[0].x == 10
    assert observations[0].y == 10.0  # oracle_a: x * 1.0
    assert observations[0].fidelity == 0

    assert observations[1].x == 20
    assert observations[1].y == 60.0  # oracle_c: x * 3.0 (cheaper at fid 1)
    assert observations[1].fidelity == 1

    assert observations[2].x == 30
    assert observations[2].y == 90.0  # oracle_c: x * 3.0
    assert observations[2].fidelity == 2

    assert observations[3].x == 40
    assert observations[3].y == 160.0  # oracle_d: x * 4.0
    assert observations[3].fidelity == 3

    # Test with budget consumption
    budget = Budget(available_budget=100.0, schedule=lambda r: 50.0)
    budget_costs = top_composite.get_costs(candidates)
    budget.consume(sum(budget_costs))
    top_composite.query(candidates)

    assert budget.available_budget == 83.5  # 100 - 16.5
