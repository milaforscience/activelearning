import pytest

from activelearning.budget.budget import Budget
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.utils.types import Candidate, Observation


@pytest.fixture
def budget():
    """Create a Budget with sufficient funds for testing."""

    def schedule(r):
        return 100.0

    return Budget(available_budget=1000.0, schedule=schedule)


def test_initialization_with_valid_configs():
    """Test MultiFidelityOracle initializes with valid fidelity configs."""

    def score_fn(x):
        return float(x)

    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            },
            1: {
                "cost_per_sample": 2.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            },
        }
    )
    assert oracle.get_supported_fidelities() == [0, 1]


def test_initialization_fails_with_missing_cost():
    """Test initialization fails if cost_per_sample is missing."""

    def score_fn(x):
        return float(x)

    with pytest.raises(ValueError, match="Missing 'cost_per_sample'"):
        MultiFidelityOracle(
            fidelity_configs={0: {"score_fn": score_fn, "fidelity_confidence": 1.0}}
        )


def test_initialization_fails_with_missing_score_fn():
    """Test initialization fails if score_fn is missing."""
    with pytest.raises(ValueError, match="Missing 'score_fn'"):
        MultiFidelityOracle(fidelity_configs={0: {"cost_per_sample": 1.0}})


def test_initialization_fails_with_missing_fidelity_confidence():
    """Test initialization fails if fidelity_confidence is missing."""

    def score_fn(x):
        return float(x)

    with pytest.raises(ValueError, match="Missing 'fidelity_confidence'"):
        MultiFidelityOracle(
            fidelity_configs={0: {"cost_per_sample": 1.0, "score_fn": score_fn}}
        )


def test_initialization_fails_with_out_of_range_fidelity_confidence():
    """Test initialization fails when fidelity_confidence is outside [0, 1]."""

    def score_fn(x):
        return float(x)

    with pytest.raises(ValueError, match=r"fidelity_confidence must be in \[0, 1\]"):
        MultiFidelityOracle(
            fidelity_configs={
                0: {
                    "cost_per_sample": 1.0,
                    "score_fn": score_fn,
                    "fidelity_confidence": 1.5,
                }
            }
        )


def test_initialization_fails_with_non_int_fidelity():
    """Test initialization fails if fidelity is not an integer."""

    def score_fn(x):
        return float(x)

    with pytest.raises(ValueError, match="Fidelity must be int"):
        MultiFidelityOracle(
            fidelity_configs={
                "low": {
                    "cost_per_sample": 1.0,
                    "score_fn": score_fn,
                    "fidelity_confidence": 1.0,
                }
            }
        )


def test_get_supported_fidelities():
    """Test get_supported_fidelities returns correct fidelity levels."""

    def score_fn(x):
        return float(x)

    oracle = MultiFidelityOracle(
        fidelity_configs={
            2: {
                "cost_per_sample": 1.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            },
            0: {
                "cost_per_sample": 2.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            },
            1: {
                "cost_per_sample": 1.5,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            },
        }
    )
    assert oracle.get_supported_fidelities() == [0, 1, 2]


def test_get_fidelity_confidences():
    """Test get_fidelity_confidences returns the expected mapping."""

    def score_fn(x):
        return float(x)

    oracle = MultiFidelityOracle(
        fidelity_configs={
            2: {
                "cost_per_sample": 1.0,
                "score_fn": score_fn,
                "fidelity_confidence": 0.6,
            },
            0: {
                "cost_per_sample": 2.0,
                "score_fn": score_fn,
                "fidelity_confidence": 0.9,
            },
        }
    )

    assert oracle.get_fidelity_confidences() == {0: 0.9, 2: 0.6}


def test_get_costs_single_fidelity(budget):
    """Test get_costs for candidates with single fidelity."""

    def score_fn(x):
        return float(x)

    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.5,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            }
        }
    )
    candidates = [Candidate(x=i, fidelity=0) for i in range(5)]
    costs = oracle.get_costs(candidates)

    assert len(costs) == 5
    assert all(cost == 1.5 for cost in costs)


def test_get_costs_mixed_fidelities(budget):
    """Test get_costs for candidates with mixed fidelities."""

    def score_fn(x):
        return float(x)

    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            },
            1: {
                "cost_per_sample": 2.5,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            },
        }
    )
    candidates = [
        Candidate(x=1, fidelity=0),
        Candidate(x=2, fidelity=1),
        Candidate(x=3, fidelity=0),
        Candidate(x=4, fidelity=1),
    ]
    costs = oracle.get_costs(candidates)

    assert costs == [1.0, 2.5, 1.0, 2.5]


def test_get_costs_unsupported_fidelity():
    """Test get_costs raises error for unsupported fidelity."""

    def score_fn(x):
        return float(x)

    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            }
        }
    )
    candidates = [Candidate(x=1, fidelity=5)]

    with pytest.raises(ValueError, match="Unsupported fidelity 5"):
        oracle.get_costs(candidates)


def test_query_single_fidelity(budget):
    """Test query with single fidelity candidates."""

    def score_fn(x):
        return float(x) * 2

    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            }
        }
    )
    candidates = [Candidate(x=i, fidelity=0) for i in [5, 10, 15]]
    observations = oracle.query(candidates)

    assert len(observations) == 3
    for i, obs in enumerate(observations):
        assert isinstance(obs, Observation)
        assert obs.x == [5, 10, 15][i]
        assert obs.y == [5, 10, 15][i] * 2
        assert obs.fidelity == 0


def test_query_mixed_fidelities(budget):
    """Test query with mixed fidelity candidates."""

    def score_fn_0(x):
        return float(x)

    def score_fn_1(x):
        return float(x) + 0.5

    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": score_fn_0,
                "fidelity_confidence": 1.0,
            },
            1: {
                "cost_per_sample": 2.0,
                "score_fn": score_fn_1,
                "fidelity_confidence": 1.0,
            },
        }
    )
    candidates = [
        Candidate(x=10, fidelity=0),
        Candidate(x=20, fidelity=1),
        Candidate(x=30, fidelity=0),
    ]

    observations = oracle.query(candidates)

    assert len(observations) == 3
    assert observations[0].x == 10
    assert observations[0].y == 10.0
    assert observations[0].fidelity == 0

    assert observations[1].x == 20
    assert observations[1].y == 20.5
    assert observations[1].fidelity == 1

    assert observations[2].x == 30
    assert observations[2].y == 30.0
    assert observations[2].fidelity == 0


def test_query_consumes_budget():
    """Test that explicit budget consumption works correctly."""

    def schedule(r):
        return 50.0

    def score_fn(x):
        return float(x)

    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 2.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            },
            1: {
                "cost_per_sample": 3.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            },
        }
    )
    budget = Budget(available_budget=100.0, schedule=schedule)
    candidates = [
        Candidate(x=1, fidelity=0),  # cost 2.0
        Candidate(x=2, fidelity=1),  # cost 3.0
        Candidate(x=3, fidelity=0),  # cost 2.0
    ]  # Total: 7.0

    costs = oracle.get_costs(candidates)
    budget.consume(sum(costs))
    oracle.query(candidates)

    assert budget.available_budget == 93.0


def test_query_raises_when_budget_insufficient():
    """Test that budget consumption raises ValueError when budget is insufficient."""

    def schedule(r):
        return 10.0

    def score_fn(x):
        return float(x)

    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 5.0,
                "score_fn": score_fn,
                "fidelity_confidence": 1.0,
            }
        }
    )
    budget = Budget(available_budget=8.0, schedule=schedule)
    candidates = [Candidate(x=i, fidelity=0) for i in range(3)]  # Total cost: 15.0

    costs = oracle.get_costs(candidates)
    with pytest.raises(ValueError, match="Cost .* exceeds available budget"):
        budget.consume(sum(costs))
