import pytest

from activelearning.oracle.augmented_function_oracle import (
    BraninOracle,
    Hartmann6DOracle,
)
from activelearning.utils.types import Candidate


class TestBraninOracleQuery:
    """Test BraninOracle.query() returns valid observations."""

    def test_query_returns_observation_for_each_candidate(self):
        oracle = BraninOracle()
        candidates = [Candidate(x=[0.5, 7.5], fidelity=3)]
        observations = oracle.query(candidates)
        assert len(observations) == 1

    def test_query_observation_has_matching_x_and_fidelity(self):
        oracle = BraninOracle()
        candidates = [Candidate(x=[1.0, 3.0], fidelity=2)]
        obs = oracle.query(candidates)[0]
        assert obs.x == [1.0, 3.0]
        assert obs.fidelity == 2

    def test_query_observation_y_is_float(self):
        oracle = BraninOracle()
        candidates = [Candidate(x=[0.0, 0.0], fidelity=1)]
        obs = oracle.query(candidates)[0]
        assert isinstance(obs.y, float)

    def test_query_raises_on_none_fidelity(self):
        oracle = BraninOracle()
        with pytest.raises(ValueError, match="fidelity"):
            oracle.query([Candidate(x=[0.5, 7.5])])

    def test_fidelity_confidences_proportional_to_cost(self):
        oracle = BraninOracle()
        confidences = oracle.get_fidelity_confidences()
        # Highest-cost fidelity (3) must have confidence 1.0
        assert confidences[3] == pytest.approx(1.0)
        assert confidences[2] < confidences[3]
        assert confidences[1] < confidences[2]


class TestHartmann6DOracleQuery:
    """Test Hartmann6DOracle.query() returns valid observations."""

    def test_query_returns_observation_for_each_candidate(self):
        oracle = Hartmann6DOracle()
        candidates = [Candidate(x=[0.2] * 6, fidelity=3)]
        observations = oracle.query(candidates)
        assert len(observations) == 1

    def test_query_observation_y_is_float(self):
        oracle = Hartmann6DOracle()
        candidates = [Candidate(x=[0.5] * 6, fidelity=2)]
        obs = oracle.query(candidates)[0]
        assert isinstance(obs.y, float)

    def test_query_raises_on_none_fidelity(self):
        oracle = Hartmann6DOracle()
        with pytest.raises(ValueError, match="fidelity"):
            oracle.query([Candidate(x=[0.5] * 6)])
