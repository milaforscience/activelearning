import pytest
import torch

from activelearning.sampler.hypercube_uniform_sampler import HypercubeUniformSampler
from activelearning.utils.types import Candidate


BRANIN_BOUNDS = [(-5.0, 10.0), (0.0, 15.0)]
HARTMANN_BOUNDS = [(0.0, 1.0)] * 6


def test_sample_returns_correct_count():
    """sample() returns exactly num_samples candidates."""
    sampler = HypercubeUniformSampler(bounds=BRANIN_BOUNDS, num_samples=20)
    result = sampler.sample()
    assert len(result) == 20


def test_sample_returns_candidate_objects():
    """Each item returned by sample() is a Candidate instance."""
    sampler = HypercubeUniformSampler(bounds=BRANIN_BOUNDS, num_samples=5)
    result = sampler.sample()
    assert all(isinstance(c, Candidate) for c in result)


def test_x_has_correct_dimensionality():
    """Candidate.x length equals the number of bound dimensions."""
    sampler = HypercubeUniformSampler(bounds=HARTMANN_BOUNDS, num_samples=10)
    for candidate in sampler.sample():
        assert len(candidate.x) == len(HARTMANN_BOUNDS)


def test_samples_within_bounds():
    """All sampled x values lie within their respective dimension bounds."""
    sampler = HypercubeUniformSampler(bounds=BRANIN_BOUNDS, num_samples=200)
    for candidate in sampler.sample():
        for value, (lower, upper) in zip(candidate.x, BRANIN_BOUNDS):
            assert lower <= value <= upper, (
                f"Value {value} outside bound [{lower}, {upper}]"
            )


def test_samples_within_bounds_hartmann():
    """All sampled values for 6D bounds lie in [0, 1]."""
    sampler = HypercubeUniformSampler(bounds=HARTMANN_BOUNDS, num_samples=200)
    for candidate in sampler.sample():
        for value in candidate.x:
            assert 0.0 <= value <= 1.0


def test_fidelity_is_none_by_default():
    """Fidelity defaults to None when fidelities not specified."""
    sampler = HypercubeUniformSampler(bounds=BRANIN_BOUNDS, num_samples=5)
    for candidate in sampler.sample():
        assert candidate.fidelity is None


def test_fidelity_sampled_from_list():
    """All candidate fidelities are drawn from the supplied fidelities list."""
    fidelities = [1, 2, 3]
    sampler = HypercubeUniformSampler(
        bounds=BRANIN_BOUNDS, num_samples=50, fidelities=fidelities
    )
    for candidate in sampler.sample():
        assert candidate.fidelity in fidelities


def test_all_fidelities_represented_in_large_sample():
    """With enough samples, all fidelity levels are drawn at least once."""
    fidelities = [1, 2, 3]
    sampler = HypercubeUniformSampler(
        bounds=BRANIN_BOUNDS, num_samples=300, fidelities=fidelities
    )
    observed = {c.fidelity for c in sampler.sample()}
    assert observed == set(fidelities)


def test_single_fidelity_list_always_produces_that_fidelity():
    """A single-element fidelities list always produces that fidelity."""
    sampler = HypercubeUniformSampler(
        bounds=BRANIN_BOUNDS, num_samples=10, fidelities=[3]
    )
    for candidate in sampler.sample():
        assert candidate.fidelity == 3


def test_reproducibility_with_seed():
    """Fixing torch seed before sample() produces identical results."""
    sampler = HypercubeUniformSampler(
        bounds=BRANIN_BOUNDS, num_samples=8, fidelities=[1, 2, 3]
    )

    torch.manual_seed(42)
    first = sampler.sample()

    torch.manual_seed(42)
    second = sampler.sample()

    assert [c.x for c in first] == [c.x for c in second]
    assert [c.fidelity for c in first] == [c.fidelity for c in second]


def test_independent_calls_differ():
    """Consecutive calls (without re-seeding) produce different samples."""
    sampler = HypercubeUniformSampler(bounds=BRANIN_BOUNDS, num_samples=10)
    first = sampler.sample()
    second = sampler.sample()
    # With overwhelming probability these differ; if they match, the RNG is broken.
    assert [c.x for c in first] != [c.x for c in second]


def test_sample_ignores_acquisition_and_observations():
    """sample() accepts acquisition and observations kwargs without error."""
    sampler = HypercubeUniformSampler(bounds=BRANIN_BOUNDS, num_samples=5)
    result = sampler.sample(acquisition=None, observations=iter([]))
    assert len(result) == 5


def test_single_dimension_bounds():
    """Works correctly for 1D input domains."""
    sampler = HypercubeUniformSampler(bounds=[(2.0, 5.0)], num_samples=50)
    for candidate in sampler.sample():
        assert len(candidate.x) == 1
        assert 2.0 <= candidate.x[0] <= 5.0


def test_raises_on_invalid_bounds():
    """Raises ValueError when lower >= upper."""
    with pytest.raises(ValueError, match="lower >= upper"):
        HypercubeUniformSampler(bounds=[(5.0, 2.0)], num_samples=10)


def test_raises_on_equal_bounds():
    """Raises ValueError when lower == upper."""
    with pytest.raises(ValueError, match="lower >= upper"):
        HypercubeUniformSampler(bounds=[(3.0, 3.0)], num_samples=10)


def test_raises_on_non_positive_num_samples():
    """Raises ValueError when num_samples <= 0."""
    with pytest.raises(ValueError, match="num_samples must be > 0"):
        HypercubeUniformSampler(bounds=BRANIN_BOUNDS, num_samples=0)


def test_empty_bounds_produces_zero_dimensional_candidates():
    """Empty bounds are allowed and produce 0-dimensional candidate vectors."""
    sampler = HypercubeUniformSampler(bounds=[], num_samples=5)
    result = sampler.sample()
    assert len(result) == 5
    # Each candidate's x should be 0-dimensional when bounds is empty.
    assert all(len(candidate.x) == 0 for candidate in result)
