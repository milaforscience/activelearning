import pytest

from activelearning.runtime import RuntimeContext
from activelearning.sampler.fidelity_policy import UniformFidelityPolicy
from activelearning.sampler.random_token_sequence_sampler import (
    RandomTokenSequenceSampler,
)


def test_random_token_sequence_sampler_generates_bounded_unique_sequences():
    """RandomTokenSequenceSampler should generate unique bounded sequences."""

    sampler = RandomTokenSequenceSampler(
        tokens=["A", "B"],
        num_samples=10,
        min_length=1,
        max_length=3,
    )
    sampler.bind_runtime_context(RuntimeContext(seed=7))

    candidates = sampler.sample()

    assert len(candidates) == 10
    assert len({candidate.x for candidate in candidates}) == 10
    assert all(1 <= len(candidate.x) <= 3 for candidate in candidates)
    assert all(set(candidate.x) <= {"A", "B"} for candidate in candidates)


def test_random_token_sequence_sampler_is_reproducible_by_round_and_seed():
    """RandomTokenSequenceSampler should use runtime seed and round index."""

    sampler = RandomTokenSequenceSampler(tokens=["A", "B"], num_samples=5, max_length=2)
    runtime_context = RuntimeContext(seed=7, active_learning_round=0)
    sampler.bind_runtime_context(runtime_context)

    first_round = sampler.sample()
    runtime_context.active_learning_round = 1
    second_round = sampler.sample()
    runtime_context.active_learning_round = 0
    repeated_first_round = sampler.sample()

    assert first_round == repeated_first_round
    assert first_round != second_round


def test_random_token_sequence_sampler_applies_fidelity_policy_after_sampling():
    """RandomTokenSequenceSampler should stamp fidelities via the shared policy."""

    sampler = RandomTokenSequenceSampler(
        tokens=["A", "B"],
        num_samples=5,
        max_length=2,
        fidelity_policy=UniformFidelityPolicy(values=(1, 2)),
    )
    sampler.bind_runtime_context(RuntimeContext(seed=3))

    candidates = sampler.sample()

    assert len(candidates) == 5
    assert {candidate.fidelity for candidate in candidates} <= {1, 2}


def test_random_token_sequence_sampler_raises_when_unique_space_is_too_small():
    """RandomTokenSequenceSampler should fail early when uniqueness is impossible."""

    sampler = RandomTokenSequenceSampler(
        tokens=["A"],
        num_samples=2,
        min_length=1,
        max_length=1,
        max_attempts=5,
    )
    sampler.bind_runtime_context(RuntimeContext(seed=1))

    with pytest.raises(RuntimeError, match="Cannot sample 2 unique sequences"):
        sampler.sample()


def test_random_token_sequence_sampler_deduplicates_tokens_for_uniqueness_bounds():
    """Duplicate tokens should not inflate the unique sequence space."""

    sampler = RandomTokenSequenceSampler(
        tokens=["A", "A"],
        num_samples=2,
        min_length=1,
        max_length=1,
    )
    sampler.bind_runtime_context(RuntimeContext(seed=1))

    with pytest.raises(RuntimeError, match="Cannot sample 2 unique sequences"):
        sampler.sample()
