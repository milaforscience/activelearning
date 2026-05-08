import random
from typing import Iterable, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.fidelity_policy import (
    DiscreteFidelityPolicy,
    apply_fidelity_policy,
)
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class RandomTokenSequenceSampler(Sampler):
    """Sample bounded token sequences from a random token-or-stop policy.

    The sampler draws a final length from the distribution induced by repeatedly
    choosing uniformly between all tokens and an implicit stop action, then
    samples that many tokens uniformly. Returned sequences therefore have length
    in ``[min_length, max_length]``.
    """

    def __init__(
        self,
        tokens: Sequence[str],
        num_samples: int,
        max_length: int,
        min_length: int = 1,
        fidelity_policy: DiscreteFidelityPolicy | None = None,
        seed_offset: int = 0,
        unique: bool = True,
        max_attempts: int = 100000,
    ) -> None:
        """Initialize the sampler.

        Parameters
        ----------
        tokens : Sequence[str]
            Vocabulary used to construct each sequence.
        num_samples : int
            Number of candidate sequences to return from each ``sample()`` call.
        max_length : int
            Maximum allowed sequence length.
        min_length : int, default=1
            Minimum allowed sequence length before stopping is permitted.
        fidelity_policy : DiscreteFidelityPolicy | None, default=None
            Optional policy used to assign fidelities after sequences are drawn.
        seed_offset : int, default=0
            Constant offset added to the runtime seed before sampling.
        unique : bool, default=True
            Whether returned sequences must be unique within one sampling call.
        max_attempts : int, default=100000
            Maximum number of draws used to satisfy uniqueness before failing.
        """
        if not tokens:
            raise ValueError("tokens must not be empty.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if min_length < 1:
            raise ValueError("min_length must be at least 1.")
        if max_length < min_length:
            raise ValueError("max_length must be greater than or equal to min_length.")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        self.tokens = tuple(dict.fromkeys(str(token) for token in tokens))
        self.num_samples = int(num_samples)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.fidelity_policy = fidelity_policy
        self.seed_offset = int(seed_offset)
        self.unique = bool(unique)
        self.max_attempts = int(max_attempts)
        self._lengths = tuple(range(self.min_length, self.max_length + 1))
        self._length_weights = self._build_length_weights()
        self._max_unique_sequences = sum(
            len(self.tokens) ** length for length in self._lengths
        )

    def _build_length_weights(self) -> tuple[float, ...]:
        """Return the length weights induced by uniform token-or-stop actions."""
        num_tokens = len(self.tokens)
        continue_probability = num_tokens / (num_tokens + 1)
        stop_probability = 1 / (num_tokens + 1)

        weights: list[float] = []
        continuation_mass = 1.0
        for _ in self._lengths[:-1]:
            weights.append(continuation_mass * stop_probability)
            continuation_mass *= continue_probability
        weights.append(continuation_mass)
        return tuple(weights)

    def _sample_sequence(self, rng: random.Random) -> str:
        """Generate one sequence from the rollout-induced length distribution."""
        length = rng.choices(self._lengths, weights=self._length_weights, k=1)[0]
        return "".join(rng.choices(self.tokens, k=length))

    def sample(
        self,
        acquisition: Acquisition | None = None,
        observations: Iterable[Observation] | None = None,
    ) -> list[Candidate]:
        """Return random candidate sequences, optionally with sampled fidelities."""
        rng = random.Random(
            self.runtime_context.seed + self.seed_offset + self.active_learning_round
        )

        if self.unique and self.num_samples > self._max_unique_sequences:
            raise RuntimeError(
                f"Cannot sample {self.num_samples} unique sequences from a space of "
                f"{self._max_unique_sequences} sequences."
            )

        if not self.unique:
            candidates = [
                Candidate(x=self._sample_sequence(rng)) for _ in range(self.num_samples)
            ]
            if self.fidelity_policy is None:
                return candidates
            return apply_fidelity_policy(candidates, self.fidelity_policy, rng)

        sequences: list[str] = []
        seen_sequences: set[str] = set()

        attempts = 0
        while len(sequences) < self.num_samples and attempts < self.max_attempts:
            sequence = self._sample_sequence(rng)
            attempts += 1
            if sequence in seen_sequences:
                continue
            sequences.append(sequence)
            seen_sequences.add(sequence)

        if len(sequences) != self.num_samples:
            raise RuntimeError(
                f"Reached max_attempts={self.max_attempts} before sampling "
                f"{self.num_samples} sequences."
            )

        candidates = [Candidate(x=sequence) for sequence in sequences]
        if self.fidelity_policy is None:
            return candidates
        return apply_fidelity_policy(candidates, self.fidelity_policy, rng)
