import random
import torch
from typing import Any, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class PoolScoreSampler(Sampler):
    """Samples from a candidate pool weighted by acquisition scores using softmax.

    Converts acquisition scores to probabilities via softmax transformation,
    which handles negative scores and ensures numerical stability.

    Args:
        candidate_pool: Fixed pool of candidates to sample from.
        num_samples: Number of candidates to sample per call.
    """

    def __init__(self, candidate_pool: Sequence[Candidate], num_samples: int) -> None:
        self.candidate_pool = candidate_pool
        self.num_samples = num_samples

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Sequence[Observation]] = None,
        **kwargs: Any,
    ) -> list[Candidate]:
        """Samples from the candidate pool weighted by acquisition scores.

        Args:
            acquisition: Acquisition function to score candidates.
            observations: Optional list of observations (not used by this sampler).
            **kwargs: Additional arguments (unused).

        Returns:
            List of sampled candidates weighted by softmax probabilities.
        """
        if self.num_samples >= len(self.candidate_pool):
            return list(self.candidate_pool)

        if acquisition is None:
            raise ValueError("Acquisition function is required for PoolScoreSampler.")

        scores = acquisition(self.candidate_pool)

        # Apply softmax to convert scores to valid probabilities
        weights = torch.softmax(torch.tensor(scores), dim=0).tolist()

        return random.choices(
            population=list(self.candidate_pool), weights=weights, k=self.num_samples
        )
