import random
from typing import Any, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class PoolUniformSampler(Sampler):
    """Samples uniformly at random from a fixed candidate pool.

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
        """Samples uniformly from the candidate pool.

        Args:
            acquisition: Optional acquisition function (not used by this sampler).
            observations: Optional list of observations (not used by this sampler).
            **kwargs: Additional arguments (unused).

        Returns:
            List of randomly sampled candidates.
        """
        if self.num_samples >= len(self.candidate_pool):
            return list(self.candidate_pool)
        return random.sample(list(self.candidate_pool), self.num_samples)
