import random
from typing import Any, Iterable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class PoolUniformSampler(Sampler):
    """Samples uniformly at random from a fixed candidate pool.

    Parameters
    ----------
        candidate_pool : Sequence[Candidate]
            Fixed pool of candidates to sample from.
        num_samples : int
            Number of candidates to sample per call.
    """

    def __init__(self, candidate_pool: Sequence[Candidate], num_samples: int) -> None:
        self.candidate_pool = candidate_pool
        self.num_samples = num_samples

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
        **kwargs: Any,
    ) -> list[Candidate]:
        """Samples uniformly from the candidate pool.

        Parameters
        ----------
            acquisition : Optional[Acquisition]
                Optional acquisition function (not used by this sampler).
            observations : Optional[Iterable[Observation]]
                Optional iterable of observations (not used by this sampler).
            **kwargs
                Additional arguments (unused).

        Returns
        -------
            result : list[Candidate]
            List of randomly sampled candidates.
        """
        if self.num_samples >= len(self.candidate_pool):
            return list(self.candidate_pool)
        return random.sample(list(self.candidate_pool), self.num_samples)
