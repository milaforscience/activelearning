import random

from activelearning.sampler.sampler import Sampler


class PoolUniformSampler(Sampler):
    def __init__(self, candidate_pool, num_samples):
        """Samples uniformly at random from a candidate pool."""
        self.candidate_pool = candidate_pool
        self.num_samples = num_samples

    def sample(self, acquisition=None, observations=None, **kwargs):
        """Samples uniformly from the candidate pool.

        Args:
            acquisition: Optional acquisition function (not used by this sampler).
            observations: Optional list of observations (not used by this sampler).
            **kwargs: Ignored additional arguments.
        """
        if self.num_samples >= len(self.candidate_pool):
            return list(self.candidate_pool)
        return random.sample(list(self.candidate_pool), self.num_samples)
