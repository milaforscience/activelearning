import random

from activelearning.sampler.sampler import Sampler


class PoolUniformSampler(Sampler):
    def __init__(self, candidate_pool, num_samples):
        """Samples uniformly at random from a candidate pool."""
        self.candidate_pool = candidate_pool
        self.num_samples = num_samples

    def sample(self):
        """Samples uniformly from the candidate pool."""
        if self.num_samples >= len(self.candidate_pool):
            return list(self.candidate_pool)
        return random.sample(list(self.candidate_pool), self.num_samples)
