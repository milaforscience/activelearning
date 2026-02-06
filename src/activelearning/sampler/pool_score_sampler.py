import random

from activelearning.sampler.sampler import Sampler


class PoolScoreSampler(Sampler):
    def __init__(self, candidate_pool, num_samples, score_fn):
        """Samples from a candidate pool according to scores provided by a scoring function."""
        self.candidate_pool = candidate_pool
        self.num_samples = num_samples
        self.score_fn = score_fn

    def sample(self):
        """Samples from the candidate pool based on scores."""
        if self.num_samples >= len(self.candidate_pool):
            return list(self.candidate_pool)

        scores = self.score_fn(self.candidate_pool)
        return random.choices(
            population=list(self.candidate_pool), weights=scores, k=self.num_samples
        )
