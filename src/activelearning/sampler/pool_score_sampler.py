import random

from activelearning.sampler.sampler import Sampler


class PoolScoreSampler(Sampler):
    def __init__(self, candidate_pool, num_samples):
        """Samples from a candidate pool according to scores provided by a scoring function."""
        self.candidate_pool = candidate_pool
        self.num_samples = num_samples

    def sample(self, acquisition=None, observations=None, **kwargs):
        """Samples from the candidate pool based on scores.

        Args:
            acquisition: The acquisition function used to score candidates.
            observations: Optional list of observations (not used by this sampler).
            **kwargs: Ignored additional arguments.
        """
        if self.num_samples >= len(self.candidate_pool):
            return list(self.candidate_pool)

        if acquisition is None:
            raise ValueError("Acquisition function is required for PoolScoreSampler.")

        scores = acquisition(self.candidate_pool)
        return random.choices(
            population=list(self.candidate_pool), weights=scores, k=self.num_samples
        )
