import torch

from typing import Any, Iterable, Optional, Sequence
from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class PoolScoreSampler(Sampler):
    """Samples from a candidate pool weighted by acquisition scores using softmax.

    Converts acquisition scores to probabilities via softmax transformation,
    which handles negative scores and ensures numerical stability.

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
        """Samples from the candidate pool weighted by acquisition scores.

        Parameters
        ----------
            acquisition : Optional[Acquisition]
                Acquisition function to score candidates.
            observations : Optional[Iterable[Observation]]
                Optional iterable of observations (not used by this sampler).
            **kwargs
                Additional arguments (unused).

        Returns
        -------
            result : list[Candidate]
            List of sampled candidates weighted by softmax probabilities,
            without replacement.
        """
        if self.num_samples >= len(self.candidate_pool):
            return list(self.candidate_pool)

        if acquisition is None:
            raise ValueError("Acquisition function is required for PoolScoreSampler.")

        scores = acquisition(self.candidate_pool)

        # Apply softmax to convert scores to valid probabilities
        weights = torch.softmax(torch.tensor(scores), dim=0)
        sampled_indices = torch.multinomial(
            weights, num_samples=self.num_samples, replacement=False
        ).tolist()
        return [self.candidate_pool[i] for i in sampled_indices]
