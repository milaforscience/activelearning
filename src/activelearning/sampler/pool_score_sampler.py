import torch

from typing import Iterable, Optional, Sequence
from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class PoolScoreSampler(Sampler):
    """Samples candidates without replacement from a fixed pool using softmax-weighted values.

    Acquisition values are converted to probabilities with a numerically stable
    softmax transformation, which also supports negative values.

    Parameters
    ----------
    candidate_pool : Sequence[Candidate]
        Fixed pool of candidates to sample from.
    num_samples : int
        Number of candidates to sample per call, without replacement.
    """

    def __init__(self, candidate_pool: Sequence[Candidate], num_samples: int) -> None:
        self.candidate_pool = candidate_pool
        self.num_samples = num_samples

    def _get_sampling_weights(
        self, acquisition_values: Sequence[float]
    ) -> torch.Tensor:
        """Convert acquisition values into sampling probabilities."""
        acquisition_tensor = torch.as_tensor(
            acquisition_values,
            dtype=self.dtype,
            device=self.device,
        )
        return torch.softmax(acquisition_tensor, dim=0)

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
    ) -> list[Candidate]:
        """Samples from the candidate pool weighted by acquisition values.

        Parameters
        ----------
        acquisition : Optional[Acquisition]
            Acquisition function to compute acquisition values for candidates.
        observations : Optional[Iterable[Observation]]
            Optional iterable of observations (not used by this sampler).

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

        acq_values = acquisition.score(self.candidate_pool)

        # Apply softmax to convert acquisition values to valid probabilities
        weights = self._get_sampling_weights(acq_values)
        sampled_indices = (
            torch.multinomial(weights, num_samples=self.num_samples, replacement=False)
            .cpu()
            .tolist()
        )
        return [self.candidate_pool[i] for i in sampled_indices]
