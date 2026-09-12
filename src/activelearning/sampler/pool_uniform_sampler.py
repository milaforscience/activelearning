import random
from typing import Callable, Iterable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation
from activelearning.utils.warnings import warn_ignored_args


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
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> list[Candidate]:
        """Samples uniformly from the candidate pool.

        Parameters
        ----------
        acquisition : Optional[Acquisition]
            Unused. Present for interface compatibility. Passing a non-``None``
            value raises a :class:`UserWarning`.
        observations : Optional[Iterable[Observation]]
            Unused. Present for interface compatibility. Passing a non-``None``
            value raises a :class:`UserWarning`.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Unused. Present for interface compatibility. Passing a non-``None``
            value raises a :class:`UserWarning`.

        Returns
        -------
        result : list[Candidate]
            List of randomly sampled candidates.
        """
        warn_ignored_args(
            self, acquisition=acquisition, observations=observations, cost_fn=cost_fn
        )
        if self.num_samples >= len(self.candidate_pool):
            return list(self.candidate_pool)
        return random.sample(list(self.candidate_pool), self.num_samples)
