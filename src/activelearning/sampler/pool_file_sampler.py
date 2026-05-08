import random
from pathlib import Path
from typing import Iterable, Optional, Union

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.fidelity_policy import DiscreteFidelityPolicy
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class PoolFileSampler(Sampler):
    """Samples candidates uniformly at random from a plain-text pool file.

    Each non-empty line of ``candidate_pool_file`` is treated as one candidate
    entry.  Up to ``num_samples`` candidates are drawn without replacement on
    each ``sample()`` call; if the file has fewer lines than ``num_samples``
    all lines are returned.

    Fidelity assignment mirrors :class:`~activelearning.sampler.hypercube_sampler.HypercubeSampler`:

    * ``None`` — every candidate gets ``fidelity=None``.
    * ``[1, 2, 3]`` — each candidate is assigned a uniformly random fidelity
      from the list.
    * ``{1: 1.0, 2: 5.0}`` — fidelities are sampled with probability inversely
      proportional to their cost (cheaper fidelities appear more often).

    Parameters
    ----------
    candidate_pool_file : Path or str
        Path to a text file containing one candidate per line.
    num_samples : int
        Maximum number of candidates to return per ``sample()`` call.
    fidelity_policy : DiscreteFidelityPolicy or None
        Policy controlling fidelity assignment for sampled candidates.
    """

    def __init__(
        self,
        candidate_pool_file: Union[Path, str],
        num_samples: int,
        fidelity_policy: DiscreteFidelityPolicy | None = None,
    ) -> None:
        self._pool_file = Path(candidate_pool_file)
        self.num_samples = num_samples
        self.fidelity_policy = fidelity_policy

    def _load_pool(self) -> list[str]:
        """Read and return all non-empty lines from the pool file."""
        if not self._pool_file.exists():
            raise FileNotFoundError(f"Candidate pool file not found: {self._pool_file}")
        lines = [line.strip() for line in self._pool_file.read_text().splitlines()]
        pool = [line for line in lines if line]
        if not pool:
            raise ValueError(f"Candidate pool file is empty: {self._pool_file}")
        return pool

    def _assign_fidelities(
        self,
        n: int,
        rng: random.Random,
    ) -> list[Optional[int]]:
        """Sample ``n`` fidelity assignments using the configured strategy."""
        if self.fidelity_policy is None:
            return [None] * n
        return self.fidelity_policy.sample_fidelities(n, rng)

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
    ) -> list[Candidate]:
        """Sample candidates from the pool file.

        Parameters
        ----------
        acquisition : Optional[Acquisition]
            Unused; present for interface compatibility.
        observations : Optional[Iterable[Observation]]
            Unused; present for interface compatibility.

        Returns
        -------
        result : list[Candidate]
            Up to ``num_samples`` candidates with ``x`` set to the raw line
            value and ``fidelity`` drawn from the configured strategy.
        """
        pool = self._load_pool()
        k = min(self.num_samples, len(pool))
        rng = random.Random(self.runtime_context.seed + self.active_learning_round)
        chosen = rng.sample(pool, k)
        fidelities = self._assign_fidelities(k, rng)
        return [Candidate(x=x, fidelity=fid) for x, fid in zip(chosen, fidelities)]
