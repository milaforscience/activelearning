import random
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import torch

from activelearning.acquisition.acquisition import Acquisition
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
    fidelities : list[int] or dict[int, float] or None
        Fidelity assignment strategy (see above).
    """

    def __init__(
        self,
        candidate_pool_file: Union[Path, str],
        num_samples: int,
        fidelities: Union[None, Sequence[int], dict[int, float]] = None,
    ) -> None:
        self._pool_file = Path(candidate_pool_file)
        self.num_samples = num_samples

        self._fidelity_levels: Optional[list[int]] = None
        self._fidelity_costs: Optional[dict[int, float]] = None

        if isinstance(fidelities, dict):
            if not fidelities:
                raise ValueError("fidelities dict must not be empty")
            for fid, cost in fidelities.items():
                if cost <= 0:
                    raise ValueError(
                        f"All fidelity costs must be positive; fidelity {fid} has cost {cost}"
                    )
            self._fidelity_levels = sorted(fidelities.keys())
            self._fidelity_costs = fidelities
        elif fidelities is not None:
            fidelities = list(fidelities)
            if not fidelities:
                raise ValueError("fidelities list must not be empty")
            self._fidelity_levels = fidelities

    def _load_pool(self) -> list[str]:
        """Read and return all non-empty lines from the pool file."""
        if not self._pool_file.exists():
            raise FileNotFoundError(f"Candidate pool file not found: {self._pool_file}")
        lines = [line.strip() for line in self._pool_file.read_text().splitlines()]
        pool = [line for line in lines if line]
        if not pool:
            raise ValueError(f"Candidate pool file is empty: {self._pool_file}")
        return pool

    def _assign_fidelities(self, n: int) -> list[Optional[int]]:
        """Sample ``n`` fidelity assignments using the configured strategy."""
        if self._fidelity_levels is None:
            return [None] * n

        fidelity_tensor = torch.tensor(self._fidelity_levels, dtype=torch.long)

        if self._fidelity_costs is not None:
            costs = torch.tensor(
                [self._fidelity_costs[f] for f in self._fidelity_levels],
                dtype=torch.float64,
            )
            weights = costs.reciprocal()
            indices = torch.multinomial(weights, num_samples=n, replacement=True)
        else:
            indices = torch.randint(0, len(self._fidelity_levels), (n,))

        return fidelity_tensor[indices].tolist()

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
        chosen = random.sample(pool, k)
        fidelities = self._assign_fidelities(k)
        return [Candidate(x=x, fidelity=fid) for x, fid in zip(chosen, fidelities)]
