import random
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import torch

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, DEFAULT_FIDELITY, Observation
from activelearning.utils.warnings import warn_ignored_args


class PoolFileSampler(Sampler):
    """Samples candidates uniformly at random from a plain-text pool file.

    Each non-empty line of ``candidate_pool_file`` is treated as one candidate
    entry.  Up to ``num_samples`` candidates are drawn without replacement on
    each ``sample()`` call; if the file has fewer lines than ``num_samples``
    all lines are returned.

    Fidelity assignment mirrors :class:`~activelearning.sampler.hypercube_sampler.HypercubeSampler`:

    * ``[DEFAULT_FIDELITY]`` — the default single-fidelity configuration.
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
    fidelities : Sequence[int] or dict[int, float]
        Fidelity assignment strategy (see above).
    """

    def __init__(
        self,
        candidate_pool_file: Path | str,
        num_samples: int,
        fidelities: Sequence[int] | dict[int, float] = (DEFAULT_FIDELITY,),
    ) -> None:
        self._pool_file = Path(candidate_pool_file)
        self.num_samples = num_samples

        self._fidelity_levels: list[int]
        self._fidelity_costs: Optional[dict[int, float]] = None

        if isinstance(fidelities, dict):
            self._fidelity_levels = sorted(fidelities)
            self._fidelity_costs = fidelities
        else:
            self._fidelity_levels = list(fidelities)

    def _load_pool(self) -> list[str]:
        """Read and return all non-empty lines from the pool file.

        Returns
        -------
        list[str]
            Non-empty, stripped lines from the pool file.

        Raises
        ------
        FileNotFoundError
            If the pool file does not exist.
        ValueError
            If the pool file contains no non-empty lines.
        """
        if not self._pool_file.exists():
            raise FileNotFoundError(f"Candidate pool file not found: {self._pool_file}")
        lines = [line.strip() for line in self._pool_file.read_text().splitlines()]
        pool = [line for line in lines if line]
        if not pool:
            raise ValueError(f"Candidate pool file is empty: {self._pool_file}")
        return pool

    def _assign_fidelities(self, n: int) -> list[int]:
        """Sample ``n`` fidelity assignments using the configured strategy.

        Parameters
        ----------
        n : int
            Number of fidelity values to draw.

        Returns
        -------
        list[int]
            List of ``n`` integer fidelity levels drawn according to the
            configured strategy (uniform or cost-inverse weighted).
        """
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
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> list[Candidate]:
        """Sample candidates from the pool file.

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
            Up to ``num_samples`` candidates with ``x`` set to the raw line
            value and ``fidelity`` drawn from the configured strategy.
        """
        warn_ignored_args(
            self, acquisition=acquisition, observations=observations, cost_fn=cost_fn
        )
        pool = self._load_pool()
        k = min(self.num_samples, len(pool))
        chosen = random.sample(pool, k)
        fidelities = self._assign_fidelities(k)
        return [Candidate(x=x, fidelity=fid) for x, fid in zip(chosen, fidelities)]
