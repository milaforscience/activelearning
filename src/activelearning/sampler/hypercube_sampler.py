import random
import torch

from typing import Iterable, Literal, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.fidelity_policy import (
    DiscreteFidelityPolicy,
)
from activelearning.sampler.sampler import Sampler
from activelearning.utils.sampling import latin_hypercube
from activelearning.utils.types import Candidate, Observation


class HypercubeSampler(Sampler):
    """Generates candidates by sampling from a bounded hypercube.

    Supports sampling strategies for both point generation and fidelity assignment:
    - Point generation: "uniform" (i.i.d. random) or "lhs" (Latin Hypercube Sampling,
      one point per stratum per dimension, improving space-filling).
    - Fidelity assignment:  controlled by the ``fidelities`` parameter type.

    Parameters
    ----------
    bounds : Sequence[tuple[float, float]]
        Per-dimension ``(lower, upper)`` bounds. Each pair must satisfy
        ``lower < upper``. The length determines input dimensionality.
    num_samples : int
        Number of candidates to generate per ``sample()`` call. Must be > 0.
    fidelity_policy : DiscreteFidelityPolicy or None
        Policy controlling how fidelities are assigned to generated candidates.
        ``None`` leaves candidates without fidelity metadata.
    point_strategy : Literal["uniform", "lhs"]
        How x-values are generated within the hypercube.
        "uniform" draws i.i.d. uniform samples; "lhs" uses Latin
        Hypercube Sampling for better space coverage. Defaults to "uniform".

    Raises
    ------
    ValueError
        If bounds are empty or invalid, ``num_samples <= 0``, ``fidelities`` is
        an empty sequence or dict, any cost is non-positive, or
        ``point_strategy`` is unrecognised.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        num_samples: int,
        fidelity_policy: DiscreteFidelityPolicy | None = None,
        point_strategy: Literal["uniform", "lhs"] = "uniform",
    ) -> None:
        if len(bounds) == 0:
            raise ValueError("bounds must not be empty")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        for idx, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValueError(f"Bound {idx} has lower >= upper: ({lower}, {upper})")
        if point_strategy not in ("uniform", "lhs"):
            raise ValueError(
                f"point_strategy must be 'uniform' or 'lhs', got {point_strategy!r}"
            )

        self.bounds = bounds
        self.num_samples = num_samples
        self.point_strategy = point_strategy
        self.fidelity_policy = fidelity_policy

        # Store scalar values and materialize tensors lazily so a later-bound
        # runtime context can still control dtype.
        lowers, diffs = zip(*[(lower, upper - lower) for lower, upper in bounds])
        self._lower_values = tuple(lowers)
        self._range_values = tuple(diffs)

    def _get_bounds_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize lower/range tensors using the currently bound runtime dtype."""
        return (
            torch.tensor(self._lower_values, dtype=self.dtype),
            torch.tensor(self._range_values, dtype=self.dtype),
        )

    def _generate_points(self) -> torch.Tensor:
        """Generate raw points in the unit hypercube ``[0, 1]^d``.

        Returns
        -------
        points : torch.Tensor
            Shape ``(num_samples, n_dims)`` with values in ``[0, 1]``.
        """
        n_dims = len(self.bounds)
        if self.point_strategy == "lhs":
            return latin_hypercube(self.num_samples, n_dims, dtype=self.dtype)
        # Default: uniform
        return torch.rand(self.num_samples, n_dims, dtype=self.dtype)

    def _assign_fidelities(self) -> list[Optional[int]]:
        """Assign fidelity levels to ``num_samples`` candidates.

        Returns
        -------
        fidelities : list[Optional[int]]
            One fidelity per candidate. ``None`` if no fidelities were configured.
        """
        if self.fidelity_policy is None:
            return [None] * self.num_samples
        rng = random.Random(self.runtime_context.seed + self.active_learning_round)
        return self.fidelity_policy.sample_fidelities(self.num_samples, rng)

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
    ) -> list[Candidate]:
        """Generate candidates from the hypercube.

        Parameters
        ----------
        acquisition : Optional[Acquisition]
            Unused. Present for interface compatibility.
        observations : Optional[Iterable[Observation]]
            Unused. Present for interface compatibility.

        Returns
        -------
        result : list[Candidate]
            ``num_samples`` candidates with ``x`` as a plain Python list of
            floats and ``fidelity`` drawn from the configured fidelity strategy.
        """
        # Generate points in [0,1]^d then scale to bounds
        lower, ranges = self._get_bounds_tensors()
        unit_points = self._generate_points()
        points = lower + unit_points * ranges

        fidelities = self._assign_fidelities()

        return [
            Candidate(x=points[i].tolist(), fidelity=fidelities[i])
            for i in range(self.num_samples)
        ]
