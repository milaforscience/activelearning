import torch

from typing import Iterable, Literal, Optional, Sequence, Union

from activelearning.acquisition.acquisition import Acquisition
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
    fidelities : Sequence[int] or dict[int, float] or None
        Controls fidelity assignment for each candidate:
        - ``None`` — no fidelity (``candidate.fidelity = None``).
        - ``[1, 2, 3]`` — uniform sampling across fidelity levels.
        - ``{1: 1.0, 2: 5.0}`` — cost-inverse sampling: each key is a fidelity
          level and each value is its cost. Candidates are assigned fidelities
          with probability **inversely proportional** to cost (cheaper fidelities
          are sampled more often). All costs must be positive.
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
        fidelities: Union[None, Sequence[int], dict[int, float]] = None,
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

        # Normalize the fidelities param into internal fields
        self._fidelity_levels = None
        self._fidelity_costs = None
        if isinstance(fidelities, dict):
            if len(fidelities) == 0:
                raise ValueError("fidelities must not be empty when specified")
            for fidelity, cost in fidelities.items():
                if cost <= 0:
                    raise ValueError(
                        f"All costs must be positive; fidelity {fidelity} has cost {cost}"
                    )
            self._fidelity_levels = sorted(fidelities.keys())
            self._fidelity_costs = fidelities
        elif isinstance(fidelities, list):
            if len(fidelities) == 0:
                raise ValueError("fidelities must not be empty when specified")
            self._fidelity_levels = fidelities

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
            return latin_hypercube(self.num_samples, n_dims)
        # Default: uniform
        return torch.rand(self.num_samples, n_dims, dtype=self.dtype)


    def _assign_fidelities(self) -> list[Optional[int]]:
        """Assign fidelity levels to ``num_samples`` candidates.

        Returns
        -------
        fidelities : list[Optional[int]]
            One fidelity per candidate. ``None`` if no fidelities were configured.
        """
        if self._fidelity_levels is None:
            return [None] * self.num_samples

        fidelity_tensor = torch.tensor(
            self._fidelity_levels,
            dtype=torch.long,
        )

        if self._fidelity_costs is not None:
            # Weights ∝ 1/cost; cheaper fidelities are sampled more often
            costs = torch.tensor(
                [self._fidelity_costs[f] for f in self._fidelity_levels],
                dtype=self.dtype,
            )
            weights = costs.reciprocal()
            indices = torch.multinomial(
                weights,
                num_samples=self.num_samples,
                replacement=True,
            )
        else:
            # Uniform sampling across fidelity levels
            indices = torch.randint(
                0,
                len(self._fidelity_levels),
                (self.num_samples,),
            )

        selected = fidelity_tensor[indices]
        return selected.tolist()

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
