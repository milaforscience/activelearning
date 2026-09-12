import torch

from typing import Callable, Iterable, Literal, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.sampling import latin_hypercube
from activelearning.utils.types import Candidate, DEFAULT_FIDELITY, Observation
from activelearning.utils.warnings import warn_ignored_args


class HypercubeSampler(Sampler):
    """Generates candidates by sampling from a bounded hypercube.

    This sampler is purely generative: every :meth:`sample` call draws a new
    batch of points inside ``bounds``. It supports both point-generation
    strategy selection and optional fidelity assignment.

    - Point generation: ``"uniform"`` for i.i.d. random draws or ``"lhs"``
      for Latin hypercube sampling.
    - Fidelity assignment: controlled by the type of ``fidelities``.

    Parameters
    ----------
    bounds : Sequence[tuple[float, float]]
        Per-dimension ``(lower, upper)`` bounds. Each pair must satisfy
        ``lower < upper``. The length determines input dimensionality.
    num_samples : int
        Number of candidates to generate per ``sample()`` call. Must be > 0.
    fidelities : Sequence[int] or dict[int, float]
        Controls fidelity assignment for each candidate:

        - ``[DEFAULT_FIDELITY]`` — the default single-fidelity configuration.
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
        If bounds are empty or invalid, ``num_samples <= 0``, or
        ``point_strategy`` is unrecognised.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        num_samples: int,
        fidelities: Sequence[int] | dict[int, float] = (DEFAULT_FIDELITY,),
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

        if isinstance(fidelities, dict):
            self._fidelity_levels = sorted(fidelities)
            self._fidelity_costs = fidelities
        else:
            self._fidelity_levels = list(fidelities)
            self._fidelity_costs = None

        # Store scalar values and materialize tensors lazily so a later-bound
        # runtime context can still control dtype.
        lowers, diffs = zip(*[(lower, upper - lower) for lower, upper in bounds])
        self._lower_values = tuple(lowers)
        self._range_values = tuple(diffs)

    def _get_bounds_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize lower/range tensors using the currently bound runtime dtype.

        Returns
        -------
        lower : torch.Tensor
            1-D tensor of per-dimension lower bounds.
        ranges : torch.Tensor
            1-D tensor of per-dimension widths (``upper - lower``).
        """
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

    def _assign_fidelities(self) -> list[int]:
        """Assign fidelity levels to ``num_samples`` candidates.

        Returns
        -------
        fidelities : list[int]
            One fidelity per candidate.
        """
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
                weights, num_samples=self.num_samples, replacement=True
            )
        else:
            # Uniform sampling across fidelity levels
            indices = torch.randint(0, len(self._fidelity_levels), (self.num_samples,))

        selected = fidelity_tensor[indices]
        return selected.tolist()

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> list[Candidate]:
        """Generate candidates from the hypercube.

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
            ``num_samples`` candidates with ``x`` as a plain Python list of
            floats and ``fidelity`` drawn from the configured fidelity strategy.
        """
        warn_ignored_args(
            self, acquisition=acquisition, observations=observations, cost_fn=cost_fn
        )
        # Generate points in [0,1]^d then scale to bounds
        lower, ranges = self._get_bounds_tensors()
        unit_points = self._generate_points()
        points = lower + unit_points * ranges

        fidelities = self._assign_fidelities()

        return [
            Candidate(x=points[i].tolist(), fidelity=fidelities[i])
            for i in range(self.num_samples)
        ]
