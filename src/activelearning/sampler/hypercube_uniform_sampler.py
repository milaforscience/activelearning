import torch

from typing import Callable, Iterable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, DEFAULT_FIDELITY, Observation
from activelearning.utils.warnings import warn_ignored_args


class HypercubeUniformSampler(Sampler):
    """Generate candidates uniformly from a bounded hypercube.

    Unlike pool-based samplers, this sampler is generative: it draws fresh
    random candidates on every ``sample()`` call. This makes it suitable for
    continuous-domain functions such as ``BraninOracle`` and
    ``Hartmann6DOracle`` where no finite candidate pool exists.

    Fidelity levels are sampled uniformly at random from ``fidelities`` for
    each candidate, enabling multi-fidelity exploration without a fixed pool.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        num_samples: int,
        fidelities: Sequence[int] = (DEFAULT_FIDELITY,),
    ) -> None:
        """Initialize the hypercube uniform sampler.

        Parameters
        ----------
        bounds : Sequence[tuple[float, float]]
            Per-dimension ``(lower, upper)`` bounds that define the hypercube.
            Each pair must satisfy ``lower < upper``.
            The length determines the input dimensionality.
        num_samples : int
            Number of candidates to generate per ``sample()`` call. Must be > 0.
        fidelities : Sequence[int]
            Fidelity levels to sample from uniformly at random. Each generated
            candidate is assigned one level drawn with equal probability.
            Defaults to the single level
            :data:`~activelearning.utils.types.DEFAULT_FIDELITY`.

        Raises
        ------
        ValueError
            If ``bounds`` is empty, any lower bound >= upper bound, or
            ``num_samples`` <= 0.
        """
        if len(bounds) == 0:
            raise ValueError("bounds must not be empty")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        for i, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValueError(f"Bound {i} has lower >= upper: ({lower}, {upper})")

        self.bounds = bounds
        self.num_samples = num_samples
        self.fidelities = list(fidelities)

        # Store raw scalars; tensors are materialized in sample() using self.dtype
        # so that any RuntimeContext dtype binding is respected (same pattern as
        # HypercubeSampler).
        self._lower_values = tuple(b[0] for b in bounds)
        self._range_values = tuple(b[1] - b[0] for b in bounds)

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> list[Candidate]:
        """Generate candidates by sampling uniformly from the hypercube.

        Each candidate's fidelity is drawn uniformly at random from the
        ``fidelities`` list supplied at construction.

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
            floats and ``fidelity`` drawn uniformly from ``fidelities``.
        """
        warn_ignored_args(
            self, acquisition=acquisition, observations=observations, cost_fn=cost_fn
        )
        # Shape: (num_samples, n_dims)
        lower = torch.tensor(self._lower_values, dtype=self.dtype)
        range_ = torch.tensor(self._range_values, dtype=self.dtype)
        uniform = torch.rand(self.num_samples, len(self.bounds), dtype=self.dtype)
        points = lower + uniform * range_  # broadcast scaling

        fidelity_indices = torch.randint(0, len(self.fidelities), (self.num_samples,))
        fidelities = [self.fidelities[idx] for idx in fidelity_indices.tolist()]
        return [
            Candidate(x=points[i].tolist(), fidelity=fidelities[i])
            for i in range(self.num_samples)
        ]
