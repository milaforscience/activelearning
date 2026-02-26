import torch

from typing import Any, Iterable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, Observation


class HypercubeUniformSampler(Sampler):
    """Generates candidates by sampling uniformly from a bounded hypercube.

    Unlike pool-based samplers, this sampler is generative: it draws fresh
    random candidates on every ``sample()`` call. This makes it suitable for
    continuous-domain functions such as ``BraninOracle`` and
    ``Hartmann6DOracle`` where no finite candidate pool exists.

    Fidelity levels are sampled uniformly at random from ``fidelities`` for
    each candidate, enabling multi-fidelity exploration without a fixed pool.

    Parameters
    ----------
    bounds : Sequence[tuple[float, float]]
        Per-dimension ``(lower, upper)`` bounds that define the hypercube.
        Each pair must satisfy ``lower < upper``.
        The length determines the input dimensionality.
    num_samples : int
        Number of candidates to generate per ``sample()`` call. Must be > 0.
    fidelities : Optional[Sequence[int]]
        Fidelity levels to sample from uniformly at random. Each generated
        candidate is assigned one level drawn with equal probability. When
        ``None``, candidates are created with ``fidelity=None``.

    Raises
    ------
    ValueError
        If any lower bound >= upper bound or num_samples <= 0.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        num_samples: int,
        fidelities: Optional[Sequence[int]] = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        for i, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValueError(f"Bound {i} has lower >= upper: ({lower}, {upper})")

        if fidelities is not None and len(fidelities) == 0:
            raise ValueError("fidelities must not be empty when specified")
        self.bounds = bounds
        self.num_samples = num_samples
        self.fidelities = list(fidelities) if fidelities is not None else None

        lowers = [b[0] for b in bounds]
        uppers = [b[1] for b in bounds]
        self._lower = torch.tensor(lowers, dtype=torch.float64)
        self._range = torch.tensor(
            [upper - lower for lower, upper in zip(lowers, uppers)], dtype=torch.float64
        )

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
        **kwargs: Any,
    ) -> list[Candidate]:
        """Generate candidates by sampling uniformly from the hypercube.

        Each candidate's fidelity is drawn uniformly at random from the
        ``fidelities`` list supplied at construction.

        Parameters
        ----------
        acquisition : Optional[Acquisition]
            Unused. Present for interface compatibility.
        observations : Optional[Iterable[Observation]]
            Unused. Present for interface compatibility.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        result : list[Candidate]
            ``num_samples`` candidates with ``x`` as a plain Python list of
            floats and ``fidelity`` drawn uniformly from ``fidelities``.
        """
        # Shape: (num_samples, n_dims)
        uniform = torch.rand(self.num_samples, len(self.bounds), dtype=torch.float64)
        points = self._lower + uniform * self._range  # broadcast scaling

        if self.fidelities is not None:
            fidelity_indices = torch.randint(
                0, len(self.fidelities), (self.num_samples,)
            )
            fidelities = [self.fidelities[idx] for idx in fidelity_indices.tolist()]
        else:
            fidelities = [None] * self.num_samples

        return [
            Candidate(x=points[i].tolist(), fidelity=fidelities[i])
            for i in range(self.num_samples)
        ]
