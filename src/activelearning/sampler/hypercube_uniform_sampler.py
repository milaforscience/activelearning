import random
import torch

from typing import Iterable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.sampler.fidelity_policy import DiscreteFidelityPolicy
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
    fidelity_policy : DiscreteFidelityPolicy or None
        Policy controlling fidelity assignment for generated candidates.

    Raises
    ------
    ValueError
        If any lower bound >= upper bound or num_samples <= 0.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        num_samples: int,
        fidelity_policy: DiscreteFidelityPolicy | None = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        for i, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValueError(f"Bound {i} has lower >= upper: ({lower}, {upper})")

        self.bounds = bounds
        self.num_samples = num_samples
        self.fidelity_policy = fidelity_policy

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
    ) -> list[Candidate]:
        """Generate candidates by sampling uniformly from the hypercube.

        Each candidate's fidelity is assigned by the configured policy.

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
            floats and ``fidelity`` drawn uniformly from ``fidelities``.
        """
        # Shape: (num_samples, n_dims)
        uniform = torch.rand(self.num_samples, len(self.bounds), dtype=torch.float64)
        points = self._lower + uniform * self._range  # broadcast scaling

        if self.fidelity_policy is None:
            fidelities = [None] * self.num_samples
        else:
            rng = random.Random(self.runtime_context.seed + self.active_learning_round)
            fidelities = self.fidelity_policy.sample_fidelities(self.num_samples, rng)

        return [
            Candidate(x=points[i].tolist(), fidelity=fidelities[i])
            for i in range(self.num_samples)
        ]
