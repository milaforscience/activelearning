from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Sequence

from activelearning.utils.types import Candidate, Observation
from activelearning.runtime import ALRuntimeMixin


class Sampler(ABC, ALRuntimeMixin):
    """Abstract sampler interface used to propose candidate subsets.

    Samplers generate candidate pools from which selectors choose the
    final candidates to query. Some samplers ignore acquisition scores and
    costs entirely, while others use them to bias which candidates enter the
    pool.
    """

    @abstractmethod
    def sample(
        self,
        acquisition: Optional[Any] = None,
        observations: Optional[Iterable[Observation]] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> Sequence[Candidate]:
        """Generate a pool of candidate samples.

        Parameters
        ----------
        acquisition : Optional[Any]
            Acquisition function used by score-aware samplers. Uniform and
            purely generative samplers may ignore it.
        observations : Optional[Iterable[Observation]]
            Current observations. Samplers may use these to avoid resampling
            previously observed candidates or to warm-start internal state.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Per-candidate cost function. Samplers that support cost-aware
            weighting may use it to transform acquisition scores before
            sampling.

        Returns
        -------
        result : Sequence[Candidate]
            Sequence of sampled candidates.
        """
        pass
