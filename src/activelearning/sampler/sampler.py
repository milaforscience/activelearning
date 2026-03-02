from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Sequence

from activelearning.utils.types import Candidate, Observation


class Sampler(ABC):
    """Abstract sampler interface used to propose candidate subsets.

    Samplers generate candidate pools from which selectors choose the
    final candidates to query.
    """

    @abstractmethod
    def sample(
        self,
        acquisition: Optional[Any] = None,
        observations: Optional[Iterable[Observation]] = None,
        **kwargs: Any,
    ) -> Sequence[Candidate]:
        """Generate a pool of candidate samples.

        Parameters
        ----------
        acquisition : Optional[Any]
            Acquisition function to score candidates (optional).
            Used by samplers that weight candidates by acquisition values.
        observations : Optional[Iterable[Observation]]
            Current observations to avoid resampling (optional).
        **kwargs
            Additional sampler-specific arguments.

        Returns
        -------
        result : Sequence[Candidate]
            Sequence of sampled candidates.
        """
        pass
