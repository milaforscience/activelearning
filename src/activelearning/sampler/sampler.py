from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

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
        observations: Optional[Sequence[Observation]] = None,
        **kwargs: Any,
    ) -> Sequence[Candidate]:
        """Generate a pool of candidate samples.

        Args:
            acquisition: Acquisition function to score candidates (optional).
            observations: Current observations to avoid resampling (optional).
            **kwargs: Additional sampler-specific arguments.

        Returns:
            Sequence of sampled candidates.
        """
        pass
