from abc import ABC, abstractmethod


class Sampler(ABC):
    """Abstract sampler interface used to propose candidate subsets."""

    @abstractmethod
    def sample(self, acquisition=None, observations=None, **kwargs):
        """Propose candidate subsets.

        Args:
            acquisition: The acquisition function used to score candidates.
            observations: The current set of observations (optional).
            **kwargs: Additional arguments for specific sampler implementations.
        """
        pass
