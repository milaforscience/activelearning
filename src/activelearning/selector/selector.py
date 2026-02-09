from abc import ABC, abstractmethod


class Selector(ABC):
    """Abstract selector interface used to choose candidates."""

    @abstractmethod
    def __call__(self, candidates, acquisition=None, **kwargs):
        """Select from a pool of candidates based on a specific strategy.

        Args:
            candidates: The candidates to select from.
            acquisition: The acquisition function used to score candidates.
            **kwargs: Additional arguments.
        """
        pass
