from abc import ABC, abstractmethod


class Selector(ABC):
    """Abstract selector interface used to choose candidates."""

    @abstractmethod
    def __call__(self, candidates):
        """Select from a pool of candidates based on a specific strategy."""
        pass
