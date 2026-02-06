from abc import ABC, abstractmethod
from typing import Sequence

from activelearning.utils.types import Candidate, Observation


class Dataset(ABC):
    """Abstract dataset interface for managing observations and candidates."""

    @abstractmethod
    def add_samples(self, samples: Sequence[Candidate], scores) -> None:
        """Add labeled samples to the dataset."""
        pass

    @abstractmethod
    def get_top_k(self, k=1):
        """Return the top-k samples by score."""
        pass

    @abstractmethod
    def get_observations(self) -> Sequence[Observation]:
        """Return observations used by the surrogate."""
        pass
