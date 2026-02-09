from abc import ABC, abstractmethod
from typing import Sequence

from activelearning.utils.types import Candidate, Observation


class Dataset(ABC):
    """Abstract dataset interface for managing observations and candidates."""

    @abstractmethod
    def add_samples(self, samples: Sequence[Candidate], scores) -> None:
        """Add labeled samples to the dataset as observations."""
        pass

    @abstractmethod
    def get_observations(self) -> Sequence[Observation]:
        """Return all stored observations."""
        pass
