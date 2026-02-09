from abc import ABC, abstractmethod
from typing import Any, Sequence

from activelearning.utils.types import Candidate, Observation


class Dataset(ABC):
    """Abstract dataset interface for managing observations and candidates.

    Datasets store observations (labeled data points) and provide access
    to them for model training and evaluation.
    """

    @abstractmethod
    def add_samples(self, samples: Sequence[Candidate], scores: Sequence[Any]) -> None:
        """Add labeled samples to the dataset as observations.

        Args:
            samples: Sequence of candidates to add.
            scores: Corresponding labels/scores for each candidate.
        """
        pass

    @abstractmethod
    def get_observations(self) -> Sequence[Observation]:
        """Retrieve all stored observations.

        Returns:
            Sequence of all observations in the dataset.
        """
        pass
