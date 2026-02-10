from abc import ABC, abstractmethod
from typing import Sequence

from activelearning.utils.types import Observation


class Dataset(ABC):
    """Abstract dataset interface for managing observations and candidates.

    Datasets store observations (labeled data points) and provide access
    to them for model training and evaluation.
    """

    @abstractmethod
    def add_observations(self, observations: Sequence[Observation]) -> None:
        """Add new observations to the dataset.

        Args:
            observations: Sequence of observations to add.
        """
        pass

    @abstractmethod
    def get_observations(self) -> Sequence[Observation]:
        """Retrieve all stored observations.

        Returns:
            Sequence of all observations in the dataset.
        """
        pass
