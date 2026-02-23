from typing import Sequence

from activelearning.dataset.dataset import Dataset
from activelearning.utils.types import Observation


class DummyDataset(Dataset):
    """In-memory dataset storing observations in a list."""

    def __init__(self) -> None:
        self._records: list[Observation] = []

    def add_observations(self, observations: Sequence[Observation]) -> None:
        """Add new observations to the dataset by appending to the list of records.

        Parameters
        ----------
        observations : Sequence[Observation]
            Sequence of observations to add.
        """
        self._records.extend(observations)

    def get_observations_iterable(self) -> list[Observation]:
        """Retrieve all stored observations.

        Returns
        -------
        observations_iterable : list[Observation]
            List of all observations stored in the dataset.
        """
        return list(self._records)
