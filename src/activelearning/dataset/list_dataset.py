import heapq
from typing import Sequence

from activelearning.dataset.dataset import Dataset
from activelearning.utils.types import Observation


class ListDataset(Dataset):
    """In-memory dataset storing observations in a list."""

    def __init__(self) -> None:
        self._records: list[Observation] = []
        self._latest_records: list[Observation] = []

    def add_observations(self, observations: Sequence[Observation]) -> None:
        """Add new observations to the dataset by appending to the list of records.

        Parameters
        ----------
        observations : Sequence[Observation]
            Sequence of observations to add.
        """
        self._records.extend(observations)
        self._latest_records = list(observations)

    def get_observations_iterable(self) -> list[Observation]:
        """Retrieve all stored observations.

        Returns
        -------
        observations_iterable : list[Observation]
            List of all observations stored in the dataset.
        """
        return list(self._records)

    def get_latest_observations_iterable(self) -> list[Observation]:
        """Retrieve the most recently added observations.

        Returns
        -------
        latest_observations_iterable : list[Observation]
            List of observations from the most recent add_observations() call.
            Returns an empty list if no observations have been added yet.
        """
        return list(self._latest_records)

    def get_best_candidates(self, k: int = 1) -> list[Observation]:
        """Return the top-k observations with highest y values.

        This implementation assumes a maximization problem where higher
        y values are better. For minimization or other problem types,
        subclass DummyDataset and override this method.

        Parameters
        ----------
        k : int
            Number of top observations to return.

        Returns
        -------
        best_candidates : list[Observation]
            List of top-k observations sorted by y value (descending).
            Returns empty list if no observations exist.

        Notes
        -----
        Filters out observations with None y values.
        Assumes y values support comparison operations (numerical).
        Uses heapq.nlargest for efficient O(n log k) selection without
        creating unnecessary copies of the data.
        """
        if not self._records:
            return []
        # Use generator to avoid creating an intermediate list
        valid_obs = (o for o in self._records if o.y is not None)
        # heapq.nlargest is O(n log k) and doesn't sort the entire dataset
        return heapq.nlargest(k, valid_obs, key=lambda r: r.y)
