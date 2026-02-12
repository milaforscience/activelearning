from abc import ABC, abstractmethod
from typing import Sequence, Iterable

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
    def get_observations_iterable(self) -> Iterable[Observation]:
        """Retrieve an iterable over all stored observations.

        This method may be called multiple times during the active learning loop
        (e.g., once for surrogate fitting, once for acquisition update, etc.).
        Each call should return a fresh iterable over the current observations.

        Returns:
            Iterable of all observations in the dataset. May be:
            - A list for small in-memory datasets (cheap to return multiple times)
            - A DataLoader or generator for large datasets (creates fresh iterator each call)
            - Any iterable that can be traversed to access observations

        Note:
            Concrete implementations should ensure this method is efficient to call
            multiple times per active learning iteration. For example:
            - Returning a list reference is O(1)
            - Creating a new DataLoader is typically fast (just wraps existing data)
            - Avoid expensive recomputation on each call

        Examples:
            Simple list implementation (cheap multiple calls):
            ```python
            def get_observations_iterable(self):
                return self._records  # Returns same list each time
            ```

            DataLoader implementation (fresh iterator each call):
            ```python
            def get_observations_iterable(self):
                return DataLoader(self._dataset)  # New DataLoader each time
            ```
        """
        pass

    def get_best_candidates(self, k: int = 1) -> list[Observation]:
        """Return the top-k observations by y value from the dataset.

        Args:
            k: Number of top observations to return.

        Returns:
            List of top-k observations sorted by y value (descending).
            Returns empty list if no observations exist.

        Note:
            Filters out observations with None y values.
            Assumes y values support comparison operations.
        """
        observations = list(self.get_observations_iterable())
        if not observations:
            return []
        valid_obs = [o for o in observations if o.y is not None]
        sorted_records = sorted(valid_obs, key=lambda r: r.y, reverse=True)
        return sorted_records[:k]
