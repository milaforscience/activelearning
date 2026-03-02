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

        Parameters
        ----------
        observations : Sequence[Observation]
            Sequence of observations to add.
        """
        pass

    @abstractmethod
    def get_observations_iterable(self) -> Iterable[Observation]:
        """Retrieve an iterable over all stored observations.

        This method may be called multiple times during the active learning loop
        (e.g., once for surrogate fitting, once for acquisition update, etc.).
        Each call should return a fresh iterable over the current observations.

        Returns
        -------
        observations_iterable : Iterable[Observation]
            Iterable of all observations in the dataset. May be:
            Iterable[Observation]
            - A list for small in-memory datasets (cheap to return multiple times)
            - A DataLoader or generator for large datasets (creates fresh iterator each call)
            - Any iterable that can be traversed to access observations

        Notes
        -----
        Concrete implementations should ensure this method is efficient to call
        multiple times per active learning iteration. For example:
        - Returning a list reference is O(1)
        - Creating a new DataLoader is typically fast (just wraps existing data)
        - Avoid expensive recomputation on each call

        Examples
        --------
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

    @abstractmethod
    def get_best_candidates(self, k: int = 1) -> list[Observation]:
        """Return the top-k "best" observations from the dataset.

        The definition of "best" is problem-specific and should be implemented
        by concrete subclasses. For example:
        - Maximization problems: higher y values are better
        - Minimization problems: lower y values are better
        - Categorical problems: may require different ranking criteria

        Parameters
        ----------
        k : int
            Number of top observations to return.

        Returns
        -------
        best_candidates : list[Observation]
            List of top-k observations according to the problem-specific
            definition of "best". Returns empty list if no observations exist.

        Notes
        -----
        Subclasses should define what "best" means for their specific problem
        and implement appropriate ranking logic. Observations with None y values
        should typically be filtered out.
        """
        pass
