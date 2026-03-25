from abc import ABC, abstractmethod
from typing import Sequence, Iterable

from activelearning.utils.types import Observation
from activelearning.runtime import ALRuntimeMixin


class Dataset(ABC, ALRuntimeMixin):
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
        """Retrieve an iterable over all stored observations for one AL round.

        The returned iterable must support **consistent multiple iterations**:
        calling ``iter()`` on it more than once must yield the same sequence of
        observations in the same order. This guarantee allows a single call per
        round to be shared across all consumers (surrogate, acquisition,
        sampler) without risk of data inconsistency.

        Implementations are responsible for pinning any randomness at call
        time. For example, a DataLoader-backed dataset should pre-generate the
        shuffled index permutation here so that every pass over the returned
        iterable replays the same order without holding all data in memory.

        Returns
        -------
        observations_iterable : Iterable[Observation]
            Consistently re-iterable view of all current observations.

        Examples
        --------
        In-memory list (trivially consistent)::

            def get_observations_iterable(self):
                return list(self._records)

        Large-dataset DataLoader (pin permutation at call time)::

            def get_observations_iterable(self):
                indices = torch.randperm(len(self._dataset)).tolist()
                return DataLoader(self._dataset, sampler=indices)
        """
        pass

    @abstractmethod
    def get_latest_observations_iterable(self) -> Iterable[Observation]:
        """Retrieve an iterable over the most recently added observations.

        Returns observations from the most recent call to add_observations().
        This is useful for incremental model updates where only new data needs
        to be processed.

        The same consistency guarantee as ``get_observations_iterable()`` applies:
        the returned iterable must support consistent multiple iterations. Implementations
        are responsible for pinning any randomness at call time.

        Returns
        -------
        latest_observations_iterable : Iterable[Observation]
            Iterable of observations from the most recent add_observations() call.
            Returns an empty iterable if no observations have been added yet.

        Notes
        -----
        - This method should return only the observations from the last
            add_observations() call, not all historical observations
        - Like get_observations_iterable(), this may be called multiple times
            and should return a fresh iterable each time
        - If add_observations() is called again, subsequent calls to this method
            should return the new batch, not the previous one

        Examples
        --------
        After first batch::

            dataset.add_observations([obs1, obs2, obs3])
            latest = list(dataset.get_latest_observations_iterable())
            # latest == [obs1, obs2, obs3]

        After second batch::

            dataset.add_observations([obs4, obs5])
            latest = list(dataset.get_latest_observations_iterable())
            # latest == [obs4, obs5]  (not obs1-obs5)
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
