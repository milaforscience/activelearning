from abc import ABC, abstractmethod


class Oracle(ABC):
    """Abstract oracle interface used to obtain labels and costs."""

    @abstractmethod
    def get_cost(self, candidates):
        """Get the cost of querying the given candidates."""
        pass

    @abstractmethod
    def query(self, candidates):
        """Query the oracle for labels of the given candidates."""
        pass
