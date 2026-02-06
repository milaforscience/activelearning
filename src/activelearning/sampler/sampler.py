from abc import ABC, abstractmethod


class Sampler(ABC):
    """Abstract sampler interface used to propose candidate subsets."""

    @abstractmethod
    def sample(self, **kwargs):
        pass
