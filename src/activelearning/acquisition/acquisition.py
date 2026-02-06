from abc import ABC, abstractmethod
from typing import Sequence, Optional

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate


class Acquisition(ABC):
    """Abstract acquisition interface used to score candidates."""

    def __init__(self):
        self._surrogate: Optional[Surrogate] = None

    @property
    def surrogate(self) -> Optional[Surrogate]:
        return getattr(self, "_surrogate", None)

    def update(self, surrogate: Surrogate) -> None:
        """Update internal state based on the surrogate."""
        self._surrogate = surrogate

    @abstractmethod
    def __call__(self, candidates: Sequence[Candidate]):
        """Compute acquisition values for given candidates."""
        pass
