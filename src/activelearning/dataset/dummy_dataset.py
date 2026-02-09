from typing import Sequence

from activelearning.dataset.dataset import Dataset
from activelearning.utils.types import Candidate, Observation


class DummyDataset(Dataset):
    """In-memory dataset storing observations in a list."""

    def __init__(self):
        self._records = []

    def add_samples(self, samples: Sequence[Candidate], scores) -> None:
        """Append observations for the given samples and scores."""
        for sample, score in zip(samples, scores):
            self._records.append(
                Observation(x=sample.x, fidelity=sample.fidelity, y=score)
            )

    def get_observations(self):
        """Return all stored observations."""
        return list(self._records)
