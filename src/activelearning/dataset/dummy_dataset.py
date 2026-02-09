from typing import Any, Sequence

from activelearning.dataset.dataset import Dataset
from activelearning.utils.types import Candidate, Observation


class DummyDataset(Dataset):
    """In-memory dataset storing observations in a list."""

    def __init__(self) -> None:
        self._records: list[Observation] = []

    def add_samples(self, samples: Sequence[Candidate], scores: Sequence[Any]) -> None:
        """Add new observations by pairing samples with their scores.

        Args:
            samples: Sequence of candidate samples.
            scores: Corresponding scores for each sample.
        """
        for sample, score in zip(samples, scores):
            self._records.append(
                Observation(x=sample.x, fidelity=sample.fidelity, y=score)
            )

    def get_observations(self) -> list[Observation]:
        """Retrieve all stored observations.

        Returns:
            List of all observations stored in the dataset.
        """
        return list(self._records)
