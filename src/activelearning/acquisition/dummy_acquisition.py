from typing import Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.utils.types import Candidate


class DummyAcquisition(Acquisition):
    """UCB-style acquisition over "mean" and optional "std" predictions.

    Computes acquisition scores as mean + beta * std when std is available,
    otherwise returns mean values only.

    Args:
        beta: Exploration parameter controlling the weight of uncertainty.
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self._beta = float(beta)

    def __call__(self, candidates: Sequence[Candidate]) -> list[float]:
        """Compute UCB scores for candidates.

        Args:
            candidates: Sequence of candidates to score.

        Returns:
            List of acquisition scores (mean + beta * std).
        """
        if self.surrogate is None:
            return [0.0 for _ in candidates]
        pred = self.surrogate.predict(candidates)
        means = pred.get("mean")
        if means is None:
            raise ValueError("DummyAcquisition expects prediction payload key 'mean'.")
        stds = pred.get("std")
        if stds is None:
            return list(means)
        return [mean + self._beta * std for mean, std in zip(means, stds)]
