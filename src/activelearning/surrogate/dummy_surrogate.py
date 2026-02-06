from typing import Sequence

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


class DummySurrogate(Surrogate):
    """Minimal in-memory surrogate using cached observations.

    Stores observations keyed by (x, fidelity). When predicting, it returns the
    observed value for known candidates and falls back to the global mean for
    unseen ones.
    """

    def __init__(self):
        self._model = {}
        self._mean_score = 0.0

    def fit(self, observations: Sequence[Observation]) -> None:
        """Cache observations and update the global mean."""
        if not observations:
            self._mean_score = 0.0
            self._model = {}
            return
        self._model = {(obs.x, obs.fidelity): obs.y for obs in observations}
        self._mean_score = sum(obs.y for obs in observations) / len(observations)

    def predict(self, candidates: Sequence[Candidate]):
        """Return "mean" and optional "std" predictions for candidates."""
        means = []
        stds = []
        for candidate in candidates:
            key = (candidate.x, candidate.fidelity)
            if key in self._model:
                means.append(self._model[key])
                stds.append(0.1)
            else:
                means.append(self._mean_score)
                stds.append(1.0)
        return {"mean": means, "std": stds}
