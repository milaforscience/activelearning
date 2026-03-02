from typing import Any, Iterable, Mapping, Sequence

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


class DummyMeanSurrogate(Surrogate):
    """Minimal in-memory surrogate using cached observations.

    Stores observations keyed by (x, fidelity). When predicting, returns the
    observed value for known candidates and the global mean for unseen ones.
    Useful for testing and baseline comparisons.

    Notes
    -----
    This surrogate requires a Sequence (supports len() and multiple iterations).
    Automatically materializes iterables to lists as needed.
    """

    def __init__(self) -> None:
        self._model: dict[tuple[Any, Any], Any] = {}
        self._mean_score: float = 0.0

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fit the surrogate by caching observations and computing global mean.

        Parameters
        ----------
        observations : Iterable[Observation]
            Iterable of observations to cache. Will be materialized
            to a Sequence if needed for len() and multiple iteration.
        """
        # Materialize to list if not already a Sequence
        if not isinstance(observations, Sequence):
            observations = list(observations)

        if not observations:
            self._mean_score = 0.0
            self._model = {}
            return
        self._model = {(obs.x, obs.fidelity): obs.y for obs in observations}
        self._mean_score = sum(obs.y for obs in observations) / len(observations)

    def predict(self, candidates: Sequence[Candidate]) -> Mapping[str, list[float]]:
        """Predict mean and standard deviation for candidates.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to predict.

        Returns
        -------
        result : Mapping[str, list[float]]
            Dictionary with "mean" and "std" keys containing predictions.
            Known candidates return cached values with low std (0.1),
            unknown candidates return global mean with high std (1.0).
        """
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
