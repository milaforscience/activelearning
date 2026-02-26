from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation

if TYPE_CHECKING:
    from activelearning.dataset.dataset import Dataset


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

    _KNOWN_STD: float = 0.1
    _UNKNOWN_STD: float = 1.0

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

        if len(observations) == 0:
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
            Known candidates return cached values with low std (_KNOWN_STD),
            unknown candidates return global mean with high std (_UNKNOWN_STD).
        """
        means = []
        stds = []
        for candidate in candidates:
            key = (candidate.x, candidate.fidelity)
            if key in self._model:
                means.append(self._model[key])
                stds.append(self._KNOWN_STD)
            else:
                means.append(self._mean_score)
                stds.append(self._UNKNOWN_STD)
        return {"mean": means, "std": stds}

    def update(self, dataset: "Dataset") -> None:
        """Update the surrogate by refitting with all observations from the dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing all observations.

        Notes
        -----
            DummySurrogate always performs full refitting regardless of the update method.
            It does not support incremental learning.
        """
        self.fit(dataset.get_observations_iterable())
