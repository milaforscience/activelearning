from typing import Iterable

from activelearning.acquisition.acquisition import Acquisition
from activelearning.utils.types import Candidate


class DummyAcquisition(Acquisition):
    """UCB-style acquisition over "mean" and optional "std" predictions.

    Computes acquisition values as mean + beta * std when std is available,
    otherwise returns mean values only.

    Surrogate Requirements:
        Requires surrogate.predict() to return a dict with "mean" key.
        Optionally uses "std" key if available for uncertainty-based exploration.

    Parameters
    ----------
    beta : float
        Exploration parameter controlling the weight of uncertainty.
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self._beta = float(beta)

    def score(self, candidates: Iterable[Candidate]) -> list[float]:
        """Compute UCB acquisition values for candidates.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Candidates to evaluate independently.

        Returns
        -------
        result : list[float]
            List of acquisition values (mean + beta * std).

        Raises
        ------
        ValueError
            If surrogate is not set or does not implement predict().
        ValueError
            If predict() does not return required "mean" key.
        """
        candidate_list = list(candidates)
        if self.surrogate is None:
            return [0.0 for _ in candidate_list]

        try:
            pred = self.surrogate.predict(candidate_list)
        except NotImplementedError as e:
            raise ValueError(
                f"DummyAcquisition requires surrogate.predict() but "
                f"{self.surrogate.__class__.__name__} does not implement it."
            ) from e

        means = pred.get("mean")
        if means is None:
            raise ValueError("DummyAcquisition expects prediction payload key 'mean'.")
        stds = pred.get("std")
        if stds is None:
            return list(means)
        return [mean + self._beta * std for mean, std in zip(means, stds)]

    def supports_singleton_scoring(self) -> bool:
        """Return whether singleton scoring is supported."""
        return True
