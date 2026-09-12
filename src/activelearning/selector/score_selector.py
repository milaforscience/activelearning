from typing import Callable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.acquisition.cost_utility import cost_weighting_from_cost_fn
from activelearning.selector.selector import Selector
from activelearning.utils.types import Candidate


class TopKAcquisitionSelector(Selector):
    """Selector that ranks candidates by acquisition value and selects the top-k.

    Parameters
    ----------
    num_samples : int
        Number of top-scored candidates to select.
    """

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Acquisition] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
    ) -> list[Candidate]:
        """Select the top num_samples candidates by acquisition value.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Pool of candidates to select from.
        acquisition : Optional[Acquisition]
            Acquisition function to score candidates.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Optional candidate cost function. When provided, candidates are
            ranked by inverse-cost-weighted acquisition values.
        round_budget : Optional[float]
            Budget limit (ignored by this selector).

        Returns
        -------
        result : list[Candidate]
            List of top candidates by ranking score.
        """
        if acquisition is None:
            raise ValueError("Acquisition function is required for ScoreSelector.")
        if not candidates:
            return []

        if cost_fn is None:
            acq_values = acquisition.score(candidates)
        else:
            acq_values = acquisition.score(
                candidates,
                cost_weighting=cost_weighting_from_cost_fn(cost_fn),
            )
        ranked = sorted(zip(candidates, acq_values), key=lambda cv: cv[1], reverse=True)
        return [candidate for candidate, _ in ranked[: self.num_samples]]
