from typing import Any, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.selector.selector import Selector
from activelearning.utils.types import Candidate


class ScoreSelector(Selector):
    """Selector that ranks candidates by acquisition score and selects the top-k.

    Args:
        num_samples: Number of top-scored candidates to select.
    """

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Acquisition] = None,
        **kwargs: Any,
    ) -> list[Candidate]:
        """Select the top num_samples candidates by acquisition score.

        Args:
            candidates: Pool of candidates to select from.
            acquisition: Acquisition function to score candidates.
            **kwargs: Additional arguments (unused).

        Returns:
            List of top-scored candidates.

        Raises:
            ValueError: If acquisition function is not provided.
        """
        if acquisition is None:
            raise ValueError("Acquisition function is required for ScoreSelector.")

        values = acquisition(candidates)
        ranked = sorted(zip(candidates, values), key=lambda cv: cv[1], reverse=True)
        return [candidate for candidate, _ in ranked[: self.num_samples]]
