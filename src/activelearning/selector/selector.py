from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from activelearning.utils.types import Candidate
from activelearning.runtime import ALRuntimeMixin


@dataclass(frozen=True)
class SelectionScores:
    """Scores captured by a selector for optional round diagnostics."""

    acquisition_scores: tuple[float, ...]
    ranking_scores: tuple[float, ...]
    selected_indices: tuple[int, ...]


class Selector(ABC, ALRuntimeMixin):
    """Abstract selector interface used to choose candidates.

    Selectors implement strategies for choosing the final subset of
    candidates to query from a larger pool.
    """

    def _clear_selection_scores(self) -> None:
        """Discard score data left by a previous selection call."""
        self._pending_selection_scores: SelectionScores | None = None

    def _record_selection_scores(
        self,
        acquisition_scores: Sequence[float],
        ranking_scores: Sequence[float],
        selected_indices: Sequence[int],
    ) -> None:
        """Store one completed selection's scores for optional diagnostics."""
        self._pending_selection_scores = SelectionScores(
            acquisition_scores=tuple(float(score) for score in acquisition_scores),
            ranking_scores=tuple(float(score) for score in ranking_scores),
            selected_indices=tuple(int(index) for index in selected_indices),
        )

    def drain_selection_scores(self) -> SelectionScores | None:
        """Return and clear scores captured by the most recent selection."""
        scores = getattr(self, "_pending_selection_scores", None)
        self._pending_selection_scores = None
        return scores

    @abstractmethod
    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Any] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
    ) -> list[Candidate]:
        """Select candidates from a pool based on a specific strategy.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Pool of candidates to select from.
        acquisition : Optional[Any]
            Acquisition function to score candidates (optional).
            Required by score-based selectors, unused by random selectors.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Function to compute per-candidate costs (optional).
            Required by cost-aware selectors.
        round_budget : Optional[float]
            Budget limit for this round (optional).
            Required by cost-aware selectors.
        Returns
        -------
        result : list[Candidate]
            Selected subset of the input candidates, in query order. May be empty.
        """
        pass
