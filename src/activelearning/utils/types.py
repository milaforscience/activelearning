from dataclasses import dataclass
from typing import Any, Sequence, Optional


@dataclass(frozen=True)
class Candidate:
    """Represents a candidate item to be evaluated or sampled.

    Attributes:
        x: Input feature or identifier for the candidate.
        fidelity: Optional fidelity level (e.g., for multi-fidelity optimization).
    """

    x: Any
    fidelity: Optional[int] = None


@dataclass(frozen=True)
class Observation:
    """Represents an observed (x, y) pair, optionally at a fidelity.

    Attributes:
        x: Input feature or identifier for the observation.
        y: Observed output or label value.
        fidelity: Optional fidelity level at which the observation was made.
    """

    x: Any
    y: Any
    fidelity: Optional[int] = None


def label_candidates(
    candidates: Sequence[Candidate], labels: Sequence[Any]
) -> Sequence[Observation]:
    """Convert a sequence of candidates and their corresponding labels into observations.

    Args:
        candidates: Sequence of Candidate objects.
        labels: Sequence of label values corresponding to each candidate.

    Returns:
        Sequence of Observation objects, where each observation combines the
        candidate's x and fidelity with its label as y.
    """
    assert len(candidates) == len(labels), "Length of candidates and labels must match."
    return [
        Observation(x=candidate.x, y=label, fidelity=candidate.fidelity)
        for candidate, label in zip(candidates, labels)
    ]
