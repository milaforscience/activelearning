from dataclasses import dataclass
from typing import Any, Sequence, Optional


@dataclass(frozen=True)
class Candidate:
    """Represents a candidate item to be evaluated or sampled.

    Uses maximum type flexibility to support various data representations.

    Attributes
    ----------
    x : Any
        Input feature or identifier. Commonly: primitives (int, float, str),
        arrays (numpy.ndarray), tensors (torch.Tensor), or structured data (dict, tuple).
    fidelity : Optional[int]
        Optional fidelity level for multi-fidelity optimization.
        Higher fidelity typically means more accurate but more expensive.
    """

    x: Any
    fidelity: Optional[int] = None


@dataclass(frozen=True)
class Observation:
    """Represents an observed (x, y) pair, optionally at a fidelity.

    Uses maximum type flexibility to support various data representations.

    Attributes
    ----------
    x : Any
        Input feature or identifier. Same semantics as Candidate.x.
    y : Any
        Observed output or label. Commonly: scalar (float), vector (list, array),
        or categorical label (str, int).
    fidelity : Optional[int]
        Optional fidelity level at which the observation was made.
    """

    x: Any
    y: Any
    fidelity: Optional[int] = None


def label_candidates(
    candidates: Sequence[Candidate], labels: Sequence[Any]
) -> Sequence[Observation]:
    """Convert a sequence of candidates and their corresponding labels into observations.

    Parameters
    ----------
    candidates : Sequence[Candidate]
        Sequence of Candidate objects.
    labels : Sequence[Any]
        Sequence of label values corresponding to each candidate.

    Returns
    -------
    result : Sequence[Observation]
        Sequence of Observation objects, where each observation combines the
        candidate's x and fidelity with its label as y.
    """
    if len(candidates) != len(labels):
        raise ValueError("Length of candidates and labels must match.")
    return [
        Observation(x=candidate.x, y=label, fidelity=candidate.fidelity)
        for candidate, label in zip(candidates, labels)
    ]
