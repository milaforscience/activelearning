from dataclasses import dataclass
from typing import Any, Optional


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
