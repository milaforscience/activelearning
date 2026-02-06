from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Candidate:
    """Represents a candidate item to be evaluated or sampled."""

    x: Any
    fidelity: Optional[int] = None


@dataclass(frozen=True)
class Observation:
    """Represents an observed (x, y) pair, optionally at a fidelity."""

    x: Any
    y: Any
    fidelity: Optional[int] = None
