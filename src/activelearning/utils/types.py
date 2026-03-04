from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Optional

import torch


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


def _to_2d_tensor(values: list[Any], dtype: torch.dtype) -> torch.Tensor:
    """Convert a list of values to a 2D float tensor.

    Tries fast batch conversion first; falls back to element-wise stacking
    for heterogeneous or non-array types.

    Parameters
    ----------
    values : list[Any]
        List of scalar, array-like, or tensor values of uniform shape.
    dtype : torch.dtype
        Target dtype for the output tensor.

    Returns
    -------
    result : torch.Tensor
        2D tensor of shape (n, d). Scalar inputs produce shape (n, 1).
    """
    try:
        t = torch.as_tensor(values, dtype=dtype)
        return t.unsqueeze(-1) if t.dim() == 1 else t
    except (TypeError, ValueError, RuntimeError):
        return torch.stack(
            [torch.atleast_1d(torch.as_tensor(v, dtype=dtype)) for v in values]
        )


def observations_to_tensors(
    observations: Iterable[Observation],
    fidelity_confidences: Optional[dict[int, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Convert an iterable of Observations to tensors.

    Tries fast batch conversion for inputs and labels; falls back to
    element-wise stacking for heterogeneous types.

    Parameters
    ----------
    observations : Iterable[Observation]
        Observations to convert. Materialized to a list if not already a list.
    fidelity_confidences : dict[int, float], optional
        Mapping from integer fidelity IDs to continuous confidence values.
        If None or empty, single-fidelity mode is assumed and fidelities
        returns as an empty list. Raises ``KeyError`` if a fidelity ID in
        the observations is missing from the mapping.

    Returns
    -------
    train_X : torch.Tensor
        Input features tensor of shape (n, d).
    train_Y : torch.Tensor
        Output labels tensor of shape (n, 1).
    fidelities : list[float]
        Mapped fidelity confidence values; empty if single-fidelity.
    """
    obs_list = observations if isinstance(observations, list) else list(observations)

    train_X = _to_2d_tensor([obs.x for obs in obs_list], torch.float64)
    train_Y = torch.as_tensor(
        [obs.y for obs in obs_list], dtype=torch.float64
    ).view(-1, 1)
    fidelities = (
        [fidelity_confidences[obs.fidelity] for obs in obs_list if obs.fidelity is not None]
        if fidelity_confidences
        else []
    )

    return train_X, train_Y, fidelities


def candidates_to_tensor(
    candidates: Sequence[Candidate],
    fidelity_confidences: Optional[dict[int, float]] = None,
) -> tuple[torch.Tensor, list[float]]:
    """Convert a sequence of Candidates to a tensor.

    Tries fast batch conversion for inputs; falls back to element-wise
    stacking for heterogeneous types.

    Parameters
    ----------
    candidates : Sequence[Candidate]
        Candidates to convert.
    fidelity_confidences : dict[int, float], optional
        Mapping from integer fidelity IDs to continuous confidence values.
        If None or empty, single-fidelity mode is assumed and fidelities
        returns as an empty list. Raises ``KeyError`` if a fidelity ID in
        the candidates is missing from the mapping.

    Returns
    -------
    test_X : torch.Tensor
        Input features tensor of shape (n, d).
    fidelities : list[float]
        Mapped fidelity confidence values; empty if single-fidelity.
    """
    test_X = _to_2d_tensor([cand.x for cand in candidates], torch.float64)
    fidelities = (
        [fidelity_confidences[cand.fidelity] for cand in candidates if cand.fidelity is not None]
        if fidelity_confidences
        else []
    )

    return test_X, fidelities
