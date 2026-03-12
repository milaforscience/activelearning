from dataclasses import dataclass
from typing import Any, Iterable, Optional

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
    candidates: Iterable[Candidate], labels: Iterable[Any]
) -> list[Observation]:
    """Convert candidates and their corresponding labels into observations.

    Parameters
    ----------
    candidates : Iterable[Candidate]
        Iterable of Candidate objects. Will be materialized.
    labels : Iterable[Any]
        Iterable of label values corresponding to each candidate. Will be materialized.

    Returns
    -------
    result : list[Observation]
        List of Observation objects, where each observation combines the
        candidate's x and fidelity with its label as y.
    """
    candidates_list = candidates if isinstance(candidates, list) else list(candidates)
    labels_list = labels if isinstance(labels, list) else list(labels)
    if len(candidates_list) != len(labels_list):
        raise ValueError("Length of candidates and labels must match.")
    return [
        Observation(x=candidate.x, y=label, fidelity=candidate.fidelity)
        for candidate, label in zip(candidates_list, labels_list)
    ]


def _to_tensor(values: list[Any], dtype: torch.dtype) -> torch.Tensor:
    """Convert a list of values to a tensor, preserving the natural shape.

    Tries fast batch conversion first; falls back to element-wise stacking
    for heterogeneous or non-array types.

    Parameters
    ----------
    values : list[Any]
        List of scalar, array-like, or tensor values.
    dtype : torch.dtype
        Target dtype for the output tensor.

    Returns
    -------
    result : torch.Tensor
        Tensor with shape determined by the input data. Scalar inputs produce
        shape (n,); array inputs produce shape (n, d).
    """
    try:
        return torch.as_tensor(values, dtype=dtype)
    except (TypeError, ValueError, RuntimeError):
        return torch.stack([torch.as_tensor(v, dtype=dtype) for v in values])


def observations_to_tensors(
    observations: Iterable[Observation],
    fidelity_confidences: Optional[dict[int, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Convert an iterable of Observations to tensors.

    Tries fast batch conversion for inputs and labels; falls back to
    element-wise stacking for heterogeneous types. The natural shape of
    the data is preserved — no reshaping is applied.

    Parameters
    ----------
    observations : Iterable[Observation]
        Observations to convert. Materialized to a list if not already a list.
    fidelity_confidences : dict[int, float], optional
        Mapping from integer fidelity IDs to continuous confidence values.
        If None, single-fidelity mode is assumed and fidelities returns as an
        empty list. If a mapping is provided, every non-None fidelity must be
        present in it. Raises ``KeyError`` if a fidelity ID in the observations
        is missing from the mapping.

    Returns
    -------
    X : torch.Tensor
        Input features tensor. Shape is determined by the input data.
    y : torch.Tensor
        Output labels tensor. Shape is determined by the input data.
    fidelities : list[float]
        Mapped fidelity confidence values; empty if single-fidelity.
    """
    obs_list = observations if isinstance(observations, list) else list(observations)

    X = _to_tensor([obs.x for obs in obs_list], torch.float64)
    y = _to_tensor([obs.y for obs in obs_list], torch.float64)
    fidelities = (
        [
            fidelity_confidences[obs.fidelity]
            for obs in obs_list
            if obs.fidelity is not None
        ]
        if fidelity_confidences is not None
        else []
    )

    return X, y, fidelities


def candidates_to_tensor(
    candidates: Iterable[Candidate],
    fidelity_confidences: Optional[dict[int, float]] = None,
) -> tuple[torch.Tensor, list[float]]:
    """Convert an iterable of Candidates to a tensor.

    Tries fast batch conversion for inputs; falls back to element-wise
    stacking for heterogeneous types. The natural shape of the data is
    preserved — no reshaping is applied.

    Parameters
    ----------
    candidates : Iterable[Candidate]
        Candidates to convert. Materialized to a list internally.
    fidelity_confidences : dict[int, float], optional
        Mapping from integer fidelity IDs to continuous confidence values.
        If None, single-fidelity mode is assumed and fidelities returns as an
        empty list. If a mapping is provided, every non-None fidelity must be
        present in it. Raises ``KeyError`` if a fidelity ID in the candidates
        is missing from the mapping.

    Returns
    -------
    X : torch.Tensor
        Input features tensor. Shape is determined by the input data.
    fidelities : list[float]
        Mapped fidelity confidence values; empty if single-fidelity.
    """
    cand_list = candidates if isinstance(candidates, list) else list(candidates)
    X = _to_tensor([cand.x for cand in cand_list], torch.float64)
    if fidelity_confidences is not None:
        fidelities = [
            fidelity_confidences[cand.fidelity]
            for cand in cand_list
            if cand.fidelity is not None
        ]
    else:
        fidelities = []

    return X, fidelities
