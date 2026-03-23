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
        Observations to convert.
    fidelity_confidences : dict[int, float], optional
        Mapping from integer fidelity IDs to continuous confidence values.
        If None, observations must not include fidelity values. If a mapping is
        provided, every non-None fidelity must be present in it.

    Returns
    -------
    X : torch.Tensor
        Input features tensor. Shape is determined by the input data.
    y : torch.Tensor
        Output labels tensor. Shape is determined by the input data.
    fidelities : list[float]
        Mapped fidelity confidence values; empty if single-fidelity.

    Raises
    ------
    ValueError
        If fidelity values are present in observations but ``fidelity_confidences``
        is ``None``.
    KeyError
        If a fidelity ID in observations is missing from ``fidelity_confidences``.
    """
    xs: list = []
    ys: list = []
    fidelities: list[float] = []
    any_fidelity = False

    for obs in observations:
        xs.append(obs.x)
        ys.append(obs.y)
        if obs.fidelity is not None:
            any_fidelity = True
            if fidelity_confidences is not None:
                fidelities.append(fidelity_confidences[obs.fidelity])

    if fidelity_confidences is None and any_fidelity:
        raise ValueError(
            "Observations include fidelity values, but no fidelity_confidences "
            "mapping was provided."
        )

    X = _to_tensor(xs, torch.float64)
    y = _to_tensor(ys, torch.float64)
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
        Candidates to convert.
    fidelity_confidences : dict[int, float], optional
        Mapping from integer fidelity IDs to continuous confidence values.
        If None, candidates must not include fidelity values. If a mapping is
        provided, every non-None fidelity must be present in it.

    Returns
    -------
    X : torch.Tensor
        Input features tensor. Shape is determined by the input data.
    fidelities : list[float]
        Mapped fidelity confidence values; empty if single-fidelity.

    Raises
    ------
    ValueError
        If fidelity values are present in candidates but ``fidelity_confidences``
        is ``None``.
    KeyError
        If a fidelity ID in candidates is missing from ``fidelity_confidences``.
    """
    xs: list = []
    fidelities: list[float] = []
    any_fidelity = False

    for cand in candidates:
        xs.append(cand.x)
        if cand.fidelity is not None:
            any_fidelity = True
            if fidelity_confidences is not None:
                fidelities.append(fidelity_confidences[cand.fidelity])

    if fidelity_confidences is None and any_fidelity:
        raise ValueError(
            "Candidates include fidelity values, but no fidelity_confidences "
            "mapping was provided."
        )

    X = _to_tensor(xs, torch.float64)
    return X, fidelities
