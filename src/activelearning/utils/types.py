from dataclasses import dataclass
from collections.abc import Hashable, Mapping
import math
from typing import Any, Iterable, Optional

import torch

#: Default fidelity level when none is specified, including single-fidelity runs.
DEFAULT_FIDELITY: int = 0


@dataclass(frozen=True)
class Candidate:
    """Represents a candidate item to be evaluated or sampled.

    Uses maximum type flexibility to support various data representations.

    Single-fidelity experiments use a single fidelity level (typically the
    oracle's only declared level).

    Attributes
    ----------
    x : Any
        Input feature or identifier. Commonly: primitives (int, float, str),
        arrays (numpy.ndarray), tensors (torch.Tensor), or structured data (dict, tuple).
    fidelity : int
        Fidelity level at which this candidate is to be evaluated.
        Defaults to :data:`DEFAULT_FIDELITY`.
    metadata : Optional[dict[str, Any]]
        Domain-specific auxiliary data carried alongside the candidate.
        Not consumed by the core AL loop but available to user components
        (e.g., provenance tracking, auxiliary scores, or domain-specific
        annotations).
    """

    x: Any
    fidelity: int = DEFAULT_FIDELITY
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class Observation:
    """Represents an observed (x, y) pair at a fidelity level.

    Uses maximum type flexibility to support various data representations.

    Attributes
    ----------
    x : Any
        Input feature or identifier. Same semantics as Candidate.x.
    y : Any
        Observed output or label. Commonly: scalar (float), vector (list, array),
        or categorical label (str, int).
    fidelity : int
        Fidelity level at which the observation was made.
        Defaults to :data:`DEFAULT_FIDELITY`.
    metadata : Optional[dict[str, Any]]
        Domain-specific auxiliary data carried alongside the observation.
        See :class:`Candidate` for usage details.
    """

    x: Any
    y: Any
    fidelity: int = DEFAULT_FIDELITY
    metadata: Optional[dict[str, Any]] = None


_UNSUPPORTED_IDENTITY = object()


def candidate_identity(
    candidate_or_observation: Candidate | Observation,
) -> Hashable | None:
    """Return a deterministic identity for a candidate or observation.

    Primitive values and nested list, tuple, mapping, NumPy, and tensor values
    are supported. Domain-specific objects return ``None`` so generic
    diagnostics do not impose a representation on them.
    """
    canonical = _canonical_identity(candidate_or_observation.x)
    if canonical is _UNSUPPORTED_IDENTITY:
        return None
    return candidate_or_observation.fidelity, canonical


def candidate_inputs_match(
    candidate: Candidate,
    observation: Observation,
) -> bool | None:
    """Compare candidate and observation inputs when their identities are known."""
    candidate_key = candidate_identity(candidate)
    observation_key = candidate_identity(observation)
    if candidate_key is None or observation_key is None:
        return None
    return candidate_key == observation_key


def _canonical_identity(value: Any) -> Hashable | object:
    """Convert a supported input value into a tagged hashable structure."""
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return _UNSUPPORTED_IDENTITY
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)

    if hasattr(value, "detach") and callable(value.detach):
        try:
            value = value.detach().cpu().tolist()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return _UNSUPPORTED_IDENTITY
    elif hasattr(value, "tolist") and callable(value.tolist):
        try:
            converted_value = value.tolist()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return _UNSUPPORTED_IDENTITY
        if converted_value is value:
            return _UNSUPPORTED_IDENTITY
        return _canonical_identity(converted_value)

    if isinstance(value, Mapping):
        items: list[tuple[Hashable, Hashable]] = []
        for key, item in value.items():
            canonical_key = _canonical_identity(key)
            canonical_item = _canonical_identity(item)
            if (
                canonical_key is _UNSUPPORTED_IDENTITY
                or canonical_item is _UNSUPPORTED_IDENTITY
            ):
                return _UNSUPPORTED_IDENTITY
            items.append((canonical_key, canonical_item))
        return ("mapping", tuple(sorted(items, key=repr)))
    if isinstance(value, list):
        items = [_canonical_identity(item) for item in value]
        if any(item is _UNSUPPORTED_IDENTITY for item in items):
            return _UNSUPPORTED_IDENTITY
        return ("sequence", tuple(items))
    if isinstance(value, tuple):
        items = [_canonical_identity(item) for item in value]
        if any(item is _UNSUPPORTED_IDENTITY for item in items):
            return _UNSUPPORTED_IDENTITY
        return ("sequence", tuple(items))
    return _UNSUPPORTED_IDENTITY


def has_finite_target(observation: Observation) -> bool:
    """Return True if the observation has a usable label for model training.

    Returns False when:

    - ``y`` is ``None`` (explicit absence of a label, e.g. a failed oracle evaluation)
    - ``y`` is a numeric scalar, list, or array that contains NaN or infinite values

    Non-numeric targets (strings, dicts, etc.) cannot be checked for finiteness
    and are treated as valid (``True``).

    Parameters
    ----------
    observation : Observation
        The observation to check.

    Returns
    -------
    bool
        ``True`` if the target is usable, ``False`` if it should be excluded
        from model training.
    """
    value = observation.y
    if value is None:
        return False
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
        return bool(torch.isfinite(tensor).all())
    except (TypeError, ValueError, RuntimeError):
        return True


def filter_finite_target_observations(
    observations: Iterable[Observation],
) -> list[Observation]:
    """Return only observations with finite numeric or non-numeric targets.

    Drops observations where ``y`` is ``None`` or a numeric value containing NaN
    or infinite entries. Non-numeric targets (strings, dicts, etc.) pass through
    unchanged.

    Parameters
    ----------
    observations : Iterable[Observation]
        Iterable of observations to filter.

    Returns
    -------
    list[Observation]
        Filtered list containing only valid observations.
    """
    return [obs for obs in observations if has_finite_target(obs)]


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

    Raises
    ------
    ValueError
        If ``candidates`` and ``labels`` have different lengths.
    """
    candidates_list = candidates if isinstance(candidates, list) else list(candidates)
    labels_list = labels if isinstance(labels, list) else list(labels)
    if len(candidates_list) != len(labels_list):
        raise ValueError("Length of candidates and labels must match.")
    return [
        Observation(
            x=candidate.x,
            y=label,
            fidelity=candidate.fidelity,
            metadata=candidate.metadata,
        )
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
        When provided (multi-fidelity mode), each observation's fidelity is
        looked up in this mapping and the resulting confidence is appended to
        the ``fidelities`` return list.  When ``None`` (single-fidelity mode),
        fidelity values are ignored and an empty list is returned.

    Returns
    -------
    X : torch.Tensor
        Input features tensor. Shape is determined by the input data.
    y : torch.Tensor
        Output labels tensor. Shape is determined by the input data.
    fidelities : list[float]
        Mapped fidelity confidence values; empty in single-fidelity mode.

    Raises
    ------
    KeyError
        If a fidelity ID in observations is missing from ``fidelity_confidences``.
    """
    xs: list = []
    ys: list = []
    fidelities: list[float] = []

    for obs in observations:
        xs.append(obs.x)
        ys.append(obs.y)
        if fidelity_confidences is not None:
            fidelities.append(fidelity_confidences[obs.fidelity])

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
        When provided (multi-fidelity mode), each candidate's fidelity is
        looked up in this mapping and the resulting confidence is appended to
        the ``fidelities`` return list.  When ``None`` (single-fidelity mode),
        fidelity values are ignored and an empty list is returned.

    Returns
    -------
    X : torch.Tensor
        Input features tensor. Shape is determined by the input data.
    fidelities : list[float]
        Mapped fidelity confidence values; empty in single-fidelity mode.

    Raises
    ------
    KeyError
        If a fidelity ID in candidates is missing from ``fidelity_confidences``.
    """
    xs: list = []
    fidelities: list[float] = []

    for cand in candidates:
        xs.append(cand.x)
        if fidelity_confidences is not None:
            fidelities.append(fidelity_confidences[cand.fidelity])

    X = _to_tensor(xs, torch.float64)
    return X, fidelities
