"""Candidate-set specifications for entropy-based acquisition functions.

Some acquisition functions — such as Max-Value Entropy Search (MES) variants —
require a discrete set of candidate points to approximate the distribution of
the optimum.  The quality of this approximation depends on how well the
candidate set covers the search space.

A :class:`CandidateSetSpec` encapsulates the logic for constructing this set.
Subclasses are instantiated with the parameters that describe the desired
coverage strategy; the actual candidate tensor is produced by calling
:meth:`~CandidateSetSpec.build` once a fitted surrogate is available.

Three built-in strategies are provided:

- :class:`HypercubeCandidateSetSpec` — for **continuous** domains. Generates
  points by sampling from a user-specified bounded hypercube (uniform or Latin
  Hypercube Sampling). Handles multi-fidelity problems automatically by
  appending the target fidelity to each candidate.
- :class:`TrainDataCandidateSetSpec` — a simple default for **discrete**
  domains. Reuses the surrogate's training inputs as the support set.
- :class:`TensorCandidateSetSpec` — for users who wish to supply their own
  precomputed candidate set directly.

Custom strategies can be implemented by subclassing :class:`CandidateSetSpec`
and overriding :meth:`~CandidateSetSpec.build`.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch

from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate


class CandidateSetSpec(ABC):
    """Base class for candidate-set construction strategies.

    A :class:`CandidateSetSpec` defines how to produce the discrete set of
    points used by entropy-based acquisition functions to approximate the
    distribution of the optimum.  The candidate set is constructed by calling
    :meth:`build` after the surrogate has been fitted.

    The ``target_fidelity_value`` parameter is provided by the calling
    acquisition and represents the already-resolved encoded fidelity value
    (e.g. ``1.0`` for the highest-confidence fidelity).  Passing it explicitly
    ensures the candidate set is always consistent with the acquisition's own
    target-fidelity resolution — including any user-specified overrides.

    The returned tensor is in **model space** — the feature representation
    expected by the surrogate's internal model — with shape ``(N, d)``, where
    ``d`` includes any fidelity column in multi-fidelity settings.

    Implement this class to define custom candidate-set strategies.
    """

    @abstractmethod
    def build(
        self,
        surrogate: BoTorchGPSurrogate,
        *,
        target_fidelity_value: Optional[float] = None,
    ) -> torch.Tensor:
        """Materialize the candidate set tensor.

        Parameters
        ----------
        surrogate : BoTorchGPSurrogate
            A fitted surrogate.
        target_fidelity_value : float, optional
            Encoded target fidelity value, passed by the acquisition after its
            own resolution step.  When provided, implementations should use
            this value to populate the fidelity column rather than inferring
            it independently.

        Returns
        -------
        candidate_set : torch.Tensor
            Model-space candidate tensor of shape ``(N, d)``.
        """


class HypercubeCandidateSetSpec(CandidateSetSpec):
    """Build a candidate set by sampling from a bounded hypercube.

    Generates points by uniform or Latin Hypercube Sampling within the
    specified feature bounds.  Suitable for **continuous** domains.

    In multi-fidelity mode the target fidelity value is appended as the last
    column of the returned tensor.  This value is provided by the calling
    acquisition at :meth:`build` time, ensuring it is always consistent with
    the acquisition's own target-fidelity resolution.

    Parameters
    ----------
    bounds : list[tuple[float, float]]
        Per-dimension ``(lower, upper)`` bounds for the **feature** dimensions.
        Do not include the fidelity dimension — it is appended automatically
        in multi-fidelity mode.
    n_points : int
        Number of candidate points to generate.  Must be > 0.
    strategy : {"uniform", "lhs"}, default="uniform"
        Sampling strategy.  ``"uniform"`` draws i.i.d. uniform samples;
        ``"lhs"`` uses Latin Hypercube Sampling for better space coverage.

    Raises
    ------
    ValueError
        If ``bounds`` is empty, any bound is invalid, ``n_points <= 0``, or
        ``strategy`` is unrecognised.
    """

    def __init__(
        self,
        *,
        bounds: list[tuple[float, float]],
        n_points: int,
        strategy: Literal["uniform", "lhs"] = "uniform",
    ) -> None:
        if len(bounds) == 0:
            raise ValueError("bounds must not be empty")
        if n_points <= 0:
            raise ValueError(f"n_points must be > 0, got {n_points}")
        for idx, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValueError(f"Bound {idx} has lower >= upper: ({lower}, {upper})")
        if strategy not in ("uniform", "lhs"):
            raise ValueError(f"strategy must be 'uniform' or 'lhs', got {strategy!r}")

        self.bounds = bounds
        self.n_points = n_points
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_unit(self) -> torch.Tensor:
        """Return ``(n_points, n_dims)`` samples in ``[0, 1]^d``."""
        n_dims = len(self.bounds)
        if self.strategy == "lhs":
            return self._latin_hypercube(self.n_points, n_dims)
        return torch.rand(self.n_points, n_dims, dtype=torch.float64)

    @staticmethod
    def _latin_hypercube(n_points: int, n_dims: int) -> torch.Tensor:
        """Generate an LHS design in ``[0, 1]^d``."""
        offsets = torch.rand(n_points, n_dims, dtype=torch.float64)
        perms = torch.stack([torch.randperm(n_points) for _ in range(n_dims)], dim=1)
        return (perms.to(torch.float64) + offsets) / n_points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        surrogate: BoTorchGPSurrogate,
        *,
        target_fidelity_value: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample from the hypercube and return a model-space tensor.

        Parameters
        ----------
        surrogate : BoTorchGPSurrogate
            Accepted for interface compatibility; not used directly.
        target_fidelity_value : float, optional
            When provided, appended as the fidelity column of every candidate
            in the returned tensor.  Should be the encoded confidence value
            (e.g. ``1.0``) as resolved by the calling acquisition.

        Returns
        -------
        candidate_set : torch.Tensor
            Tensor of shape ``(n_points, d)`` or ``(n_points, d + 1)`` when
            ``target_fidelity_value`` is provided.
        """
        lowers = torch.tensor([lo for lo, _ in self.bounds], dtype=torch.float64)
        ranges = torch.tensor([hi - lo for lo, hi in self.bounds], dtype=torch.float64)
        feature_points = lowers + self._sample_unit() * ranges  # (N, d)

        if target_fidelity_value is not None:
            fid_col = torch.full(
                (self.n_points, 1), target_fidelity_value, dtype=torch.float64
            )
            return torch.cat([feature_points, fid_col], dim=-1)

        return feature_points


class TrainDataCandidateSetSpec(CandidateSetSpec):
    """Build a candidate set from the surrogate's training inputs.

    The candidate set is taken directly from ``surrogate.get_train_data()[0]``
    (``train_X`` in model space).  This is a convenient default for
    **discrete** domains where the observed inputs form a natural support set.

    .. note::
       In discrete settings the f* samples drawn from this candidate set will
       be anchored to the observed data points, which can introduce bias —
       particularly when the observed set is small or unrepresentative of the
       full domain.  This is an acceptable library default, but users with
       large or structured discrete spaces should consider
       :class:`TensorCandidateSetSpec` or a custom subclass.
    """

    def build(
        self,
        surrogate: BoTorchGPSurrogate,
        *,
        target_fidelity_value: Optional[float] = None,
    ) -> torch.Tensor:
        """Return the surrogate's training inputs as the candidate set.

        Parameters
        ----------
        surrogate : BoTorchGPSurrogate
            Fitted surrogate.
        target_fidelity_value : float, optional
            Accepted for interface compatibility; not used.  The training
            inputs already include the encoded fidelity column when the
            surrogate is in multi-fidelity mode.

        Returns
        -------
        candidate_set : torch.Tensor
            ``train_X`` tensor of shape ``(N, d)`` already in model space.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted yet.
        """
        train_X, _ = surrogate.get_train_data()
        return train_X


class TensorCandidateSetSpec(CandidateSetSpec):
    """Wrap a precomputed tensor as a :class:`CandidateSetSpec`.

    This is an escape hatch for advanced users who already have a model-space
    tensor (e.g. from precomputation, a pool dataset, or a custom grid) and
    want to remain within the spec API without subclassing.

    Parameters
    ----------
    tensor : torch.Tensor
        Precomputed candidate set in model space.  Shape ``(N, d)``.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def build(
        self,
        surrogate: BoTorchGPSurrogate,
        *,
        target_fidelity_value: Optional[float] = None,
    ) -> torch.Tensor:
        """Return the stored tensor unchanged.

        Parameters
        ----------
        surrogate : BoTorchGPSurrogate
            Accepted for interface compatibility; not used.
        target_fidelity_value : float, optional
            Accepted for interface compatibility; not used.

        Returns
        -------
        candidate_set : torch.Tensor
            The tensor passed at construction time.
        """
        return self.tensor
