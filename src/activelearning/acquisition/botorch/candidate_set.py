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
  Hypercube Sampling). The target fidelity column is appended automatically in
  multi-fidelity mode.
- :class:`TrainDataCandidateSetSpec` — a simple default for **discrete**
  domains. Reuses the observations seen so far as the support set.
- :class:`TensorCandidateSetSpec` — for users who want to supply their own
  precomputed candidate set directly.

Custom strategies can be implemented by subclassing :class:`CandidateSetSpec`
and overriding :meth:`~CandidateSetSpec.build`.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Literal, Optional

import torch

from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.utils.sampling import latin_hypercube
from activelearning.utils.types import Candidate, Observation


class CandidateSetSpec(ABC):
    """Base class for candidate-set construction strategies.

    A :class:`CandidateSetSpec` defines how to produce the discrete set of
    points used by entropy-based acquisition functions to approximate the
    distribution of the optimum.

    The returned tensor is in **model space** — the feature representation
    expected by the surrogate's internal model — with shape ``(N, d)``, where
    ``d`` includes any fidelity column in multi-fidelity settings.

    Implement this class to define custom candidate-set strategies.
    """

    def update(self, observations: Iterable[Observation]) -> None:
        """Notify the spec of the current observations.

        Called before :meth:`build` whenever the observation set changes.
        The default implementation is a no-op. Subclasses that build their
        candidate set from observed data (e.g. :class:`TrainDataCandidateSetSpec`)
        should override this to cache the observations.

        Parameters
        ----------
        observations : Iterable[Observation]
            The current set of observations. May be a one-pass iterable;
            implementations should materialise it if multiple passes are needed.
        """

    @abstractmethod
    def build(
        self,
        surrogate: BoTorchGPSurrogate,
        *,
        target_fidelity_value: Optional[float] = None,
    ) -> torch.Tensor:
        """Build the candidate set tensor in BoTorch model space.

        The returned tensor is encoded by the surrogate and ready to be passed
        directly to a BoTorch acquisition function. It has shape ``(N, d)``,
        where ``d`` is the full input dimensionality of the surrogate's model,
        including the fidelity column when operating in multi-fidelity mode.

        Parameters
        ----------
        surrogate : BoTorchGPSurrogate
            A fitted BoTorch surrogate used to encode candidates into model
            space.
        target_fidelity_value : float, optional
            The encoded fidelity value to append as the last column of the
            candidate set in multi-fidelity mode (e.g. ``1.0`` for the
            highest-confidence fidelity). ``None`` in single-fidelity mode.

        Returns
        -------
        candidate_set : torch.Tensor
            BoTorch-ready candidate tensor of shape ``(N, d)``.
        """


class HypercubeCandidateSetSpec(CandidateSetSpec):
    """Build a candidate set by sampling from a bounded hypercube.

    Generates points by uniform or Latin Hypercube Sampling within the
    specified feature bounds.  Suitable for **continuous** domains.

    In multi-fidelity mode, the target fidelity value is appended as the
    last column of the returned tensor.

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
            return latin_hypercube(self.n_points, n_dims)
        return torch.rand(self.n_points, n_dims, dtype=torch.float64)

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
        surrogate : Surrogate
            A fitted surrogate. Not used by this implementation.
        target_fidelity_value : float, optional
            When provided, appended as the fidelity column of every candidate
            in the returned tensor (e.g. ``1.0`` for the highest-confidence
            fidelity).

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
    """Build a candidate set from observed data.

    The candidate set is built from the candidates seen in the observations
    provided via :meth:`update`.  This is a convenient default for
    **discrete** domains where the observed inputs form a natural support set.

    Call :meth:`update` with the current observations before calling
    :meth:`build`.  In the library, acquisitions that hold a
    :class:`TrainDataCandidateSetSpec` do this automatically.

    .. note::
       In discrete settings the f* samples drawn from this candidate set will
       be anchored to the observed data points, which can introduce bias —
       particularly when the observed set is small or unrepresentative of the
       full domain.  This is an acceptable library default, but users with
       large or structured discrete spaces should consider
       :class:`TensorCandidateSetSpec` or a custom subclass.
    """

    def __init__(self) -> None:
        self._cached_candidates: list[Candidate] = []

    def update(self, observations: Iterable[Observation]) -> None:
        """Cache the candidates from the current observations.

        Parameters
        ----------
        observations : Iterable[Observation]
            Current observations. Candidates are extracted and stored for
            use in the next :meth:`build` call.
        """
        self._cached_candidates = [
            Candidate(x=obs.x, fidelity=obs.fidelity) for obs in observations
        ]

    def build(
        self,
        surrogate: BoTorchGPSurrogate,
        *,
        target_fidelity_value: Optional[float] = None,
    ) -> torch.Tensor:
        """Encode the cached observation candidates and return them as a tensor.

        Parameters
        ----------
        surrogate : Surrogate
            A fitted surrogate with an ``encode_candidates()`` method.
        target_fidelity_value : float, optional
            Not used — the encoded candidates already include fidelity
            information from the observations.

        Returns
        -------
        candidate_set : torch.Tensor
            Encoded candidate tensor of shape ``(N, d)`` in model space.

        Raises
        ------
        RuntimeError
            If :meth:`update` has not been called with at least one observation.
        AttributeError
            If the surrogate does not implement ``encode_candidates()``.
        """
        if not self._cached_candidates:
            raise RuntimeError(
                "TrainDataCandidateSetSpec has no cached candidates. "
                "Call update() with observations before build()."
            )
        if not hasattr(surrogate, "encode_candidates"):
            raise AttributeError(
                f"{type(surrogate).__name__} does not implement encode_candidates(). "
                "TrainDataCandidateSetSpec requires a surrogate with this method."
            )
        return surrogate.encode_candidates(self._cached_candidates)  # type: ignore[attr-defined]


class TensorCandidateSetSpec(CandidateSetSpec):
    """Supply a precomputed tensor as a candidate set.

    For users who want to provide their own candidate set directly —
    for example from a precomputed grid, a pool dataset, or a custom
    sampling procedure.

    Parameters
    ----------
    tensor : torch.Tensor
        Candidate set in model space.  Shape ``(N, d)``.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 2:
            raise ValueError(
                f"candidate_set tensor must be 2-D with shape (N, d), "
                f"got shape {tuple(tensor.shape)}"
            )
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
        surrogate : Surrogate
            Not used.
        target_fidelity_value : float, optional
            Not used.

        Returns
        -------
        candidate_set : torch.Tensor
            The tensor passed at construction time.
        """
        return self.tensor
