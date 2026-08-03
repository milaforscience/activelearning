"""Exact grid sampler for bounded hypercubes."""

import random
from typing import Callable, Iterable, Optional, Sequence

import torch

from activelearning.acquisition.acquisition import Acquisition
from activelearning.acquisition.cost_utility import cost_weighting_from_cost_fn
from activelearning.sampler.sampler import Sampler
from activelearning.utils.types import Candidate, DEFAULT_FIDELITY, Observation
from activelearning.utils.warnings import warn_ignored_args


class ExactGridSampler(Sampler):
    """Enumerate or subsample a Cartesian grid over a bounded hypercube.

    Parameters
    ----------
    bounds : Sequence[tuple[float, float]]
        Per-dimension ``(lower, upper)`` bounds. Each pair must satisfy
        ``lower < upper``.
    points_per_dimension : Sequence[int]
        Number of evenly spaced grid points to generate along each dimension.
        Must contain one positive entry per bound dimension.
    fidelities : Sequence[int] or dict[int, float]
        Fidelity levels to stamp onto the grid points. When a dict is passed,
        only its keys are used for pool expansion. Defaults to the single
        fidelity :data:`~activelearning.utils.types.DEFAULT_FIDELITY`.
    num_samples : int, optional
        Optional number of candidates to subsample from the full grid pool.
        When ``None``, the full pool is returned in deterministic order.
    use_acquisition_scores : bool, default=False
        If ``True``, subsample candidates in proportion to acquisition scores
        via :func:`torch.multinomial`. This exactly samples the discrete
        terminal distribution of an optimally trained GFlowNet policy using
        the same acquisition-derived rewards. Otherwise, subsample uniformly.
    with_replacement : bool, default=False
        Whether subsampling is performed with replacement.

    Raises
    ------
    ValueError
        If bounds are empty or invalid, ``points_per_dimension`` has the wrong
        length or non-positive values, ``fidelities`` is empty, or the
        sampling-related options are inconsistent.
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        points_per_dimension: Sequence[int],
        fidelities: Sequence[int] | dict[int, float] = (DEFAULT_FIDELITY,),
        num_samples: Optional[int] = None,
        use_acquisition_scores: bool = False,
        with_replacement: bool = False,
    ) -> None:
        if len(bounds) == 0:
            raise ValueError("bounds must not be empty")
        for idx, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValueError(f"Bound {idx} has lower >= upper: ({lower}, {upper})")
        if len(points_per_dimension) != len(bounds):
            raise ValueError(
                "points_per_dimension must have one entry per bound dimension"
            )
        if any(points <= 0 for points in points_per_dimension):
            raise ValueError("points_per_dimension entries must all be > 0")
        if num_samples is not None and num_samples <= 0:
            raise ValueError("num_samples must be > 0 when specified")
        if use_acquisition_scores and num_samples is None:
            raise ValueError("use_acquisition_scores requires num_samples")

        self.bounds = list(bounds)
        self.points_per_dimension = list(points_per_dimension)
        self.requested_num_samples = num_samples
        self.use_acquisition_scores = use_acquisition_scores
        self.with_replacement = with_replacement
        self.active_learning_round = 0

        if isinstance(fidelities, dict):
            self._fidelity_levels = sorted(fidelities)
        else:
            self._fidelity_levels = list(fidelities)
        if not self._fidelity_levels:
            raise ValueError("fidelities must not be empty")

    def _get_sampling_seed(self) -> int:
        """Return the deterministic subsampling seed for the current round."""
        return self.runtime_context.seed + self.active_learning_round

    def _get_rng(self) -> random.Random:
        """Return a round-seeded Python RNG for uniform pool subsampling."""
        return random.Random(self._get_sampling_seed())

    def _get_torch_generator(self) -> torch.Generator:
        """Return a round-seeded CPU torch generator for weighted sampling."""
        generator = torch.Generator()
        generator.manual_seed(self._get_sampling_seed())
        return generator

    def _generate_grid_points(self) -> torch.Tensor:
        """Return the full Cartesian grid as a tensor of shape ``(N, d)``."""
        coordinates = [
            torch.linspace(lower, upper, steps=n_points, dtype=self.dtype)
            for (lower, upper), n_points in zip(self.bounds, self.points_per_dimension)
        ]
        if len(coordinates) == 1:
            return coordinates[0].unsqueeze(-1)
        return torch.cartesian_prod(*coordinates)

    def _build_candidate_pool(self) -> list[Candidate]:
        """Return the deterministic grid expanded across fidelity levels."""
        points = self._generate_grid_points()
        return [
            Candidate(x=point.tolist(), fidelity=fidelity)
            for point in points
            for fidelity in self._fidelity_levels
        ]

    def _sample_uniform_candidates(
        self, candidate_pool: list[Candidate], num_samples: int
    ) -> list[Candidate]:
        """Uniformly subsample from the candidate pool."""
        rng = self._get_rng()
        if self.with_replacement:
            return rng.choices(candidate_pool, k=num_samples)
        return rng.sample(candidate_pool, k=num_samples)

    def _sample_scored_candidates(
        self,
        candidate_pool: list[Candidate],
        num_samples: int,
        acquisition: Optional[Acquisition],
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> list[Candidate]:
        """Sample from the exact acquisition-derived GFlowNet target distribution."""
        if acquisition is None:
            raise ValueError(
                "Acquisition function is required when use_acquisition_scores=True."
            )

        if cost_fn is None:
            score_values = acquisition.score(candidate_pool)
        else:
            score_values = acquisition.score(
                candidate_pool,
                cost_weighting=cost_weighting_from_cost_fn(cost_fn),
            )

        scores = torch.tensor(score_values, dtype=self.dtype)
        if scores.numel() != len(candidate_pool):
            raise ValueError("Acquisition returned a score count that does not match.")
        if not torch.isfinite(scores).all():
            raise ValueError("Acquisition scores must be finite for weighted sampling.")
        if (scores < 0).any():
            raise ValueError(
                "Acquisition scores must be non-negative for weighted sampling."
            )
        if scores.sum() <= 0:
            raise ValueError(
                "Acquisition scores must contain at least one positive value for "
                "weighted sampling."
            )

        sampled_indices = torch.multinomial(
            scores,
            num_samples=num_samples,
            replacement=self.with_replacement,
            generator=self._get_torch_generator(),
        ).tolist()
        return [candidate_pool[idx] for idx in sampled_indices]

    def sample(
        self,
        acquisition: Optional[Acquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> list[Candidate]:
        """Enumerate or subsample the configured grid.

        Parameters
        ----------
        acquisition : Optional[Acquisition]
            Acquisition function used only when
            ``use_acquisition_scores=True``.
        observations : Optional[Iterable[Observation]]
            Unused. Present for interface compatibility.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Optional per-candidate cost function used to reweight acquisition
            scores before weighted subsampling.

        Returns
        -------
        result : list[Candidate]
            Candidate pool in deterministic grid order when ``num_samples`` is
            omitted, otherwise the requested uniform or score-weighted
            subsample.
        """
        ignored_kwargs = {"observations": observations}
        if self.requested_num_samples is None or not self.use_acquisition_scores:
            ignored_kwargs["acquisition"] = acquisition
            ignored_kwargs["cost_fn"] = cost_fn
        warn_ignored_args(self, **ignored_kwargs)

        candidate_pool = self._build_candidate_pool()

        if self.requested_num_samples is None:
            return candidate_pool

        if not self.with_replacement and self.requested_num_samples > len(
            candidate_pool
        ):
            raise ValueError(
                "num_samples cannot exceed the candidate pool size when "
                "with_replacement=False"
            )

        if self.use_acquisition_scores:
            return self._sample_scored_candidates(
                candidate_pool,
                num_samples=self.requested_num_samples,
                acquisition=acquisition,
                cost_fn=cost_fn,
            )

        return self._sample_uniform_candidates(
            candidate_pool,
            num_samples=self.requested_num_samples,
        )
