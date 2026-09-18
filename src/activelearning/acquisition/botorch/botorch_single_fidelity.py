"""Concrete wrappers for single-fidelity BoTorch acquisition functions."""

from typing import Any, Iterable, Optional

import torch
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy as _qLowerBoundMaxValueEntropy,
)

from activelearning.acquisition.botorch.botorch_acquisition import (
    QBatchBoTorchAcquisition,
)
from activelearning.acquisition.botorch.candidate_set import CandidateSetSpec
from activelearning.runtime import RuntimeContext
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Observation


class QLowerBoundMaxValueEntropy(QBatchBoTorchAcquisition):
    """Single-fidelity lower-bound q-Max-Value Entropy Search.

    This wrapper builds BoTorch's GIBBON acquisition from a configurable
    candidate set and exposes q-batch scoring through the shared singleton
    scoring path. Information gain is theoretically non-negative, so scores
    are clamped to zero to avoid invalid downstream sampling weights.

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification for constructing the discrete candidate set used to
        approximate the distribution of the maximum objective value.
    num_mv_samples : int, default=10
        Number of maximum-value samples used by BoTorch.
    maximize : bool, default=True
        Whether the objective is maximized.
    """

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_mv_samples: int = 10,
        maximize: bool = True,
    ) -> None:
        if num_mv_samples <= 0:
            raise ValueError(f"num_mv_samples must be > 0, got {num_mv_samples}")

        super().__init__(maximize=maximize)
        self._candidate_set_spec = candidate_set_spec
        self._num_mv_samples = num_mv_samples

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Bind runtime settings to this acquisition and its candidate set."""
        super().bind_runtime_context(runtime_context)
        self._candidate_set_spec.bind_runtime_context(runtime_context)

    def update(
        self,
        surrogate: Surrogate,
        observations: Optional[Iterable[Observation]] = None,
    ) -> None:
        """Refresh the candidate set and rebuild the BoTorch acquisition."""
        if observations is not None:
            obs_list = list(observations)
            self._candidate_set_spec.update(obs_list)
            super().update(surrogate, obs_list)
        else:
            super().update(surrogate, observations)

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qLowerBoundMaxValueEntropy object."""
        if self._botorch_surrogate is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not updated with surrogate before "
                "building acquisition."
            )

        target_fidelity_value = (
            self._resolved_target_fidelity_value
            if self._botorch_surrogate.is_multi_fidelity
            else None
        )
        return _qLowerBoundMaxValueEntropy(
            model=self._botorch_surrogate.get_model(),
            candidate_set=self._candidate_set_spec.build(
                self._botorch_surrogate,
                target_fidelity_value=target_fidelity_value,
            ),
            num_mv_samples=self._num_mv_samples,
            maximize=self.maximize,
        )

    def _score_encoded(self, X: torch.Tensor) -> list[float]:
        """Evaluate GIBBON and clamp negative numerical artifacts to zero."""
        scores = super()._score_encoded(X)
        return [max(0.0, score) for score in scores]
