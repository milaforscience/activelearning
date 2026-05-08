"""Concrete wrappers for BoTorch single-fidelity q-batch acquisitions.

Currently this module contains the single-fidelity max-value entropy variants.
"""

from typing import Any, Iterable, Optional

from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy as _qLBMES,
    qMaxValueEntropy as _qMES,
)

from activelearning.acquisition.botorch.botorch_acquisition import (
    QBatchBoTorchAcquisition,
)
from activelearning.acquisition.botorch.candidate_set import CandidateSetSpec
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Observation


class _QSingleFidelityEntropyBase(QBatchBoTorchAcquisition):
    """Shared base for single-fidelity max-value entropy acquisitions."""

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_mv_samples: int = 10,
        **kwargs: Any,
    ) -> None:
        if num_mv_samples <= 0:
            raise ValueError(f"num_mv_samples must be > 0, got {num_mv_samples}")

        super().__init__(**kwargs)
        self._candidate_set_spec = candidate_set_spec
        self._num_mv_samples = num_mv_samples

    def update(
        self,
        surrogate: Surrogate,
        observations: Optional[Iterable[Observation]] = None,
    ) -> None:
        """Update the candidate set spec, then rebuild the BoTorch acquisition."""
        if observations is not None:
            obs_list = list(observations)
            self._candidate_set_spec.update(obs_list)
            super().update(surrogate, obs_list)
        else:
            super().update(surrogate, observations)

    def _build_common_kwargs(self) -> dict[str, Any]:
        """Return the common constructor kwargs shared by BoTorch MES variants."""
        if self._botorch_surrogate is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not updated with surrogate before building acquisition."
            )

        return {
            "model": self._botorch_surrogate.get_model(),
            "candidate_set": self._candidate_set_spec.build(
                self._botorch_surrogate,
                target_fidelity_value=self._resolved_target_fidelity_value,
            ),
            "num_mv_samples": self._num_mv_samples,
            "maximize": self.maximize,
        }


class QMaxValueEntropy(_QSingleFidelityEntropyBase):
    """Single-fidelity q-Max-Value Entropy Search (qMES)."""

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        **kwargs: Any,
    ) -> None:
        if num_fantasies <= 0:
            raise ValueError(f"num_fantasies must be > 0, got {num_fantasies}")
        if num_y_samples <= 0:
            raise ValueError(f"num_y_samples must be > 0, got {num_y_samples}")

        super().__init__(
            candidate_set_spec=candidate_set_spec,
            num_mv_samples=num_mv_samples,
            **kwargs,
        )
        self._num_fantasies = num_fantasies
        self._num_y_samples = num_y_samples

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qMaxValueEntropy object."""
        build_kwargs = self._build_common_kwargs()
        build_kwargs["num_fantasies"] = self._num_fantasies
        build_kwargs["num_y_samples"] = self._num_y_samples
        return _qMES(**build_kwargs)


class QLowerBoundMaxValueEntropy(_QSingleFidelityEntropyBase):
    """Single-fidelity lower-bound q-Max-Value Entropy Search."""

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_mv_samples: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            candidate_set_spec=candidate_set_spec,
            num_mv_samples=num_mv_samples,
            **kwargs,
        )

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qLowerBoundMaxValueEntropy object."""
        return _qLBMES(**self._build_common_kwargs())
