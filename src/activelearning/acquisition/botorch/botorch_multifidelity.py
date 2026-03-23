"""Concrete wrappers for BoTorch multi-fidelity acquisition functions.

All multi-fidelity acquisitions in BoTorch are q-batch / Monte Carlo.
Each class subclasses :class:`QBatchBoTorchAcquisition` and implements
:meth:`_build_botorch_acquisition`, wiring the resolved cost-aware utility
and target-fidelity projection from the base class into the BoTorch object.
"""

from typing import Any, Callable, Iterable, Optional

import torch
from botorch.acquisition.knowledge_gradient import (
    qMultiFidelityKnowledgeGradient as _qMFKG,
)
from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy as _qMFLBMES,
    qMultiFidelityMaxValueEntropy as _qMFMES,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from activelearning.acquisition.botorch.botorch_acquisition import (
    QBatchBoTorchAcquisition,
)
from activelearning.acquisition.candidate_set import CandidateSetSpec
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Observation


class QMultiFidelityMaxValueEntropy(QBatchBoTorchAcquisition):
    """Multi-fidelity q-Max-Value Entropy Search (qMFMES).

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification describing how to build the discrete candidate set used
        to approximate the max-value distribution.  The tensor is materialized
        at acquisition-build time when the fitted surrogate is available.
        Use :class:`~activelearning.acquisition.candidate_set.HypercubeCandidateSetSpec`
        for continuous domains,
        :class:`~activelearning.acquisition.candidate_set.TrainDataCandidateSetSpec`
        as a discrete default, or
        :class:`~activelearning.acquisition.candidate_set.TensorCandidateSetSpec`
        to pass a precomputed tensor directly.
    num_fantasies : int, default=16
        Number of fantasy models.
    num_mv_samples : int, default=10
        Number of max-value samples.
    num_y_samples : int, default=128
        Number of outcome samples per max-value sample.
    expand : callable, optional
        Callable for trace-observation expansion.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        expand: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        if num_fantasies <= 0:
            raise ValueError(f"num_fantasies must be > 0, got {num_fantasies}")
        if num_mv_samples <= 0:
            raise ValueError(f"num_mv_samples must be > 0, got {num_mv_samples}")
        if num_y_samples <= 0:
            raise ValueError(f"num_y_samples must be > 0, got {num_y_samples}")

        super().__init__(**kwargs)
        self._candidate_set_spec = candidate_set_spec
        self._num_fantasies = num_fantasies
        self._num_mv_samples = num_mv_samples
        self._num_y_samples = num_y_samples
        self._expand = expand

    def update(
        self,
        surrogate: Surrogate,
        observations: Optional[Iterable[Observation]] = None,
    ) -> None:
        """Update candidate set spec with current observations, then update base.

        Parameters
        ----------
        surrogate : Surrogate
            The fitted surrogate.
        observations : Iterable[Observation], optional
            Current observations forwarded to the candidate set spec and base.
        """
        if observations is not None:
            obs_list = list(observations)
            self._candidate_set_spec.update(obs_list)
            super().update(surrogate, obs_list)
        else:
            super().update(surrogate, observations)

    def _build_botorch_acquisition(self) -> Any:
        if self._botorch_surrogate is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not updated with surrogate before building acquisition."
            )

        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "candidate_set": self._candidate_set_spec.build(
                self._botorch_surrogate,
                target_fidelity_value=self._resolved_target_fidelity_value,
            ),
            "num_fantasies": self._num_fantasies,
            "num_mv_samples": self._num_mv_samples,
            "num_y_samples": self._num_y_samples,
            "maximize": self.maximize,
        }

        if self._resolved_cost_aware_utility is not None:
            build_kwargs["cost_aware_utility"] = self._resolved_cost_aware_utility
        if self._resolved_project_to_target_fidelity_fn is not None:
            build_kwargs["project"] = self._resolved_project_to_target_fidelity_fn
        if self._expand is not None:
            build_kwargs["expand"] = self._expand

        return _qMFMES(**build_kwargs)


class QMultiFidelityLowerBoundMaxValueEntropy(QBatchBoTorchAcquisition):
    """Multi-fidelity lower-bound q-Max-Value Entropy Search (qMFLBMES).

    A cheaper approximation of :class:`QMultiFidelityMaxValueEntropy`.

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification describing how to build the discrete candidate set for
        max-value approximation.  The tensor is materialized at
        acquisition-build time when the fitted surrogate is available.
        Use :class:`~activelearning.acquisition.candidate_set.HypercubeCandidateSetSpec`
        for continuous domains,
        :class:`~activelearning.acquisition.candidate_set.TrainDataCandidateSetSpec`
        as a discrete default, or
        :class:`~activelearning.acquisition.candidate_set.TensorCandidateSetSpec`
        to pass a precomputed tensor directly.
    num_fantasies : int, default=16
        Number of fantasy models.
    num_mv_samples : int, default=10
        Number of max-value samples.
    num_y_samples : int, default=128
        Number of outcome samples per max-value sample.
    expand : callable, optional
        Callable for trace-observation expansion.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        expand: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        if num_fantasies <= 0:
            raise ValueError(f"num_fantasies must be > 0, got {num_fantasies}")
        if num_mv_samples <= 0:
            raise ValueError(f"num_mv_samples must be > 0, got {num_mv_samples}")
        if num_y_samples <= 0:
            raise ValueError(f"num_y_samples must be > 0, got {num_y_samples}")

        super().__init__(**kwargs)
        self._candidate_set_spec = candidate_set_spec
        self._num_fantasies = num_fantasies
        self._num_mv_samples = num_mv_samples
        self._num_y_samples = num_y_samples
        self._expand = expand

    def update(
        self,
        surrogate: Surrogate,
        observations: Optional[Iterable[Observation]] = None,
    ) -> None:
        """Update candidate set spec with current observations, then update base.

        Parameters
        ----------
        surrogate : Surrogate
            The fitted surrogate.
        observations : Iterable[Observation], optional
            Current observations forwarded to the candidate set spec and base.
        """
        if observations is not None:
            obs_list = list(observations)
            self._candidate_set_spec.update(obs_list)
            super().update(surrogate, obs_list)
        else:
            super().update(surrogate, observations)

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qMultiFidelityLowerBoundMaxValueEntropy object."""
        if self._botorch_surrogate is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not updated with surrogate before building acquisition."
            )

        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "candidate_set": self._candidate_set_spec.build(
                self._botorch_surrogate,
                target_fidelity_value=self._resolved_target_fidelity_value,
            ),
            "num_fantasies": self._num_fantasies,
            "num_mv_samples": self._num_mv_samples,
            "num_y_samples": self._num_y_samples,
            "maximize": self.maximize,
        }

        if self._resolved_cost_aware_utility is not None:
            build_kwargs["cost_aware_utility"] = self._resolved_cost_aware_utility
        if self._resolved_project_to_target_fidelity_fn is not None:
            build_kwargs["project"] = self._resolved_project_to_target_fidelity_fn
        if self._expand is not None:
            build_kwargs["expand"] = self._expand

        return _qMFLBMES(**build_kwargs)


class QMultiFidelityKnowledgeGradient(QBatchBoTorchAcquisition):
    """Multi-fidelity q-Knowledge Gradient (qMFKG).

    Extends qKG with cost-aware utility, target-fidelity projection, and
    trace-observation expansion. When cost-aware / projection helpers are
    configured on the base class (via constructor kwargs or surrogate
    metadata), they are wired in automatically.

    Parameters
    ----------
    num_fantasies : int, default=64
        Number of fantasy models used for inner optimization.
    current_value : float, optional
        Current best objective value.
    expand : callable, optional
        Callable for trace-observation expansion.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        num_fantasies: int = 64,
        current_value: Optional[float] = None,
        expand: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        if num_fantasies <= 0:
            raise ValueError(f"num_fantasies must be > 0, got {num_fantasies}")

        super().__init__(**kwargs)
        self._num_fantasies = num_fantasies
        self._current_value = current_value
        self._expand = expand

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qMultiFidelityKnowledgeGradient object."""
        if self._botorch_surrogate is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not updated with surrogate before building acquisition."
            )

        current_value = (
            torch.tensor(self._current_value, dtype=torch.float64)
            if self._current_value is not None
            else None
        )

        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "num_fantasies": self._num_fantasies,
            "current_value": current_value,
        }

        if not self.maximize:
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=torch.float64)
            )

        if self._resolved_cost_aware_utility is not None:
            build_kwargs["cost_aware_utility"] = self._resolved_cost_aware_utility
        if self._resolved_project_to_target_fidelity_fn is not None:
            build_kwargs["project"] = self._resolved_project_to_target_fidelity_fn
        if self._expand is not None:
            build_kwargs["expand"] = self._expand

        return _qMFKG(**build_kwargs)
