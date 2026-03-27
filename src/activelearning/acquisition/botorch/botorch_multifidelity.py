"""Concrete wrappers for BoTorch multi-fidelity acquisition functions.

All multi-fidelity acquisitions in BoTorch are q-batch / Monte Carlo.
Each class subclasses :class:`QBatchBoTorchAcquisition` and implements
:meth:`_build_botorch_acquisition`, wiring the resolved cost-aware utility
and target-fidelity projection from the base class into the BoTorch object.
"""

from typing import Any, Callable, ClassVar, Iterable, Optional

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
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


class _QMultiFidelityEntropyBase(QBatchBoTorchAcquisition):
    """Shared base for multi-fidelity max-value entropy acquisition functions.

    Implements the common constructor, ``update()``, and
    ``_build_botorch_acquisition()`` shared by
    :class:`QMultiFidelityMaxValueEntropy` and
    :class:`QMultiFidelityLowerBoundMaxValueEntropy`. Subclasses set
    ``_botorch_acqf_class`` to select the underlying BoTorch implementation.

    This class is not intended to be instantiated directly.

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification describing how to build the discrete candidate set used
        to approximate the max-value distribution.
    num_fantasies : int, default=16
        Number of fantasy models used to approximate the joint entropy.
    num_mv_samples : int, default=10
        Number of samples drawn to approximate the max-value distribution.
    num_y_samples : int, default=128
        Number of outcome samples drawn per max-value sample.
    expand : callable, optional
        Optional callable to expand q-batches with trace observations,
        used for multi-fidelity information gain calculations.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    _botorch_acqf_class: ClassVar[type[AcquisitionFunction]]

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
        """Update the candidate set spec with current observations, then update the base.

        The candidate set (used to approximate the max-value distribution) is
        refreshed each round from the latest observations before the BoTorch
        acquisition object is rebuilt.

        Parameters
        ----------
        surrogate : Surrogate
            The fitted surrogate model for the current round.
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

        if self._cost_aware_utility_override is not None:
            build_kwargs["cost_aware_utility"] = self._cost_aware_utility_override
        if self._resolved_project_to_target_fidelity_fn is not None:
            build_kwargs["project"] = self._resolved_project_to_target_fidelity_fn
        if self._expand is not None:
            build_kwargs["expand"] = self._expand

        return self._botorch_acqf_class(**build_kwargs)


class QMultiFidelityMaxValueEntropy(_QMultiFidelityEntropyBase):
    """Multi-fidelity q-Max-Value Entropy Search (qMFMES).

    Estimates the information gain about the maximum objective value at the
    target fidelity from a batch of candidates queried at possibly lower
    fidelities, using fantasy models to account for multi-fidelity
    correlations.

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

    _botorch_acqf_class = _qMFMES


class QMultiFidelityLowerBoundMaxValueEntropy(_QMultiFidelityEntropyBase):
    """Multi-fidelity lower-bound q-Max-Value Entropy Search (qMFLBMES).

    A computationally cheaper approximation of
    :class:`QMultiFidelityMaxValueEntropy` that uses a lower bound on the
    entropy rather than a Monte Carlo estimate, reducing the number of model
    evaluations required per acquisition step.

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

    _botorch_acqf_class = _qMFLBMES


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

        # qMFKG internally uses fantasy models, which require GPyTorch's
        # prediction_strategy to be initialized. This happens on the first
        # posterior evaluation in eval mode.
        model = self._botorch_surrogate.get_model()
        train_X, _ = self._botorch_surrogate.get_train_data()
        model.eval()
        with torch.no_grad():
            model.posterior(train_X)

        current_value = None
        if self._current_value is not None:
            # When maximize=False the posterior_transform negates the objective,
            # so current_value must also be negated to stay in the same space.
            raw = -self._current_value if not self.maximize else self._current_value
            current_value = torch.tensor(
                raw, dtype=train_X.dtype, device=train_X.device
            )

        build_kwargs: dict[str, Any] = {
            "model": model,
            "num_fantasies": self._num_fantasies,
            "current_value": current_value,
        }

        if not self.maximize:
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=train_X.dtype, device=train_X.device)
            )

        if self._cost_aware_utility_override is not None:
            build_kwargs["cost_aware_utility"] = self._cost_aware_utility_override
        if self._resolved_project_to_target_fidelity_fn is not None:
            build_kwargs["project"] = self._resolved_project_to_target_fidelity_fn
        if self._expand is not None:
            build_kwargs["expand"] = self._expand

        return _qMFKG(**build_kwargs)
