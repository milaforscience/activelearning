from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional

import torch

from activelearning.acquisition.acquisition import Acquisition
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Candidate, Observation


class BoTorchAcquisitionBase(Acquisition, ABC):
    """Base class for BoTorch-backed acquisition functions.

    Provides shared infrastructure for all BoTorch-based acquisitions:
    - Surrogate management and validation
    - Candidate encoding and scoring
    - Multi-fidelity and cost-aware support
    - BoTorch acquisition object lifecycle

    Subclasses implement specific acquisition logic by defining
    ``_build_botorch_acquisition()`` to construct the BoTorch acquisition
    and optionally ``_score_encoded()`` to customize how scores are computed.
    """

    def __init__(
        self,
        *,
        maximize: bool = True,
        target_fidelity_value: Optional[float] = None,
        project_to_target_fidelity_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        fidelity_costs: Optional[dict[int, float]] = None,
        cost_model: Optional[Any] = None,
        cost_aware_utility: Optional[Any] = None,
    ) -> None:
        """Initialize the BoTorch acquisition base.

        Parameters
        ----------
        maximize : bool, default=True
            If True, the acquisition assumes a maximization objective.
        target_fidelity_value : float, optional
            Override for the encoded target fidelity value. If None, inferred
            from the surrogate.
        project_to_target_fidelity_fn : callable, optional
            Callable that projects encoded inputs to the target fidelity.
        fidelity_costs : dict[int, float], optional
            Mapping from fidelity ids to query costs.
        cost_model : object, optional
            Custom BoTorch-compatible cost model.
        cost_aware_utility : object, optional
            Custom BoTorch-compatible cost-aware utility.
        """
        super().__init__()
        self.maximize = maximize

        # User-specified multi-fidelity / cost-aware configuration
        self._target_fidelity_value_override = target_fidelity_value
        self._project_to_target_fidelity_fn_override = project_to_target_fidelity_fn
        self._fidelity_costs = (
            dict(fidelity_costs) if fidelity_costs is not None else None
        )
        self._cost_model_override = cost_model
        self._cost_aware_utility_override = cost_aware_utility

        # Typed runtime state populated during update()
        self._botorch_surrogate: Optional[BoTorchGPSurrogate] = None
        self._observations_cache: Optional[list[Observation]] = None
        self._botorch_acqf: Optional[Any] = None

        # Resolved helpers populated during update()
        self._resolved_target_fidelity_value: Optional[float] = None
        self._resolved_project_to_target_fidelity_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None
        self._resolved_cost_model: Optional[Any] = None
        self._resolved_cost_aware_utility: Optional[Any] = None

    @property
    def botorch_surrogate(self) -> Optional[BoTorchGPSurrogate]:
        """The typed BoTorch surrogate, or ``None`` before ``update()``."""
        return self._botorch_surrogate

    @property
    def botorch_acqf(self) -> Optional[Any]:
        """The internal BoTorch acquisition object, or ``None`` before ``update()``."""
        return self._botorch_acqf

    @property
    def observations_cache(self) -> Optional[list[Observation]]:
        """Materialized observations from the current update cycle, or ``None``."""
        return self._observations_cache

    @property
    def resolved_target_fidelity_value(self) -> Optional[float]:
        """Encoded target fidelity value, or ``None`` when not applicable."""
        return self._resolved_target_fidelity_value

    @property
    def resolved_project_to_target_fidelity_fn(
        self,
    ) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """Target-fidelity projection callable, or ``None`` if not applicable."""
        return self._resolved_project_to_target_fidelity_fn

    @property
    def resolved_cost_model(self) -> Optional[Any]:
        """Resolved cost model, or ``None`` if not used."""
        return self._resolved_cost_model

    @property
    def resolved_cost_aware_utility(self) -> Optional[Any]:
        """Resolved cost-aware utility, or ``None`` if not used."""
        return self._resolved_cost_aware_utility

    def update(
        self,
        surrogate: Surrogate,
        observations: Optional[Iterable[Observation]] = None,
    ) -> None:
        """Refresh the acquisition state after the surrogate has been updated.

        This method validates and stores a typed BoTorch surrogate, processes
        observations when provided, resolves shared multi-fidelity helpers, and
        rebuilds the internal BoTorch acquisition object.

        Parameters
        ----------
        surrogate : Surrogate
            Surrogate model to use for subsequent acquisition scoring. Must be a
            ``BoTorchGPSurrogate``.
        observations : Optional[Iterable[Observation]]
            Optional iterable of current observations.

        Raises
        ------
        TypeError
            If the surrogate is not a ``BoTorchGPSurrogate``.
        """
        super().update(surrogate, observations)

        if not isinstance(surrogate, BoTorchGPSurrogate):
            raise TypeError(
                f"{self.__class__.__name__} requires a BoTorchGPSurrogate, "
                f"but received {type(surrogate).__name__}."
            )
        self._botorch_surrogate = surrogate
        self._observations_cache = (
            list(observations) if observations is not None else None
        )

        # Resolve shared MF / cost-aware helpers before building the acqf.
        self._resolved_target_fidelity_value = self._resolve_target_fidelity_value()
        self._resolved_project_to_target_fidelity_fn = (
            self._resolve_projection_to_target_fidelity()
        )
        self._resolved_cost_model = self._resolve_cost_model()
        self._resolved_cost_aware_utility = self._resolve_cost_aware_utility()

        self._botorch_acqf = self._build_botorch_acquisition()

    def _require_botorch_acqf(self) -> Any:
        """Return the internal BoTorch acquisition object.

        Returns
        -------
        result : Any
            The internal BoTorch acquisition object.

        Raises
        ------
        RuntimeError
            If ``update()`` has not yet been called successfully.
        """
        if self._botorch_acqf is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no BoTorch acquisition object yet. "
                "Call update() after fitting the surrogate before scoring."
            )
        return self._botorch_acqf

    def _resolve_target_fidelity_value(self) -> Optional[float]:
        """Resolve the encoded target fidelity value.

        Resolution order:
        1. User-provided override.
        2. Surrogate-provided default.

        Returns
        -------
        result : Optional[float]
            Encoded target fidelity value, or ``None`` if not applicable.
        """
        if self._target_fidelity_value_override is not None:
            return self._target_fidelity_value_override

        if self._botorch_surrogate is None:
            return None

        return self._botorch_surrogate.get_target_fidelity_value()

    def _resolve_projection_to_target_fidelity(
        self,
    ) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """Resolve the projection callable used for target-fidelity projection.

        Resolution order:
        1. User-provided projection callable.
        2. Default projection built from surrogate fidelity metadata.

        Returns
        -------
        result : Optional[Callable[[torch.Tensor], torch.Tensor]]
            Projection callable, or ``None`` if not applicable.

        Raises
        ------
        RuntimeError
            If projection is requested in multi-fidelity mode but required
            fidelity metadata is unavailable.
        """
        if self._project_to_target_fidelity_fn_override is not None:
            return self._project_to_target_fidelity_fn_override

        if self._botorch_surrogate is None:
            return None
        if not self._botorch_surrogate.is_multi_fidelity():
            return None

        fidelity_dim = self._botorch_surrogate.get_fidelity_dimension()
        target_value = self._resolved_target_fidelity_value

        if fidelity_dim is None:
            raise RuntimeError(
                "Multi-fidelity surrogate reported no fidelity dimension."
            )
        if target_value is None:
            raise RuntimeError(
                "Could not resolve a target fidelity value for multi-fidelity acquisition."
            )

        def project_to_target_fidelity(X: torch.Tensor) -> torch.Tensor:
            """Project encoded model-space inputs to the target fidelity.

            Parameters
            ----------
            X : torch.Tensor
                Encoded model-space input tensor whose last dimension is the
                feature dimension.

            Returns
            -------
            result : torch.Tensor
                Copy of ``X`` with the fidelity dimension set to the target value.
            """
            X_projected = X.clone()
            X_projected[..., fidelity_dim] = target_value
            return X_projected

        return project_to_target_fidelity

    def _resolve_cost_model(self) -> Optional[Any]:
        """Resolve the cost model used by cost-aware acquisitions.

        Resolution order:
        1. User-provided custom cost model.
        2. Library default from fidelity-cost mapping.
        3. No cost model.

        Returns
        -------
        result : Optional[Any]
            Resolved cost model object, or ``None`` if not used.
        """
        if self._cost_model_override is not None:
            return self._cost_model_override

        if self._fidelity_costs is None:
            return None

        return self._build_default_cost_model_from_fidelity_costs(self._fidelity_costs)

    def _resolve_cost_aware_utility(self) -> Optional[Any]:
        """Resolve the cost-aware utility used by cost-aware acquisitions.

        Resolution order:
        1. User-provided custom cost-aware utility.
        2. Utility built from the resolved cost model.
        3. No cost-aware utility.

        Returns
        -------
        result : Optional[Any]
            Resolved cost-aware utility object, or ``None`` if not used.
        """
        if self._cost_aware_utility_override is not None:
            return self._cost_aware_utility_override

        if self._resolved_cost_model is None:
            return None

        return self._build_default_cost_aware_utility(self._resolved_cost_model)

    def _build_default_cost_model_from_fidelity_costs(
        self,
        fidelity_costs: dict[int, float],
    ) -> Any:
        """Build a default cost model from a fidelity-cost mapping.

        Parameters
        ----------
        fidelity_costs : dict[int, float]
            Mapping from integer fidelity ids to query costs.

        Returns
        -------
        result : Any
            BoTorch-compatible cost model.

        Raises
        ------
        NotImplementedError
            Default fidelity-cost-based cost modeling has not yet been
            implemented in the base class.
        """
        raise NotImplementedError(
            "Default cost-model construction from fidelity costs is not implemented yet."
        )

    def _build_default_cost_aware_utility(self, cost_model: Any) -> Any:
        """Build a default cost-aware utility from a resolved cost model.

        Parameters
        ----------
        cost_model : Any
            Resolved BoTorch-compatible cost model.

        Returns
        -------
        result : Any
            BoTorch-compatible cost-aware utility.

        Raises
        ------
        NotImplementedError
            Default cost-aware utility construction has not yet been
            implemented in the base class.
        """
        raise NotImplementedError(
            "Default cost-aware utility construction is not implemented yet."
        )

    @abstractmethod
    def _build_botorch_acquisition(self) -> Any:
        """Construct and return the internal BoTorch acquisition object.

        Returns
        -------
        result : Any
            Concrete BoTorch acquisition object built from the current surrogate
            state, observations cache, and any resolved MF / cost-aware helpers.
        """
        pass

    def _resolve_best_f(self, best_f_override: Optional[float] = None) -> float:
        """Resolve the best observed objective value for improvement-based acquisitions.

        Resolution order:
        1. User-provided override.
        2. Computed from cached observations (max if maximize, min otherwise).
        3. Computed from surrogate training targets as fallback.

        Parameters
        ----------
        best_f_override : float, optional
            User-provided best objective value.

        Returns
        -------
        result : float
            Best observed objective value.

        Raises
        ------
        RuntimeError
            If no observations or training data are available.
        """
        if best_f_override is not None:
            return best_f_override

        if self._observations_cache is not None and len(self._observations_cache) > 0:
            ys = [float(obs.y) for obs in self._observations_cache]
            return max(ys) if self.maximize else min(ys)

        if self._botorch_surrogate is not None:
            _, train_Y = self._botorch_surrogate.get_train_data()
            return train_Y.max().item() if self.maximize else train_Y.min().item()

        raise RuntimeError(
            "Cannot resolve best_f: no observations or training data available. "
            "Call update() with observations or provide best_f explicitly."
        )

    def _score_encoded(self, X: torch.Tensor) -> list[float]:
        """Evaluate the BoTorch acquisition on an already-encoded tensor.

        Parameters
        ----------
        X : torch.Tensor
            Encoded input tensor ready for the BoTorch acquisition function.

        Returns
        -------
        result : list[float]
            Acquisition scores in input order.
        """
        botorch_acqf = self._require_botorch_acqf()
        with torch.no_grad():
            values = botorch_acqf(X)
        return values.detach().cpu().reshape(-1).tolist()

    def score(
        self,
        candidates: Iterable[Candidate],
        cost_weighting: Optional[
            Callable[[list[float], list[Candidate]], list[float]]
        ] = None,
    ) -> list[float]:
        """Score candidates by evaluating them against the acquisition function.

        Returns a constant score of ``1.0`` for every candidate when the
        acquisition has not yet been coupled to a fitted surrogate. This
        allows the active learning loop to sample candidates uniformly on
        the first round before any observations are available.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Iterable of candidates to score.
        cost_weighting : callable, optional
            If provided, called as ``cost_weighting(raw_scores, candidates)``
            after scoring and its return value is used in place of the raw
            scores. Not applied before ``update()`` has been called.

        Returns
        -------
        result : list[float]
            Acquisition scores in the same order as the input candidates.
            All ``1.0`` when called before ``update()``.

        Raises
        ------
        RuntimeError
            If the internal BoTorch acquisition object cannot be built due to
            missing surrogate or observation data.
        """
        cand_list = list(candidates)
        if not cand_list:
            return []
        if self._botorch_acqf is None:
            return [1.0] * len(cand_list)
        if self._botorch_surrogate is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not updated with surrogate before scoring."
            )
        X = self._botorch_surrogate.encode_candidates(cand_list).unsqueeze(1)
        raw_scores = self._score_encoded(X)
        if cost_weighting is None:
            return raw_scores
        return cost_weighting(raw_scores, cand_list)


class AnalyticBoTorchAcquisition(BoTorchAcquisitionBase):
    """Intermediate base class for analytic BoTorch acquisition functions.

    Analytic acquisition functions support singleton scoring where each candidate
    is evaluated independently. Examples include UCB, EI, PI, and posterior mean.

    Notes
    -----
    Analytic BoTorch acquisition functions operate on independently scored
    candidates. This class implements ``score()`` and does not support batch
    joint scoring (see ``QBatchBoTorchAcquisition`` for batch support).
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the analytic BoTorch acquisition.

        Parameters
        ----------
        **kwargs
            Forwarded to ``BoTorchAcquisitionBase.__init__``.
        """
        super().__init__(**kwargs)


class QBatchBoTorchAcquisition(BoTorchAcquisitionBase):
    """Intermediate base class for q-batch / Monte Carlo BoTorch acquisitions.

    This class implements singleton scoring for BoTorch acquisition functions
    that operate on q-batches. Each candidate is evaluated independently as a
    q-batch of size 1.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the q-batch BoTorch acquisition.

        Parameters
        ----------
        **kwargs
            Forwarded to ``BoTorchAcquisitionBase.__init__``.
        """
        super().__init__(**kwargs)
