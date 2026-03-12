from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional

import torch

from activelearning.acquisition.acquisition import Acquisition
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.utils.types import Candidate, Observation


class BoTorchAcquisitionBase(Acquisition, ABC):
    """Base class for BoTorch-backed acquisition functions.

    This class provides the shared infrastructure needed by BoTorch-based
    acquisition functions in the library, including:

    - validation and storage of a typed ``BoTorchGPSurrogate``,
    - lifecycle management of the internal BoTorch acquisition object,
    - materialization of one-pass observation iterables,
    - multi-fidelity metadata resolution,
    - optional target-fidelity projection plumbing,
    - optional cost-aware utility plumbing.

    Concrete subclasses are responsible for constructing the actual BoTorch
    acquisition function by implementing ``_build_botorch_acquisition()``.
    Scoring mechanics are intentionally left to intermediate subclasses such as
    analytic or q-batch acquisition families.
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
            If True, the acquisition assumes a maximization objective. If False,
            subclasses should interpret objective-related quantities accordingly.
        target_fidelity_value : float, optional
            Optional override for the encoded target fidelity value used in
            model space. If None, the value is inferred from the surrogate.
        project_to_target_fidelity_fn : callable, optional
            Optional callable that projects encoded model-space inputs to the
            target fidelity. If None, a default projection is built when needed.
        fidelity_costs : dict[int, float], optional
            Optional mapping from integer fidelity ids to query costs. Used by
            multi-fidelity acquisitions to construct default cost-aware helpers.
        cost_model : object, optional
            Optional custom BoTorch-compatible cost model.
        cost_aware_utility : object, optional
            Optional custom BoTorch-compatible cost-aware utility. If provided,
            this takes precedence over default cost-aware construction.
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
        """Return the typed BoTorch surrogate, if available.

        Returns
        -------
        result : Optional[BoTorchGPSurrogate]
            The typed BoTorch surrogate attached during ``update()``, or
            ``None`` if ``update()`` has not yet been called.
        """
        return self._botorch_surrogate

    @property
    def botorch_acqf(self) -> Optional[Any]:
        """Return the internal BoTorch acquisition object, if available.

        Returns
        -------
        result : Optional[Any]
            The internal BoTorch acquisition object constructed during
            ``update()``, or ``None`` if not yet built.
        """
        return self._botorch_acqf

    @property
    def observations_cache(self) -> Optional[list[Observation]]:
        """Return the materialized observations cached during ``update()``.

        Returns
        -------
        result : Optional[list[Observation]]
            Materialized observations from the current update cycle, or
            ``None`` if no observations were provided.
        """
        return self._observations_cache

    @property
    def resolved_target_fidelity_value(self) -> Optional[float]:
        """Return the resolved encoded target fidelity value.

        Returns
        -------
        result : Optional[float]
            Encoded target fidelity value resolved during ``update()``, or
            ``None`` when single-fidelity or not applicable.
        """
        return self._resolved_target_fidelity_value

    @property
    def resolved_project_to_target_fidelity_fn(
        self,
    ) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """Return the resolved target-fidelity projection callable.

        Returns
        -------
        result : Optional[Callable[[torch.Tensor], torch.Tensor]]
            Projection callable resolved during ``update()``, or ``None`` if not
            applicable.
        """
        return self._resolved_project_to_target_fidelity_fn

    @property
    def resolved_cost_model(self) -> Optional[Any]:
        """Return the resolved cost model.

        Returns
        -------
        result : Optional[Any]
            Cost model resolved during ``update()``, or ``None`` if not used.
        """
        return self._resolved_cost_model

    @property
    def resolved_cost_aware_utility(self) -> Optional[Any]:
        """Return the resolved cost-aware utility.

        Returns
        -------
        result : Optional[Any]
            Cost-aware utility resolved during ``update()``, or ``None`` if not
            used.
        """
        return self._resolved_cost_aware_utility

    def update(
        self,
        surrogate,
        observations: Optional[Iterable[Observation]] = None,
    ) -> None:
        """Refresh the acquisition state after the surrogate has been updated.

        This method validates and stores a typed BoTorch surrogate, materializes
        observations when provided, resolves shared multi-fidelity helpers, and
        rebuilds the internal BoTorch acquisition object.

        Parameters
        ----------
        surrogate : Surrogate
            Surrogate model to use for subsequent acquisition scoring. Must be a
            ``BoTorchGPSurrogate``.
        observations : Optional[Iterable[Observation]]
            Optional iterable of current observations. May be one-pass and is
            therefore materialized internally when provided.

        Raises
        ------
        TypeError
            If the surrogate is not a ``BoTorchGPSurrogate``.
        """
        super().update(surrogate, observations)

        self._botorch_surrogate = self._require_botorch_surrogate(surrogate)
        self._observations_cache = self._materialize_observations(observations)

        # Clear any stale state from a previous update cycle.
        self._botorch_acqf = None
        self._resolved_target_fidelity_value = None
        self._resolved_project_to_target_fidelity_fn = None
        self._resolved_cost_model = None
        self._resolved_cost_aware_utility = None

        # Resolve shared MF / cost-aware helpers before building the acqf.
        self._resolved_target_fidelity_value = self._resolve_target_fidelity_value()
        self._resolved_project_to_target_fidelity_fn = (
            self._resolve_projection_to_target_fidelity()
        )
        self._resolved_cost_model = self._resolve_cost_model()
        self._resolved_cost_aware_utility = self._resolve_cost_aware_utility()

        self._botorch_acqf = self._build_botorch_acquisition()

    def _require_botorch_surrogate(self, surrogate: Any) -> BoTorchGPSurrogate:
        """Validate and return a typed BoTorch surrogate.

        Parameters
        ----------
        surrogate : Any
            Surrogate supplied to ``update()``.

        Returns
        -------
        result : BoTorchGPSurrogate
            The validated BoTorch surrogate.

        Raises
        ------
        TypeError
            If the surrogate is not a ``BoTorchGPSurrogate``.
        """
        if not isinstance(surrogate, BoTorchGPSurrogate):
            raise TypeError(
                f"{self.__class__.__name__} requires a BoTorchGPSurrogate, "
                f"but received {type(surrogate).__name__}."
            )
        return surrogate

    def _materialize_observations(
        self,
        observations: Optional[Iterable[Observation]],
    ) -> Optional[list[Observation]]:
        """Materialize observations into a reusable list.

        Parameters
        ----------
        observations : Optional[Iterable[Observation]]
            Iterable of observations, possibly one-pass.

        Returns
        -------
        result : Optional[list[Observation]]
            Materialized observation list, or ``None`` if no observations were
            supplied.
        """
        if observations is None:
            return None
        return list(observations)

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

    def _tensor_to_list(self, values: torch.Tensor) -> list[float]:
        """Convert an acquisition output tensor to a Python list of floats.

        Parameters
        ----------
        values : torch.Tensor
            Tensor containing one acquisition value per singleton candidate or
            per candidate batch.

        Returns
        -------
        result : list[float]
            Flattened list of floats in the same order as the tensor entries.
        """
        return values.detach().cpu().reshape(-1).tolist()

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

        Notes
        -----
        The default fidelity-cost-based construction is intentionally left as a
        placeholder for the concrete BoTorch-compatible implementation chosen by
        the library.
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


class AnalyticBoTorchAcquisition(BoTorchAcquisitionBase):
    """Intermediate base class for analytic BoTorch acquisition functions.

    This class implements singleton scoring for analytic BoTorch acquisition
    functions that operate on independently scored candidates.

    Examples include:
    - analytic UCB
    - analytic EI
    - analytic PI
    - posterior mean-based scoring

    Notes
    -----
    Analytic BoTorch acquisition functions generally do not support true joint
    q-batch scoring. Accordingly, this class implements ``score()`` only and
    leaves ``score_batches()`` unsupported.
    """

    def supports_singleton_scoring(self) -> bool:
        """Return whether singleton scoring is supported.

        Returns
        -------
        result : bool
            Always ``True`` for analytic BoTorch acquisitions.
        """
        return True

    def supports_batch_scoring(self) -> bool:
        """Return whether joint batch scoring is supported.

        Returns
        -------
        result : bool
            Always ``False`` for analytic BoTorch acquisitions.
        """
        return False

    def score(self, candidates: Iterable[Candidate]) -> list[float]:
        """Score candidates independently using the analytic BoTorch acquisition.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Iterable of candidates to score independently.

        Returns
        -------
        result : list[float]
            Acquisition scores in the same order as the input candidates.

        Raises
        ------
        RuntimeError
            If the internal BoTorch acquisition object has not yet been built.
        """
        botorch_acqf = self._require_botorch_acqf()

        assert self._botorch_surrogate is not None
        X = self._botorch_surrogate.encode_candidates(candidates).unsqueeze(1)

        with torch.no_grad():
            values = botorch_acqf(X)

        return self._tensor_to_list(values)


class QBatchBoTorchAcquisition(BoTorchAcquisitionBase):
    """Intermediate base class for q-batch / Monte Carlo BoTorch acquisitions.

    This class implements both singleton and joint batch scoring for BoTorch
    acquisition functions that naturally operate on q-batches.

    Scoring semantics
    -----------------
    - ``score(candidates)`` evaluates each candidate independently by treating
      it as a q-batch of size 1.
    - ``score_batches(candidate_batches)`` evaluates each candidate batch
      jointly, which is the native mode for q-acquisition functions.

    Examples include:
    - qUCB
    - qEI
    - qNEI
    - qMES
    - multi-fidelity q-acquisition variants
    """

    def supports_singleton_scoring(self) -> bool:
        """Return whether singleton scoring is supported.

        Returns
        -------
        result : bool
            Always ``True`` for q-batch BoTorch acquisitions, since singleton
            scoring is implemented via q-batches of size 1.
        """
        return True

    def supports_batch_scoring(self) -> bool:
        """Return whether joint batch scoring is supported.

        Returns
        -------
        result : bool
            Always ``True`` for q-batch BoTorch acquisitions.
        """
        return True

    def score(self, candidates: Iterable[Candidate]) -> list[float]:
        """Score candidates independently using q-batch semantics with q=1.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Iterable of candidates to score independently.

        Returns
        -------
        result : list[float]
            Acquisition scores in the same order as the input candidates.

        Raises
        ------
        RuntimeError
            If the internal BoTorch acquisition object has not yet been built.
        """
        botorch_acqf = self._require_botorch_acqf()

        assert self._botorch_surrogate is not None
        X = self._botorch_surrogate.encode_candidates(candidates)  # (N, d)
        X = X.unsqueeze(1)  # (N, 1, d)

        with torch.no_grad():
            values = botorch_acqf(X)

        return self._tensor_to_list(values)

    def score_batches(
        self,
        candidate_batches: Iterable[Iterable[Candidate]],
    ) -> list[float]:
        """Score candidate batches jointly using q-batch semantics.

        Parameters
        ----------
        candidate_batches : Iterable[Iterable[Candidate]]
            Iterable of candidate batches. Each inner iterable represents one
            jointly scored batch.

        Returns
        -------
        result : list[float]
            Acquisition scores in the same order as the input batches.

        Raises
        ------
        RuntimeError
            If the internal BoTorch acquisition object has not yet been built.
        """
        botorch_acqf = self._require_botorch_acqf()

        assert self._botorch_surrogate is not None
        X = self._botorch_surrogate.encode_candidate_batches(
            candidate_batches
        )  # (B, q, d)

        with torch.no_grad():
            values = botorch_acqf(X)

        return self._tensor_to_list(values)
