"""Concrete wrappers for BoTorch q-batch (Monte Carlo) acquisition functions.

Each class subclasses :class:`QBatchBoTorchAcquisition` and implements
:meth:`_build_botorch_acquisition` to construct the corresponding BoTorch
MC acquisition object.
"""

import warnings
from typing import Any, Callable, Iterable, Optional

import torch
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient as _qKG,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement as _qLogEI,
    qLogNoisyExpectedImprovement as _qLogNEI,
)
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy as _qMES,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement as _qEI,
    qNoisyExpectedImprovement as _qNEI,
    qProbabilityOfImprovement as _qPI,
    qSimpleRegret as _qSimpleRegret,
    qUpperConfidenceBound as _qUCB,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from activelearning.acquisition.botorch.botorch_acquisition import (
    BatchScoringMixin,
    QBatchBoTorchAcquisition,
)
from activelearning.acquisition.botorch.candidate_set import CandidateSetSpec
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Observation


class QExpectedImprovement(BatchScoringMixin, QBatchBoTorchAcquisition):
    """Monte Carlo q-Expected Improvement (qEI).

    Parameters
    ----------
    best_f : float, optional
        Best observed objective value. If ``None``, auto-computed.
    constraints : list of callables, optional
        Outcome constraint callables.
    eta : float, default=1e-3
        Temperature for the sigmoid approximation of the constraint indicator.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        best_f: Optional[float] = None,
        constraints: Optional[list[Callable[[torch.Tensor], torch.Tensor]]] = None,
        eta: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._best_f_override = best_f
        self._constraints = constraints
        self._eta = eta

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qExpectedImprovement object."""
        assert self._botorch_surrogate is not None
        best_f = self._resolve_best_f(self._best_f_override)
        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "best_f": -best_f if not self.maximize else best_f,
            "constraints": self._constraints,
            "eta": self._eta,
        }
        if not self.maximize:
            train_X, _ = self._botorch_surrogate.get_train_data()
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=train_X.dtype, device=train_X.device)
            )
        return _qEI(**build_kwargs)


class QLogExpectedImprovement(BatchScoringMixin, QBatchBoTorchAcquisition):
    """Monte Carlo q-Expected Improvement in log-space for improved numerics.

    Parameters
    ----------
    best_f : float, optional
        Best observed objective value. If ``None``, auto-computed.
    constraints : list of callables, optional
        Outcome constraint callables.
    eta : float, default=1e-3
        Temperature for the sigmoid approximation of the constraint indicator.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        best_f: Optional[float] = None,
        constraints: Optional[list[Callable[[torch.Tensor], torch.Tensor]]] = None,
        eta: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._best_f_override = best_f
        self._constraints = constraints
        self._eta = eta

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qLogExpectedImprovement object."""
        assert self._botorch_surrogate is not None
        best_f = self._resolve_best_f(self._best_f_override)
        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "best_f": -best_f if not self.maximize else best_f,
            "constraints": self._constraints,
            "eta": self._eta,
        }
        if not self.maximize:
            train_X, _ = self._botorch_surrogate.get_train_data()
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=train_X.dtype, device=train_X.device)
            )
        return _qLogEI(**build_kwargs)


class QNoisyExpectedImprovement(BatchScoringMixin, QBatchBoTorchAcquisition):
    """Monte Carlo q-Noisy Expected Improvement (qNEI).

    Uses the training inputs as a baseline rather than requiring an explicit
    ``best_f`` value, making it more robust in noisy settings.

    Parameters
    ----------
    prune_baseline : bool, default=True
        Whether to prune dominated baseline points.
    constraints : list of callables, optional
        Outcome constraint callables.
    eta : float, default=1e-3
        Temperature for the sigmoid approximation of the constraint indicator.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        prune_baseline: bool = True,
        constraints: Optional[list[Callable[[torch.Tensor], torch.Tensor]]] = None,
        eta: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._prune_baseline = prune_baseline
        self._constraints = constraints
        self._eta = eta

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qNoisyExpectedImprovement object."""
        assert self._botorch_surrogate is not None
        train_X, _ = self._botorch_surrogate.get_train_data()
        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "X_baseline": train_X,
            "prune_baseline": self._prune_baseline,
            "constraints": self._constraints,
            "eta": self._eta,
        }
        if not self.maximize:
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=train_X.dtype, device=train_X.device)
            )
        return _qNEI(**build_kwargs)


class QLogNoisyExpectedImprovement(BatchScoringMixin, QBatchBoTorchAcquisition):
    """Monte Carlo q-Noisy Expected Improvement in log-space.

    Parameters
    ----------
    prune_baseline : bool, default=True
        Whether to prune dominated baseline points.
    constraints : list of callables, optional
        Outcome constraint callables.
    eta : float, default=1e-3
        Temperature for the sigmoid approximation of the constraint indicator.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        prune_baseline: bool = True,
        constraints: Optional[list[Callable[[torch.Tensor], torch.Tensor]]] = None,
        eta: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._prune_baseline = prune_baseline
        self._constraints = constraints
        self._eta = eta

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qLogNoisyExpectedImprovement object."""
        assert self._botorch_surrogate is not None
        train_X, _ = self._botorch_surrogate.get_train_data()
        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "X_baseline": train_X,
            "prune_baseline": self._prune_baseline,
            "constraints": self._constraints,
            "eta": self._eta,
        }
        if not self.maximize:
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=train_X.dtype, device=train_X.device)
            )
        return _qLogNEI(**build_kwargs)


class QUpperConfidenceBound(BatchScoringMixin, QBatchBoTorchAcquisition):
    """Monte Carlo q-Upper Confidence Bound (qUCB).

    Parameters
    ----------
    beta : float, default=2.0
        Exploration weight.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(self, *, beta: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._beta = beta

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qUpperConfidenceBound object."""
        assert self._botorch_surrogate is not None
        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "beta": self._beta,
        }
        if not self.maximize:
            train_X, _ = self._botorch_surrogate.get_train_data()
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=train_X.dtype, device=train_X.device)
            )
        return _qUCB(**build_kwargs)


class QProbabilityOfImprovement(BatchScoringMixin, QBatchBoTorchAcquisition):
    """Monte Carlo q-Probability of Improvement (qPI).

    Parameters
    ----------
    best_f : float, optional
        Best observed objective value. If ``None``, auto-computed.
    constraints : list of callables, optional
        Outcome constraint callables.
    eta : float, default=1e-3
        Temperature for the sigmoid approximation of the constraint indicator.
    tau : float, default=1e-3
        Temperature for the sigmoid approximation of the improvement indicator.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        best_f: Optional[float] = None,
        constraints: Optional[list[Callable[[torch.Tensor], torch.Tensor]]] = None,
        eta: float = 1e-3,
        tau: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._best_f_override = best_f
        self._constraints = constraints
        self._eta = eta
        self._tau = tau

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qProbabilityOfImprovement object."""
        assert self._botorch_surrogate is not None
        best_f = self._resolve_best_f(self._best_f_override)
        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "best_f": -best_f if not self.maximize else best_f,
            "constraints": self._constraints,
            "eta": self._eta,
            "tau": self._tau,
        }
        if not self.maximize:
            train_X, _ = self._botorch_surrogate.get_train_data()
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=train_X.dtype, device=train_X.device)
            )
        return _qPI(**build_kwargs)


class QSimpleRegret(BatchScoringMixin, QBatchBoTorchAcquisition):
    """Monte Carlo q-Simple Regret.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qSimpleRegret object."""
        assert self._botorch_surrogate is not None
        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
        }
        if not self.maximize:
            train_X, _ = self._botorch_surrogate.get_train_data()
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0], dtype=train_X.dtype, device=train_X.device)
            )
        return _qSimpleRegret(**build_kwargs)


class QKnowledgeGradient(QBatchBoTorchAcquisition):
    """Monte Carlo q-Knowledge Gradient (qKG).

    Parameters
    ----------
    num_fantasies : int, default=64
        Number of fantasy models used for inner optimization.
    current_value : float, optional
        Current best objective value. If ``None``, BoTorch estimates it.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        num_fantasies: int = 64,
        current_value: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if num_fantasies <= 0:
            raise ValueError(f"num_fantasies must be > 0, got {num_fantasies}")
        super().__init__(**kwargs)
        self._num_fantasies = num_fantasies
        self._current_value = current_value

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qKnowledgeGradient object."""
        assert self._botorch_surrogate is not None
        train_X, _ = self._botorch_surrogate.get_train_data()

        # qKG internally uses fantasy models, which require GPyTorch's
        # prediction_strategy to be initialized. This happens on the first
        # posterior evaluation in eval mode.
        model = self._botorch_surrogate.get_model()
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
        return _qKG(**build_kwargs)


class QMaxValueEntropy(QBatchBoTorchAcquisition):
    """Monte Carlo q-Max-Value Entropy Search (qMES).

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
            if type(self._candidate_set_spec).update is not CandidateSetSpec.update:
                warnings.warn(
                    f"{type(self).__name__}.update() called without observations. "
                    "The candidate set will be built from previously cached data, "
                    "which may not reflect the current state of the experiment.",
                    UserWarning,
                    stacklevel=2,
                )
            super().update(surrogate, observations)

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qMaxValueEntropy object."""
        if self._botorch_surrogate is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not updated with surrogate before building acquisition."
            )
        return _qMES(
            model=self._botorch_surrogate.get_model(),
            candidate_set=self._candidate_set_spec.build(self._botorch_surrogate),
            num_fantasies=self._num_fantasies,
            num_mv_samples=self._num_mv_samples,
            num_y_samples=self._num_y_samples,
            maximize=self.maximize,
        )
