"""Concrete wrappers for BoTorch q-batch (Monte Carlo) acquisition functions.

Each class subclasses :class:`QBatchBoTorchAcquisition` and implements
:meth:`_build_botorch_acquisition` to construct the corresponding BoTorch
MC acquisition object.
"""

from typing import Any, Callable, Optional, Sequence

import torch
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient as _qKG,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement as _qLogEI,
    qLogNoisyExpectedImprovement as _qLogNEI,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement as _qEI,
    qNoisyExpectedImprovement as _qNEI,
    qProbabilityOfImprovement as _qPI,
    qSimpleRegret as _qSimpleRegret,
    qUpperConfidenceBound as _qUCB,
)

from activelearning.acquisition.botorch_acquisition import QBatchBoTorchAcquisition


class QExpectedImprovement(QBatchBoTorchAcquisition):
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
        return _qEI(
            model=self._botorch_surrogate.get_model(),
            best_f=self._resolve_best_f(self._best_f_override),
            constraints=self._constraints,
            eta=self._eta,
        )


class QLogExpectedImprovement(QBatchBoTorchAcquisition):
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
        return _qLogEI(
            model=self._botorch_surrogate.get_model(),
            best_f=self._resolve_best_f(self._best_f_override),
            constraints=self._constraints,
            eta=self._eta,
        )


class QNoisyExpectedImprovement(QBatchBoTorchAcquisition):
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
        return _qNEI(
            model=self._botorch_surrogate.get_model(),
            X_baseline=train_X,
            prune_baseline=self._prune_baseline,
            constraints=self._constraints,
            eta=self._eta,
        )


class QLogNoisyExpectedImprovement(QBatchBoTorchAcquisition):
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
        return _qLogNEI(
            model=self._botorch_surrogate.get_model(),
            X_baseline=train_X,
            prune_baseline=self._prune_baseline,
            constraints=self._constraints,
            eta=self._eta,
        )


class QUpperConfidenceBound(QBatchBoTorchAcquisition):
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
        return _qUCB(
            model=self._botorch_surrogate.get_model(),
            beta=self._beta,
        )


class QProbabilityOfImprovement(QBatchBoTorchAcquisition):
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
        return _qPI(
            model=self._botorch_surrogate.get_model(),
            best_f=self._resolve_best_f(self._best_f_override),
            constraints=self._constraints,
            eta=self._eta,
            tau=self._tau,
        )


class QSimpleRegret(QBatchBoTorchAcquisition):
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
        return _qSimpleRegret(
            model=self._botorch_surrogate.get_model(),
        )


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
        super().__init__(supports_batch_scoring=False, **kwargs)
        self._num_fantasies = num_fantasies
        self._current_value = current_value

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qKnowledgeGradient object."""
        assert self._botorch_surrogate is not None
        current_value = (
            torch.tensor(self._current_value) if self._current_value is not None else None
        )
        return _qKG(
            model=self._botorch_surrogate.get_model(),
            num_fantasies=self._num_fantasies,
            current_value=current_value,
        )
