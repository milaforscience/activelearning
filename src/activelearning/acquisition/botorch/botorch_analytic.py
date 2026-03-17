"""Concrete wrappers for BoTorch analytic acquisition functions.

Each class subclasses :class:`AnalyticBoTorchAcquisition` and implements
:meth:`_build_botorch_acquisition` to construct the corresponding BoTorch
analytic acquisition object.
"""

from typing import Any, Optional

from botorch.acquisition.analytic import (
    ExpectedImprovement as _EI,
    LogExpectedImprovement as _LogEI,
    LogProbabilityOfImprovement as _LogPI,
    PosteriorMean as _PosteriorMean,
    ProbabilityOfImprovement as _PI,
    UpperConfidenceBound as _UCB,
)

from activelearning.acquisition.botorch.botorch_acquisition import (
    AnalyticBoTorchAcquisition,
)


class UpperConfidenceBound(AnalyticBoTorchAcquisition):
    """Analytic Upper Confidence Bound (UCB).

    Scores candidates as ``posterior_mean + beta * posterior_std``.

    Parameters
    ----------
    beta : float, default=2.0
        Exploration weight controlling the trade-off between exploitation
        (mean) and exploration (standard deviation).
    **kwargs
        Forwarded to :class:`AnalyticBoTorchAcquisition`.
    """

    def __init__(self, *, beta: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._beta = beta

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch UpperConfidenceBound object."""
        assert self._botorch_surrogate is not None
        return _UCB(
            model=self._botorch_surrogate.get_model(),
            beta=self._beta,
            maximize=self.maximize,
        )


class ExpectedImprovement(AnalyticBoTorchAcquisition):
    """Analytic Expected Improvement (EI).

    Parameters
    ----------
    best_f : float, optional
        Best observed objective value. If ``None``, auto-computed from
        observations or training data.
    **kwargs
        Forwarded to :class:`AnalyticBoTorchAcquisition`.
    """

    def __init__(self, *, best_f: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._best_f_override = best_f

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch ExpectedImprovement object."""
        assert self._botorch_surrogate is not None
        return _EI(
            model=self._botorch_surrogate.get_model(),
            best_f=self._resolve_best_f(self._best_f_override),
            maximize=self.maximize,
        )


class LogExpectedImprovement(AnalyticBoTorchAcquisition):
    """Analytic Expected Improvement in log-space for improved numerics.

    Parameters
    ----------
    best_f : float, optional
        Best observed objective value. If ``None``, auto-computed.
    **kwargs
        Forwarded to :class:`AnalyticBoTorchAcquisition`.
    """

    def __init__(self, *, best_f: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._best_f_override = best_f

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch LogExpectedImprovement object."""
        assert self._botorch_surrogate is not None
        return _LogEI(
            model=self._botorch_surrogate.get_model(),
            best_f=self._resolve_best_f(self._best_f_override),
            maximize=self.maximize,
        )


class ProbabilityOfImprovement(AnalyticBoTorchAcquisition):
    """Analytic Probability of Improvement (PI).

    Parameters
    ----------
    best_f : float, optional
        Best observed objective value. If ``None``, auto-computed.
    **kwargs
        Forwarded to :class:`AnalyticBoTorchAcquisition`.
    """

    def __init__(self, *, best_f: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._best_f_override = best_f

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch ProbabilityOfImprovement object."""
        assert self._botorch_surrogate is not None
        return _PI(
            model=self._botorch_surrogate.get_model(),
            best_f=self._resolve_best_f(self._best_f_override),
            maximize=self.maximize,
        )


class LogProbabilityOfImprovement(AnalyticBoTorchAcquisition):
    """Analytic Probability of Improvement in log-space for improved numerics.

    Parameters
    ----------
    best_f : float, optional
        Best observed objective value. If ``None``, auto-computed.
    **kwargs
        Forwarded to :class:`AnalyticBoTorchAcquisition`.
    """

    def __init__(self, *, best_f: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._best_f_override = best_f

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch LogProbabilityOfImprovement object."""
        assert self._botorch_surrogate is not None
        return _LogPI(
            model=self._botorch_surrogate.get_model(),
            best_f=self._resolve_best_f(self._best_f_override),
            maximize=self.maximize,
        )


class PosteriorMean(AnalyticBoTorchAcquisition):
    """Analytic Posterior Mean acquisition — pure exploitation.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`AnalyticBoTorchAcquisition`.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch PosteriorMean object."""
        assert self._botorch_surrogate is not None
        return _PosteriorMean(
            model=self._botorch_surrogate.get_model(),
            maximize=self.maximize,
        )
