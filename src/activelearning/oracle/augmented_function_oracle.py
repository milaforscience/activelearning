from collections.abc import Sequence
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin, AugmentedHartmann
from botorch.test_functions.synthetic import SyntheticTestFunction

from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.oracle.plotting import build_augmented_2d_landscape_figure
from activelearning.utils.types import Candidate, Observation


class AugmentedFunctionOracle(MultiFidelityOracle):
    """Base oracle for BoTorch augmented multi-fidelity test functions.

    Handles the common fidelity cost/confidence setup, cost querying, and
    evaluation for any BoTorch ``SyntheticTestFunction`` whose last input
    dimension is a fidelity parameter in [0, 1].

    Delegates all fidelity dispatch, cost lookup, and observation creation to
    ``MultiFidelityOracle``. Subclasses only need to supply the underlying
    function instance and a mapping of integer fidelity levels to costs.

    Parameters
    ----------
    function : SyntheticTestFunction
        An instantiated BoTorch test function whose last input dimension is
        the fidelity parameter (a float in [0, 1]).
    fidelity_costs : dict[int, float]
        Mapping from integer fidelity level to query cost. The level with the
        highest cost is treated as full fidelity (confidence = 1.0).
    fidelity_confidences : dict[int, float], optional
        Mapping from integer fidelity level to confidence in [0, 1], where
        1.0 represents full fidelity. If None, confidences are derived
        proportionally from costs: ``confidence = cost / max_cost``.
    """

    def __init__(
        self,
        function: SyntheticTestFunction,
        fidelity_costs: dict[int, float],
        fidelity_confidences: Optional[dict[int, float]] = None,
    ) -> None:
        if not fidelity_costs:
            raise ValueError("fidelity_costs must not be empty")

        if fidelity_confidences is None:
            max_cost = max(fidelity_costs.values())
            fidelity_confidences = {
                fidelity: cost / max_cost for fidelity, cost in fidelity_costs.items()
            }
        elif fidelity_confidences.keys() != fidelity_costs.keys():
            raise ValueError(
                "fidelity_confidences keys must match fidelity_costs keys. "
                f"Got {sorted(fidelity_confidences.keys())}, "
                f"expected {sorted(fidelity_costs.keys())}"
            )

        self._function = function

        fidelity_configs: dict[int, dict[str, Any]] = {
            fidelity: {
                "cost_per_sample": fidelity_costs[fidelity],
                "score_fn": self._make_score_fn(
                    function, fidelity_confidences[fidelity]
                ),
                "fidelity_confidence": fidelity_confidences[fidelity],
            }
            for fidelity in fidelity_costs
        }
        super().__init__(fidelity_configs=fidelity_configs)

    def _make_score_fn(
        self,
        function: SyntheticTestFunction,
        confidence: float,
    ) -> Callable[[Any], float]:
        """Return a score function that evaluates ``function`` at a fixed fidelity level.

        The fidelity ``confidence`` value is appended as the last input
        dimension, matching BoTorch's augmented function convention.

        Parameters
        ----------
        function : SyntheticTestFunction
            The BoTorch test function to evaluate.
        confidence : float
            Fidelity confidence in [0, 1] to append as the last input dimension.

        Returns
        -------
        score_fn : Callable[[Any], float]
            A function that takes a candidate's ``x`` (without the fidelity
            dimension) and returns the scalar function value.
        """

        def score_fn(x: Any) -> float:
            x_tensor = torch.tensor(
                [*x, confidence],
                dtype=self.dtype,
                device=self.device,
            ).unsqueeze(0)
            return function(x_tensor).item()

        return score_fn


class BraninOracle(AugmentedFunctionOracle):
    """Oracle based on the Augmented Branin multi-fidelity test function.

    Uses the negated AugmentedBranin from BoTorch so the active learning loop
    can maximize oracle scores while effectively minimizing the underlying
    Branin objective.

    Each query also logs a 2-D landscape figure of the Branin objective at the
    highest configured fidelity, with the queried candidates overlaid and
    colored by fidelity when a runtime logger is bound.

    Parameters
    ----------
    fidelity_costs : dict[int, float]
        Mapping from integer fidelity level to query cost.
    fidelity_confidences : Optional[dict[int, float]], optional
        Mapping from integer fidelity level to confidence in [0, 1].
        If None, confidences are derived proportionally from costs.
    """

    def __init__(
        self,
        fidelity_costs: dict[int, float],
        fidelity_confidences: Optional[dict[int, float]] = None,
    ) -> None:
        super().__init__(
            # negate=True flips Branin so maximizing the oracle minimizes the
            # original Branin objective.
            function=AugmentedBranin(negate=True),
            fidelity_costs=fidelity_costs,
            fidelity_confidences=fidelity_confidences,
        )

    def query(self, candidates: Sequence[Candidate]) -> list[Observation]:
        """Query Branin observations and log the queried landscape when possible."""
        observations = super().query(candidates)
        self._log_query_landscape(candidates)
        return observations

    def _log_query_landscape(self, candidates: Sequence[Candidate]) -> None:
        """Log a contour plot of the Branin landscape with queried candidates."""
        if self.logger is None:
            return

        figure = build_augmented_2d_landscape_figure(
            evaluator=self._function,
            candidates=candidates,
            bounds=((-5.0, 10.0), (0.0, 15.0)),
            fidelity_confidences=self.get_fidelity_confidences(),
            supported_fidelities=self.get_supported_fidelities(),
            dtype=self.dtype,
            device=self.device,
            title="Branin landscape with queried candidates",
        )
        try:
            self.logger.log_figure("branin_landscape_query", figure)
        finally:
            plt.close(figure)


class Hartmann6DOracle(AugmentedFunctionOracle):
    """Oracle based on the Augmented Hartmann multi-fidelity test function.

    Uses the negated AugmentedHartmann from BoTorch, making all values positive.
    Fidelity levels {1, 2, 3} are mapped to the [0, 1] fidelity range, with
    level 3 corresponding to the full-fidelity Hartmann (s=1.0).

    The active learning loop is formulated as a maximization problem, so
    ``negate=True`` ensures consistency with the acquisition logic by flipping
    the naturally-negative Hartmann function to positive values.

    Parameters
    ----------
    fidelity_costs : dict[int, float]
        Mapping from integer fidelity level to query cost.
    fidelity_confidences : Optional[dict[int, float]], optional
        Mapping from integer fidelity level to confidence in [0, 1].
        If None, confidences are derived proportionally from costs.
    """

    def __init__(
        self,
        fidelity_costs: dict[int, float],
        fidelity_confidences: Optional[dict[int, float]] = None,
    ) -> None:
        super().__init__(
            # The Hartmann function is naturally negative; negate=True flips
            # its sign so values are positive. The active learning loop is
            # formulated as a maximization, so this keeps behavior consistent.
            function=AugmentedHartmann(negate=True),
            fidelity_costs=fidelity_costs,
            fidelity_confidences=fidelity_confidences,
        )
