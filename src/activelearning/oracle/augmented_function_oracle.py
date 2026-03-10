import torch
from typing import Any, Callable, Optional
from botorch.test_functions.multi_fidelity import AugmentedBranin, AugmentedHartmann
from botorch.test_functions.synthetic import SyntheticTestFunction
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle


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

    @staticmethod
    def _make_score_fn(
        function: SyntheticTestFunction, confidence: float
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
            x_tensor = torch.tensor([*x, confidence], dtype=torch.float64).unsqueeze(0)
            return function(x_tensor).item()

        return score_fn


class BraninOracle(AugmentedFunctionOracle):
    """Oracle based on the Augmented Branin multi-fidelity test function.

    Uses the positive (non-negated) AugmentedBranin from BoTorch. Fidelity
    levels {1, 2, 3} are mapped to the [0, 1] fidelity range, with level 3
    corresponding to the full-fidelity Branin (s=1.0).

    The active learning loop is formulated as a maximization problem.
    ``negate=False`` keeps the Branin function in its natural positive form,
    so the loop directly maximizes Branin values.

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
            # negate=False keeps the natural positive form; the active learning
            # loop maximizes Branin values directly.
            function=AugmentedBranin(negate=False),
            fidelity_costs=fidelity_costs,
            fidelity_confidences=fidelity_confidences,
        )


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
