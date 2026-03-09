from typing import Optional, Sequence

import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin, AugmentedHartmann
from botorch.test_functions.synthetic import SyntheticTestFunction

from activelearning.oracle.oracle import Oracle
from activelearning.utils.types import Candidate, Observation


class AugmentedFunctionOracle(Oracle):
    """Base oracle for BoTorch augmented multi-fidelity test functions.

    Handles the common fidelity cost/confidence setup, cost querying, and
    evaluation for any BoTorch ``SyntheticTestFunction`` whose last input
    dimension is a fidelity parameter in [0, 1].

    Subclasses only need to supply the underlying function instance and a
    mapping of integer fidelity levels to costs.

    Parameters
    ----------
    function : SyntheticTestFunction
        An instantiated BoTorch test function whose last input dimension is
        the fidelity parameter.
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
        super().__init__()
        if not fidelity_costs:
            raise ValueError("fidelity_costs must not be empty")

        self.fidelity_costs = fidelity_costs

        if fidelity_confidences is None:
            max_cost = max(fidelity_costs.values())
            fidelity_confidences = {
                fidelity: cost / max_cost for fidelity, cost in fidelity_costs.items()
            }

        self.fidelity_confidences = fidelity_confidences
        self._function = function

    def _validate_fidelity(self, candidate: Candidate) -> int:
        """Validate and return the candidate's fidelity level.

        Raises
        ------
        ValueError
            If fidelity is None or not in the configured fidelity levels.
        """
        if candidate.fidelity is None:
            raise ValueError(
                "Candidate fidelity must not be None for AugmentedFunctionOracle."
            )
        if candidate.fidelity not in self.fidelity_costs:
            raise ValueError(
                f"Unsupported fidelity {candidate.fidelity}. "
                f"Supported: {sorted(self.fidelity_costs.keys())}"
            )
        return candidate.fidelity

    def get_fidelity_confidences(self) -> dict[int, float]:
        """Return confidence proportional to cost for each fidelity."""
        return self.fidelity_confidences

    def get_costs(self, candidates: Sequence[Candidate]) -> list[float]:
        """Return the cost of querying each candidate based on its fidelity."""
        return [self.fidelity_costs[self._validate_fidelity(c)] for c in candidates]

    def query(self, candidates: Sequence[Candidate]) -> list[Observation]:
        """Evaluate candidates using the underlying augmented test function.

        The normalized fidelity is appended as the final input dimension before
        evaluation. All candidates are batched into a single tensor for
        efficient evaluation.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Candidates whose ``x`` matches the input dimensionality of the
            underlying function (excluding the fidelity dimension).

        Returns
        -------
        observations : list[Observation]
            Observations with the function value as y.
        """
        if not candidates:
            return []

        fidelities = [self._validate_fidelity(c) for c in candidates]

        rows = [
            [*c.x, self.fidelity_confidences[f]] for c, f in zip(candidates, fidelities)
        ]
        x_batch = torch.tensor(rows, dtype=torch.float64)
        y_batch = self._function(x_batch)

        return [
            Observation(x=c.x, y=y_val.item(), fidelity=f)
            for c, y_val, f in zip(candidates, y_batch, fidelities)
        ]


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
            function=AugmentedHartmann(negate=True),
            fidelity_costs=fidelity_costs,
            fidelity_confidences=fidelity_confidences,
        )
