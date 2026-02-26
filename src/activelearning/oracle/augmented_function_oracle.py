import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin, AugmentedHartmann
from botorch.test_functions.synthetic import SyntheticTestFunction
from activelearning.oracle.oracle import Oracle
from activelearning.utils.types import Candidate, Observation
from typing import Sequence


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
    """

    def __init__(
        self,
        function: SyntheticTestFunction,
        fidelity_costs: dict[int, float],
    ) -> None:
        super().__init__()
        self.fidelity_costs = fidelity_costs
        max_cost = max(fidelity_costs.values())
        self.fidelity_confidences: dict[int, float] = {
            fidelity: cost / max_cost for fidelity, cost in fidelity_costs.items()
        }
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
        evaluation.

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
        observations: list[Observation] = []
        for candidate in candidates:
            fidelity = self._validate_fidelity(candidate)
            normalized_fidelity = self.fidelity_confidences[fidelity]

            x_tensor = torch.tensor(
                [*candidate.x, normalized_fidelity], dtype=torch.float64
            ).unsqueeze(0)
            y_value = self._function(x_tensor).item()
            observations.append(
                Observation(x=candidate.x, y=y_value, fidelity=fidelity)
            )
        return observations


class BraninOracle(AugmentedFunctionOracle):
    """Oracle based on the Augmented Branin multi-fidelity test function.

    Uses the positive (non-negated) AugmentedBranin from BoTorch. Fidelity
    levels {1, 2, 3} are mapped to the [0, 1] fidelity range, with level 3
    corresponding to the full-fidelity Branin (s=1.0).
    """

    def __init__(self) -> None:
        super().__init__(
            # negate=False keeps the positive form (min ≈ 0.397)
            function=AugmentedBranin(negate=False),
            fidelity_costs={1: 0.01, 2: 0.1, 3: 1.0},
        )


class Hartmann6DOracle(AugmentedFunctionOracle):
    """Oracle based on the Augmented Hartmann multi-fidelity test function.

    Uses the negated AugmentedHartmann from BoTorch, making all values positive.
    Fidelity levels {1, 2, 3} are mapped to the [0, 1] fidelity range, with
    level 3 corresponding to the full-fidelity Hartmann (s=1.0).
    """

    def __init__(self) -> None:
        super().__init__(
            # negate=True flips the naturally-negative Hartmann to positive values
            function=AugmentedHartmann(negate=True),
            fidelity_costs={1: 0.125, 2: 0.25, 3: 1.0},
        )
