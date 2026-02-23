from typing import Any, Sequence

from activelearning.oracle.oracle import Oracle
from activelearning.utils.types import Candidate, Observation


class MultiFidelityOracle(Oracle):
    """Oracle that supports multiple fidelity levels with per-fidelity costs.

    Each fidelity level has its own cost_per_sample and score_fn.

    Parameters
    ----------
    fidelity_configs : dict[int, dict[str, Any]]
        Dictionary mapping fidelity level (int) to configuration.
        Each config must contain:
        - 'cost_per_sample'
            float - Cost per sample at this fidelity
        - 'score_fn'
            Callable - Function mapping candidate.x to score
        - 'fidelity_confidence'
            float - Confidence score in [0, 1]
    """

    def __init__(
        self,
        fidelity_configs: dict[int, dict[str, Any]],
    ) -> None:
        """Initialize MultiFidelityOracle with fidelity configurations.

        Raises
        ------
        ValueError
            If fidelity is not an integer or required keys are missing.
        """
        # Validate fidelity configs
        for fidelity, config in fidelity_configs.items():
            if not isinstance(fidelity, int):
                raise ValueError(f"Fidelity must be int, got {type(fidelity)}")
            if "cost_per_sample" not in config:
                raise ValueError(
                    f"Missing 'cost_per_sample' in config for fidelity {fidelity}"
                )
            if "score_fn" not in config:
                raise ValueError(
                    f"Missing 'score_fn' in config for fidelity {fidelity}"
                )
            if "fidelity_confidence" not in config:
                raise ValueError(
                    f"Missing 'fidelity_confidence' in config for fidelity {fidelity}"
                )
            fidelity_confidence = config["fidelity_confidence"]
            if not isinstance(fidelity_confidence, (int, float)) or isinstance(
                fidelity_confidence, bool
            ):
                raise ValueError(
                    "fidelity_confidence must be a number in [0, 1] "
                    f"for fidelity {fidelity}"
                )
            if not 0.0 <= float(fidelity_confidence) <= 1.0:
                raise ValueError(
                    f"fidelity_confidence must be in [0, 1] for fidelity {fidelity}"
                )

        self.fidelity_configs = fidelity_configs
        self._supported_fidelities = sorted(self.fidelity_configs.keys())

    def get_fidelity_confidences(self) -> dict[int, float]:
        """Return confidence values by fidelity level, sorted by fidelity."""
        return {
            fidelity: float(self.fidelity_configs[fidelity]["fidelity_confidence"])
            for fidelity in self._supported_fidelities
        }

    def _get_config_for_fidelity(self, fidelity: int) -> dict[str, Any]:
        """Return per-fidelity config or raise ValueError if unsupported."""
        config = self.fidelity_configs.get(fidelity)
        if config is None:
            raise ValueError(
                f"Unsupported fidelity {fidelity}. "
                f"Supported: {self._supported_fidelities}"
            )
        return config

    def get_costs(self, candidates: Sequence[Candidate]) -> list[float]:
        """Get the cost of querying each candidate.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            List of candidates with fidelity specified.

        Returns
        -------
        costs : list[float]
            List of costs, one per candidate.

        Raises
        ------
        ValueError
            If candidate has unsupported fidelity.
        """
        costs = []
        for candidate in candidates:
            fidelity_config = self._get_config_for_fidelity(candidate.fidelity)
            costs.append(fidelity_config["cost_per_sample"])
        return costs

    def query(self, candidates: Sequence[Candidate]) -> list[Observation]:
        """Query the oracle for observations of the given candidates.

        Budget consumption is the caller's responsibility.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            List of candidates to query, each with fidelity specified.

        Returns
        -------
        result : list[Observation]
            List of observations corresponding to each candidate.

        Raises
        ------
        ValueError
            If a candidate has an unsupported fidelity level.
        """
        observations = []
        for candidate in candidates:
            fidelity_config = self._get_config_for_fidelity(candidate.fidelity)
            score_fn = fidelity_config["score_fn"]
            observation = Observation(
                x=candidate.x,
                y=score_fn(candidate.x),
                fidelity=candidate.fidelity,
            )
            observations.append(observation)

        return observations
