from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Optional, Sequence, overload

from activelearning.utils.types import Candidate, Observation


class Oracle(ABC):
    """Abstract oracle interface used to obtain labels and costs.

    Oracles represent the ground truth evaluation mechanism, which may
    be expensive or time-consuming to query. Oracles can handle multiple
    fidelity levels and route candidates internally.
    """

    @overload
    @staticmethod
    def _validate_candidate_fidelity(
        candidate: Candidate,
        supported_fidelities: Collection[int],
    ) -> int: ...

    @overload
    @staticmethod
    def _validate_candidate_fidelity(
        candidate: Candidate,
        supported_fidelities: None = ...,
    ) -> None: ...

    @staticmethod
    def _validate_candidate_fidelity(
        candidate: Candidate,
        supported_fidelities: Optional[Collection[int]] = None,
    ) -> Optional[int]:
        """Validate a candidate's fidelity against a set of supported levels.

        Parameters
        ----------
        candidate : Candidate
            The candidate whose fidelity to validate.
        supported_fidelities : Collection[int] or None
            The set of supported fidelity levels. Pass ``None`` for
            single-fidelity oracles, in which case ``candidate.fidelity``
            must also be ``None``.

        Returns
        -------
        fidelity : int or None
            The validated fidelity level, or ``None`` for single-fidelity
            candidates.

        Raises
        ------
        ValueError
            If the candidate's fidelity is incompatible with
            ``supported_fidelities``.
        """
        if supported_fidelities is None:
            if candidate.fidelity is not None:
                raise ValueError(
                    "Candidate fidelity must be None for a single-fidelity oracle."
                )
            return None
        if candidate.fidelity is None:
            raise ValueError(
                "Candidate fidelity must not be None for a multi-fidelity oracle."
            )
        if candidate.fidelity not in supported_fidelities:
            raise ValueError(
                f"Unsupported fidelity {candidate.fidelity}. "
                f"Supported: {sorted(supported_fidelities)}"
            )
        return candidate.fidelity

    @staticmethod
    def _validate_fidelity_confidences(
        fidelity_confidences: dict[int, float],
    ) -> None:
        """Validate that all confidence values are real numbers in [0, 1].

        Parameters
        ----------
        fidelity_confidences : dict[int, float]
            Mapping from fidelity level to confidence value to validate.

        Raises
        ------
        ValueError
            If any value is not a real numeric type or falls outside [0, 1].
        """
        for fidelity, value in fidelity_confidences.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    "fidelity_confidence must be a number in [0, 1] "
                    f"for fidelity {fidelity}"
                )
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"fidelity_confidence must be in [0, 1] for fidelity {fidelity}"
                )

    def get_supported_fidelities(self) -> list[int]:
        """Return list of fidelity levels this oracle can handle.

        Returns
        -------
        supported_fidelities : list[int]
            List of integer fidelity levels supported by this oracle.
            For single-fidelity oracles, returns a single-element list.
        """
        return sorted(self.get_fidelity_confidences().keys())

    @abstractmethod
    def get_fidelity_confidences(self) -> dict[int, float]:
        """Return confidence for each supported fidelity.

        Returns
        -------
        fidelity_confidences : dict[int, float]
            Dictionary mapping each fidelity level to a confidence score in [0, 1].
        """
        pass

    @abstractmethod
    def get_costs(self, candidates: Sequence[Candidate]) -> list[float]:
        """Calculate the cost of querying each candidate.

        Costs are determined based on each candidate's fidelity level.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to query.

        Returns
        -------
        costs : list[float]
            List of costs, one per candidate, in the same order as input.

        Raises
        ------
        ValueError
            If a candidate has an unsupported fidelity level.
        """
        pass

    @abstractmethod
    def query(self, candidates: Sequence[Candidate]) -> Sequence[Observation]:
        """Query the oracle for labels of the given candidates.

        Handles candidates with mixed fidelity levels internally.
        Budget consumption is the caller's responsibility - calculate costs
        with get_costs() and consume budget before calling this method.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to label.

        Returns
        -------
        result : Sequence[Observation]
            Sequence of observations with same length and order as input candidates.

        Raises
        ------
        ValueError
            If a candidate has an unsupported fidelity level.
        """
        pass
