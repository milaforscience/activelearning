from collections import defaultdict
from typing import Callable, Sequence, TypeVar

from activelearning.oracle.oracle import Oracle
from activelearning.utils.types import Candidate, Observation

T = TypeVar("T")


class CompositeOracle(Oracle):
    """Composite oracle that delegates to cheapest sub-oracle per fidelity level.

    Routes candidates to sub-oracles based on fidelity, selecting the oracle
    with the lowest per-candidate cost for each fidelity level.

    Parameters
    ----------
    sub_oracles : list[Oracle]
        List of oracle instances to delegate to.
    """

    def __init__(self, sub_oracles: list[Oracle]) -> None:
        if not sub_oracles:
            raise ValueError("CompositeOracle requires at least one sub-oracle")
        self._sub_oracles = sub_oracles

    def get_fidelity_confidences(self) -> dict[int, float]:
        """Return per-fidelity confidences, enforcing cross-oracle consistency."""
        fidelity_confidences: dict[int, float] = {}
        for oracle in self._sub_oracles:
            for fidelity, confidence in oracle.get_fidelity_confidences().items():
                existing_confidence = fidelity_confidences.get(fidelity)
                if existing_confidence is None:
                    fidelity_confidences[fidelity] = confidence
                    continue
                if existing_confidence != confidence:
                    raise ValueError(
                        f"Inconsistent confidence for fidelity {fidelity}: "
                        f"{existing_confidence} vs {confidence}"
                    )
        return {
            fidelity: fidelity_confidences[fidelity]
            for fidelity in sorted(fidelity_confidences)
        }

    def _get_cheapest_oracle(
        self, fidelity: int, candidates: list[Candidate]
    ) -> Oracle:
        """Find sub-oracle with lowest total cost for given fidelity.

        Parameters
        ----------
        fidelity : int
            Fidelity level to query.
        candidates : list[Candidate]
            List of candidates at this fidelity (for cost calculation).

        Returns
        -------
        result : Oracle
            Oracle with minimum total cost for this fidelity.

        Raises
        ------
        ValueError
            If no sub-oracle supports the requested fidelity.
        """
        # Filter to oracles that support this fidelity
        valid_oracles = [
            oracle
            for oracle in self._sub_oracles
            if fidelity in oracle.get_supported_fidelities()
        ]

        if not valid_oracles:
            raise ValueError(
                f"No oracle supports fidelity {fidelity}. "
                f"Supported fidelities: {self.get_supported_fidelities()}"
            )

        # Return oracle with minimum total cost
        return min(valid_oracles, key=lambda oracle: sum(oracle.get_costs(candidates)))

    def _group_by_fidelity(
        self, candidates: Sequence[Candidate]
    ) -> dict[int, list[tuple[int, Candidate]]]:
        """Group candidates by fidelity, preserving original indices.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to group.

        Returns
        -------
        result : dict[int, list[tuple[int, Candidate]]]
            Dictionary mapping fidelity to list of (index, candidate) tuples.
        """
        fidelity_groups: dict[int, list[tuple[int, Candidate]]] = defaultdict(list)
        for i, candidate in enumerate(candidates):
            if candidate.fidelity is None:
                raise ValueError(
                    f"Candidate at index {i} has no fidelity specified. "
                    "All candidates must have a fidelity level for CompositeOracle."
                )
            fidelity_groups[candidate.fidelity].append((i, candidate))
        return fidelity_groups

    def _process_by_fidelity(
        self,
        candidates: Sequence[Candidate],
        process_fn: Callable[[Oracle, list[Candidate]], Sequence[T]],
    ) -> list[T]:
        """Process candidates in groups by fidelity using cheapest oracle.

        Groups candidates by fidelity, applies process_fn to each group using
        the cheapest oracle, and returns results in original order.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to process.
        process_fn : Callable[[Oracle, list[Candidate]], Sequence[T]]
            Function that takes (oracle, group_candidates) and returns results.

        Returns
        -------
        result : list[T]
            List of results in original candidate order.

        Raises
        ------
        ValueError
            If any candidate has unsupported fidelity.
        """
        if not candidates:
            return []

        fidelity_groups = self._group_by_fidelity(candidates)
        results: list[T] = [None] * len(candidates)  # type: ignore

        for fidelity, indexed_candidates in fidelity_groups.items():
            group_candidates = [c for _, c in indexed_candidates]
            cheapest = self._get_cheapest_oracle(fidelity, group_candidates)
            group_results = process_fn(cheapest, group_candidates)

            for (original_idx, _), result in zip(indexed_candidates, group_results):
                results[original_idx] = result

        return results

    def get_costs(self, candidates: Sequence[Candidate]) -> list[float]:
        """Calculate costs by delegating to cheapest oracle per fidelity.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to query.

        Returns
        -------
        costs : list[float]
            List of costs, one per candidate, in original order.

        Raises
        ------
        ValueError
            If any candidate has unsupported fidelity.
        """
        return self._process_by_fidelity(
            candidates, lambda oracle, group: oracle.get_costs(group)
        )

    def query(self, candidates: Sequence[Candidate]) -> Sequence[Observation]:
        """Query sub-oracles for labels, routing by fidelity.

        Budget consumption is the caller's responsibility.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Sequence of candidates to label.

        Returns
        -------
        result : Sequence[Observation]
            List of observations in same order as input candidates.

        Raises
        ------
        ValueError
            If a candidate has an unsupported fidelity level.
        """
        if not candidates:
            return []

        return self._process_by_fidelity(
            candidates, lambda oracle, group: oracle.query(group)
        )
