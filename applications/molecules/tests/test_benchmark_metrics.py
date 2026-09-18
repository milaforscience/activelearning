"""Tests for reusable molecule benchmark metrics."""

import pytest

from activelearning_molecules.benchmark_metrics import (
    finite_fidelity_three_scores,
    top_k_score_and_diversity,
)


def test_top_k_metrics_are_deterministic_and_truncated() -> None:
    """Top-k selection uses score order and reports the selected count."""
    metrics = top_k_score_and_diversity(
        {"CC": 2.0, "C": 3.0, "O": 1.0},
        k=2,
    )

    assert metrics["top_k_count"] == 2
    assert metrics["mean_score"] == pytest.approx(2.5)
    assert metrics["diversity"] is not None


def test_top_k_diversity_is_undefined_for_one_molecule() -> None:
    """Pairwise diversity has no value for a singleton."""
    metrics = top_k_score_and_diversity({"C": 1.0}, k=100)

    assert metrics["top_k_count"] == 1
    assert metrics["diversity"] is None


def test_fidelity_three_scores_canonicalize_and_reject_conflicts() -> None:
    """Equivalent molecule strings collapse while conflicting scores fail."""
    observations = [
        {"x": "C(C)", "y": 2.0, "fidelity": 3},
        {"x": "CC", "y": 2.0, "fidelity": 3},
        {"x": "O", "y": 5.0, "fidelity": 1},
    ]

    assert finite_fidelity_three_scores(observations) == {"CC": 2.0}

    with pytest.raises(ValueError, match="Conflicting"):
        finite_fidelity_three_scores(
            [
                {"x": "CC", "y": 2.0, "fidelity": 3},
                {"x": "C(C)", "y": 3.0, "fidelity": 3},
            ]
        )
