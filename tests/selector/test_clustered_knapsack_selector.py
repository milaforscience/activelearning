import warnings
from unittest.mock import Mock

import numpy as np
import pytest

from activelearning.selector.clustered_knapsack_selector import (
    ClusteredKnapsackSelector,
)
from activelearning.utils.types import Candidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_acquisition(values: list[float]) -> Mock:
    """Return a Mock acquisition that returns *values*."""
    acq = Mock()
    acq.return_value = values
    return acq


def _cost_fn(costs: list[float]):
    """Return a cost function that returns *costs* for any candidate list."""
    return lambda _candidates: list(costs)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_raises_when_max_per_cluster_is_zero():
    """Selector rejects max_per_cluster < 1 at construction time."""
    with pytest.raises(ValueError, match="max_per_cluster must be at least 1"):
        ClusteredKnapsackSelector(max_per_cluster=0)


def test_raises_when_max_per_cluster_is_negative():
    """Selector rejects negative max_per_cluster at construction time."""
    with pytest.raises(ValueError, match="max_per_cluster must be at least 1"):
        ClusteredKnapsackSelector(max_per_cluster=-3)


def test_raises_when_kmeans_without_n_clusters():
    """K-means selector requires n_clusters to be specified."""
    with pytest.raises(ValueError, match="n_clusters must be provided"):
        ClusteredKnapsackSelector(max_per_cluster=2, clustering="kmeans")


def test_raises_for_unknown_clustering_algorithm():
    """Selector rejects unrecognised algorithm names."""
    with pytest.raises(ValueError, match="clustering must be 'kmeans' or 'dbscan'"):
        ClusteredKnapsackSelector(max_per_cluster=2, clustering="birch")  # type: ignore[arg-type]


def test_valid_kmeans_construction():
    """K-means selector is constructed without error when n_clusters is given."""
    selector = ClusteredKnapsackSelector(
        max_per_cluster=2, clustering="kmeans", n_clusters=3
    )
    assert selector.max_per_cluster == 2
    assert selector.n_clusters == 3


def test_warns_when_eps_provided_with_kmeans():
    """Selector emits a UserWarning when eps is given with K-means."""
    with pytest.warns(UserWarning, match="eps is ignored"):
        ClusteredKnapsackSelector(
            max_per_cluster=1, clustering="kmeans", n_clusters=3, eps=0.3
        )


def test_warns_when_min_samples_provided_with_kmeans():
    """Selector emits a UserWarning when min_samples is given with K-means."""
    with pytest.warns(UserWarning, match="min_samples is ignored"):
        ClusteredKnapsackSelector(
            max_per_cluster=1, clustering="kmeans", n_clusters=3, min_samples=2
        )


def test_no_warning_when_dbscan_params_at_default_with_kmeans():
    """No warning when eps/min_samples are at their defaults with K-means."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ClusteredKnapsackSelector(max_per_cluster=1, clustering="kmeans", n_clusters=3)


def test_warns_when_n_clusters_provided_with_dbscan():
    """Selector emits a UserWarning when n_clusters is given with DBSCAN."""
    with pytest.warns(UserWarning, match="n_clusters is ignored"):
        ClusteredKnapsackSelector(max_per_cluster=1, clustering="dbscan", n_clusters=3)


def test_no_warning_when_n_clusters_provided_with_kmeans():
    """No warning is emitted when n_clusters is supplied for K-means."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ClusteredKnapsackSelector(max_per_cluster=1, clustering="kmeans", n_clusters=3)

    """DBSCAN selector is constructed without error (n_clusters not required)."""
    selector = ClusteredKnapsackSelector(
        max_per_cluster=1, clustering="dbscan", eps=0.3, min_samples=2
    )
    assert selector.clustering == "dbscan"
    assert selector.eps == 0.3
    assert selector.min_samples == 2


# ---------------------------------------------------------------------------
# Input validation inside __call__
# ---------------------------------------------------------------------------


def test_raises_when_acquisition_missing():
    """Selector raises when acquisition function is not provided."""
    selector = ClusteredKnapsackSelector(max_per_cluster=1, n_clusters=2)
    with pytest.raises(ValueError, match="Acquisition function is required"):
        selector(
            [Candidate(x=np.array([0.0]))],
            cost_fn=_cost_fn([1.0]),
            round_budget=5.0,
        )


def test_raises_when_cost_fn_missing():
    """Selector raises when cost function is not provided."""
    selector = ClusteredKnapsackSelector(max_per_cluster=1, n_clusters=2)
    with pytest.raises(ValueError, match="Cost function is required"):
        selector(
            [Candidate(x=np.array([0.0]))],
            acquisition=_mock_acquisition([1.0]),
            round_budget=5.0,
        )


def test_raises_when_round_budget_missing():
    """Selector raises when budget is not provided."""
    selector = ClusteredKnapsackSelector(max_per_cluster=1, n_clusters=2)
    with pytest.raises(ValueError, match="Budget is required"):
        selector(
            [Candidate(x=np.array([0.0]))],
            acquisition=_mock_acquisition([1.0]),
            cost_fn=_cost_fn([1.0]),
        )


def test_kmeans_raises_when_fewer_candidates_than_clusters():
    """K-means clustering raises when len(candidates) < n_clusters.

    Providing only 2 candidates to a selector configured with n_clusters=3
    is impossible to cluster; the selector must raise a descriptive ValueError
    before invoking KMeans.
    """
    selector = ClusteredKnapsackSelector(
        max_per_cluster=2, clustering="kmeans", n_clusters=3
    )
    candidates = [Candidate(x=np.array([float(i)])) for i in range(2)]
    with pytest.raises(ValueError, match="Cannot cluster 2 candidates into 3"):
        selector(
            candidates,
            acquisition=_mock_acquisition([1.0, 2.0]),
            cost_fn=_cost_fn([1.0, 1.0]),
            round_budget=10.0,
        )


def test_raises_for_negative_costs():
    """Selector rejects negative per-candidate costs."""
    selector = ClusteredKnapsackSelector(max_per_cluster=2, n_clusters=2)
    candidates = [Candidate(x=np.array([float(i)])) for i in range(2)]
    with pytest.raises(ValueError, match="negative cost"):
        selector(
            candidates,
            acquisition=_mock_acquisition([1.0, 2.0]),
            cost_fn=_cost_fn([1.0, -1.0]),
            round_budget=5.0,
        )


def test_returns_empty_for_empty_candidates():
    """Selector returns an empty list when the candidate pool is empty."""
    selector = ClusteredKnapsackSelector(max_per_cluster=2, n_clusters=2)
    result = selector(
        [],
        acquisition=_mock_acquisition([]),
        cost_fn=_cost_fn([]),
        round_budget=10.0,
    )
    assert result == []


# ---------------------------------------------------------------------------
# Cluster-constraint behaviour – K-means
# ---------------------------------------------------------------------------


def test_kmeans_cluster_constraint_limits_selections_per_cluster():
    """Cluster constraint prevents selecting more than max_per_cluster from any cluster.

    Four candidates split into two distinct clusters (two far-apart points each).
    max_per_cluster=1 forces the selector to pick at most one from each cluster,
    so at most 2 candidates total despite ample budget.
    """
    # Two clusters separated widely on the feature axis
    candidates = [
        Candidate(x=np.array([0.0])),  # cluster A
        Candidate(x=np.array([0.1])),  # cluster A
        Candidate(x=np.array([10.0])),  # cluster B
        Candidate(x=np.array([10.1])),  # cluster B
    ]
    selector = ClusteredKnapsackSelector(
        max_per_cluster=1, clustering="kmeans", n_clusters=2
    )
    selected = selector(
        candidates,
        acquisition=_mock_acquisition([1.0, 1.0, 1.0, 1.0]),
        cost_fn=_cost_fn([1.0, 1.0, 1.0, 1.0]),
        round_budget=100.0,
    )
    assert len(selected) <= 2


def test_kmeans_non_binding_constraint_matches_budget_limit():
    """When max_per_cluster >= cluster size, the constraint is non-binding.

    With max_per_cluster equal to the full cluster size, the solver selects
    whatever fits within the budget – same as the unconstrained knapsack.
    """
    candidates = [
        Candidate(x=np.array([0.0])),
        Candidate(x=np.array([0.1])),
    ]
    selector = ClusteredKnapsackSelector(
        max_per_cluster=2, clustering="kmeans", n_clusters=1
    )
    selected = selector(
        candidates,
        acquisition=_mock_acquisition([5.0, 3.0]),
        cost_fn=_cost_fn([1.0, 1.0]),
        round_budget=1.0,
    )
    # Budget allows only 1 candidate; cluster constraint (cap 2) is non-binding
    assert len(selected) == 1
    assert selected[0].x.tolist() == [0.0]  # highest acquisition score


def test_kmeans_selects_best_within_cluster_cap():
    """Within each cluster, the solver selects the highest-value candidates.

    Two clusters of two candidates each with max_per_cluster=1 and enough
    budget for all four.  The solver picks the best one from each cluster.
    """
    # Cluster A: candidates 0 (value 10) and 1 (value 5)
    # Cluster B: candidates 2 (value 8) and 3 (value 3)
    candidates = [
        Candidate(x=np.array([0.0])),
        Candidate(x=np.array([0.1])),
        Candidate(x=np.array([10.0])),
        Candidate(x=np.array([10.1])),
    ]
    selector = ClusteredKnapsackSelector(
        max_per_cluster=1, clustering="kmeans", n_clusters=2
    )
    selected = selector(
        candidates,
        acquisition=_mock_acquisition([10.0, 5.0, 8.0, 3.0]),
        cost_fn=_cost_fn([1.0, 1.0, 1.0, 1.0]),
        round_budget=100.0,
    )
    assert len(selected) == 2
    assert set(c.x[0] for c in selected) == {0.0, 10.0}


# ---------------------------------------------------------------------------
# Cluster-constraint behaviour – DBSCAN
# ---------------------------------------------------------------------------


def test_dbscan_cluster_constraint_limits_selections_per_cluster():
    """DBSCAN cluster constraint caps selections per cluster."""
    # Two tight clusters; eps small enough to form two separate clusters
    candidates = [
        Candidate(x=np.array([0.0])),
        Candidate(x=np.array([0.01])),
        Candidate(x=np.array([5.0])),
        Candidate(x=np.array([5.01])),
    ]
    selector = ClusteredKnapsackSelector(
        max_per_cluster=1, clustering="dbscan", eps=0.05, min_samples=2
    )
    selected = selector(
        candidates,
        acquisition=_mock_acquisition([1.0, 1.0, 1.0, 1.0]),
        cost_fn=_cost_fn([1.0, 1.0, 1.0, 1.0]),
        round_budget=100.0,
    )
    assert len(selected) <= 2


def test_dbscan_noise_points_are_treated_as_singleton_clusters():
    """DBSCAN noise points (label -1) each become their own singleton cluster.

    When min_samples is set high enough, all points are labelled as noise.
    With max_per_cluster=1 each noise point can still be selected up to 1
    time, so the budget (not the cluster cap) becomes the binding constraint.
    """
    # All points will be noise with min_samples=10 and small eps
    candidates = [Candidate(x=np.array([float(i)])) for i in range(4)]
    selector = ClusteredKnapsackSelector(
        max_per_cluster=1,
        clustering="dbscan",
        eps=0.01,
        min_samples=10,
    )
    selected = selector(
        candidates,
        acquisition=_mock_acquisition([4.0, 3.0, 2.0, 1.0]),
        cost_fn=_cost_fn([1.0, 1.0, 1.0, 1.0]),
        round_budget=2.0,
    )
    # Budget allows 2; each noise point is its own cluster (cap 1 is non-binding)
    assert len(selected) == 2
    assert selected[0].x.tolist() == [0.0]
    assert selected[1].x.tolist() == [1.0]


def test_dbscan_noise_points_still_respect_cluster_cap():
    """A noise point singleton is still capped at max_per_cluster.

    Since each noise point forms a singleton cluster, max_per_cluster=1 is
    non-binding for any individual noise point (there's only one in the
    cluster).  This test verifies that the singleton is selectable.
    """
    candidates = [Candidate(x=np.array([0.0]))]
    selector = ClusteredKnapsackSelector(
        max_per_cluster=1,
        clustering="dbscan",
        eps=0.001,
        min_samples=10,
    )
    selected = selector(
        candidates,
        acquisition=_mock_acquisition([5.0]),
        cost_fn=_cost_fn([1.0]),
        round_budget=10.0,
    )
    assert len(selected) == 1


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_accepts_torch_tensor_features():
    """Selector handles Candidate.x values that are torch Tensors."""
    torch = pytest.importorskip("torch")

    candidates = [
        Candidate(x=torch.tensor([0.0])),
        Candidate(x=torch.tensor([10.0])),
    ]
    selector = ClusteredKnapsackSelector(
        max_per_cluster=1, clustering="kmeans", n_clusters=2
    )
    selected = selector(
        candidates,
        acquisition=_mock_acquisition([2.0, 1.0]),
        cost_fn=_cost_fn([1.0, 1.0]),
        round_budget=5.0,
    )
    # max_per_cluster=1 with 2 clusters → at most 2 selected; budget allows both
    assert len(selected) == 2


def test_accepts_multidimensional_numpy_features():
    """Selector flattens multi-dimensional Candidate.x arrays before clustering."""
    candidates = [
        Candidate(x=np.array([[0.0, 0.0]])),
        Candidate(x=np.array([[10.0, 10.0]])),
    ]
    selector = ClusteredKnapsackSelector(
        max_per_cluster=1, clustering="kmeans", n_clusters=2
    )
    selected = selector(
        candidates,
        acquisition=_mock_acquisition([3.0, 1.0]),
        cost_fn=_cost_fn([1.0, 1.0]),
        round_budget=10.0,
    )
    assert len(selected) == 2


# ---------------------------------------------------------------------------
# Inherited behaviour
# ---------------------------------------------------------------------------


def test_inherits_warm_start_parameter():
    """ClusteredKnapsackSelector accepts warm_start via constructor (inherited)."""
    selector = ClusteredKnapsackSelector(
        max_per_cluster=2, n_clusters=1, warm_start=True
    )
    assert selector.warm_start is True


def test_all_zero_acquisition_scores_use_constant_fallback():
    """When all acquisition scores are zero, constant fallback selects cheapest items."""
    candidates = [
        Candidate(x=np.array([0.0])),
        Candidate(x=np.array([0.1])),
    ]
    selector = ClusteredKnapsackSelector(max_per_cluster=2, n_clusters=1)
    selected = selector(
        candidates,
        acquisition=_mock_acquisition([0.0, 0.0]),
        cost_fn=_cost_fn([1.0, 2.0]),
        round_budget=1.0,
    )
    # Cheapest candidate should be chosen when values are all zero
    assert len(selected) == 1
    assert selected[0].x.tolist() == [0.0]
