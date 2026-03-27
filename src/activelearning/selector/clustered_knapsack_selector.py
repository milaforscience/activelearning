import warnings
from collections import defaultdict
from typing import Literal, Optional, Sequence

import numpy as np
import pulp
import torch

from activelearning.selector.knapsack_selector import KnapsackSelector
from activelearning.utils.types import Candidate


class ClusteredKnapsackSelector(KnapsackSelector):
    """Knapsack selector with per-cluster cardinality constraints.

    Extends ``KnapsackSelector`` by first clustering the candidate
    features and then adding a MIP constraint that limits the number of
    selected candidates to ``max_per_cluster`` within each cluster.

    The clustering is computed on ``Candidate.x``, which must be a flat
    numeric array or a ``torch.Tensor`` that can be converted to a
    NumPy array.

    Parameters
    ----------
    max_per_cluster : int
        Maximum number of candidates that may be selected from any single
        cluster (``K`` in the MIP constraint).
    clustering : {"kmeans", "dbscan"}
        Clustering algorithm to use.  Defaults to ``"kmeans"``.
    n_clusters : int or None
        Number of clusters for K-means.  Required when
        ``clustering="kmeans"``; ignored for DBSCAN.
    eps : float
        Neighbourhood radius for DBSCAN.  Ignored for K-means.
    min_samples : int
        Minimum number of samples to form a core point in DBSCAN.
        Ignored for K-means.
    time_limit : float or None
        Optional CBC solver time limit in seconds (inherited).
    verbose : bool
        Whether to emit solver logs (inherited).
    warm_start : bool
        Whether to seed the solver with a greedy knapsack solution
        before solving the exact MIP (inherited).

    Raises
    ------
    ValueError
        If ``max_per_cluster < 1``, or if ``clustering="kmeans"`` and
        ``n_clusters`` is not provided.
    """

    def __init__(
        self,
        max_per_cluster: int,
        clustering: Literal["kmeans", "dbscan"] = "kmeans",
        n_clusters: Optional[int] = None,
        eps: float = 0.5,
        min_samples: int = 5,
        time_limit: Optional[float] = None,
        verbose: bool = False,
        warm_start: bool = False,
    ) -> None:
        if clustering not in {"kmeans", "dbscan"}:
            raise ValueError(
                f"clustering must be 'kmeans' or 'dbscan', got '{clustering}'."
            )
        if max_per_cluster < 1:
            raise ValueError(
                f"max_per_cluster must be at least 1, got {max_per_cluster}."
            )
        if clustering == "kmeans" and n_clusters is None:
            raise ValueError("n_clusters must be provided when clustering='kmeans'.")
        if clustering == "dbscan" and n_clusters is not None:
            warnings.warn(
                "n_clusters is ignored when clustering='dbscan'.",
                UserWarning,
                stacklevel=2,
            )
        if clustering == "kmeans" and eps != 0.5:
            warnings.warn(
                "eps is ignored when clustering='kmeans'.",
                UserWarning,
                stacklevel=2,
            )
        if clustering == "kmeans" and min_samples != 5:
            warnings.warn(
                "min_samples is ignored when clustering='kmeans'.",
                UserWarning,
                stacklevel=2,
            )

        super().__init__(
            time_limit=time_limit,
            verbose=verbose,
            warm_start=warm_start,
        )
        self.max_per_cluster = max_per_cluster
        self.clustering = clustering
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples

    def _add_extra_constraints(
        self,
        prob: pulp.LpProblem,
        x: list[pulp.LpVariable],
        candidates: Sequence[Candidate],
    ) -> None:
        """Add per-cluster cardinality constraints to the MIP.

        For each cluster, at most ``max_per_cluster`` candidates may be
        selected.

        Parameters
        ----------
        prob : pulp.LpProblem
            The in-progress maximisation problem.
        x : list[pulp.LpVariable]
            Binary decision variables aligned with *candidates*.
        candidates : Sequence[Candidate]
            The candidate pool being optimised over.
        """
        cluster_labels = self._get_cluster_labels(candidates)

        cluster_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            cluster_to_indices[int(label)].append(i)

        for indices in cluster_to_indices.values():
            prob += pulp.lpSum([x[i] for i in indices]) <= self.max_per_cluster

    def _get_cluster_labels(self, candidates: Sequence[Candidate]) -> np.ndarray:
        """Cluster candidates and return per-candidate integer cluster labels.

        Each element of ``Candidate.x`` is converted to a flat NumPy array
        before clustering.  DBSCAN noise points (label ``-1``) are each
        reassigned to a unique positive cluster id so that they are each
        treated as a singleton cluster, still subject to the
        ``max_per_cluster`` cap.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Candidates whose ``x`` attributes provide the clustering features.

        Returns
        -------
        labels : np.ndarray of shape (n_candidates,) with dtype int
            Non-negative integer cluster label for each candidate.
        """
        features = np.array(
            [
                np.asarray(c.x.detach().cpu()).ravel()
                if isinstance(c.x, torch.Tensor)
                else np.asarray(c.x).ravel()
                for c in candidates
            ]
        )

        if self.clustering == "kmeans":
            from sklearn.cluster import KMeans  # noqa: PLC0415

            if len(candidates) < self.n_clusters:
                raise ValueError(
                    f"Cannot cluster {len(candidates)} candidates into {self.n_clusters} clusters."
                    f"Provide at least {self.n_clusters} candidates or reduce n_clusters."
                )

            labels: np.ndarray = KMeans(
                n_clusters=self.n_clusters, n_init="auto"
            ).fit_predict(features)
        else:  # dbscan
            from sklearn.cluster import DBSCAN  # noqa: PLC0415

            labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(
                features
            )

            # Reassign noise points to unique singleton cluster ids.
            next_id = int(labels.max()) + 1
            for i in np.where(labels == -1)[0]:
                labels[i] = next_id
                next_id += 1

        return labels
