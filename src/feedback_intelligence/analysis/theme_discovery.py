"""Theme discovery utilities based on review embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(slots=True)
class ThemeDiscoveryResult:
    """Cluster assignments and human-readable theme labels."""

    assignments: list[int]
    top_terms_by_cluster: dict[int, list[str]]


def discover_themes(
    embeddings: np.ndarray,
    tfidf_matrix,
    feature_names: np.ndarray,
    n_clusters: int,
    top_terms_per_cluster: int,
    seed: int,
) -> ThemeDiscoveryResult:
    """Cluster review embeddings and derive representative terms per cluster."""
    if len(embeddings) == 0:
        return ThemeDiscoveryResult(assignments=[], top_terms_by_cluster={})

    cluster_count = max(1, min(n_clusters, len(embeddings)))
    model = KMeans(n_clusters=cluster_count, random_state=seed, n_init="auto")
    assignments = model.fit_predict(embeddings)

    top_terms_by_cluster: dict[int, list[str]] = {}
    for cluster_id in range(cluster_count):
        cluster_rows = np.where(assignments == cluster_id)[0]
        cluster_matrix = tfidf_matrix[cluster_rows]
        weights = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        ranked_indices = weights.argsort()[::-1]
        terms = [
            str(feature_names[index])
            for index in ranked_indices[:top_terms_per_cluster]
            if weights[index] > 0
        ]
        top_terms_by_cluster[cluster_id] = terms or ["misc"]

    return ThemeDiscoveryResult(
        assignments=assignments.tolist(),
        top_terms_by_cluster=top_terms_by_cluster,
    )
