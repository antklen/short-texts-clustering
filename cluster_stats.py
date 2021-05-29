"""
Compute cluster statistics.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_cluster_stats(clusters, phrases, embeddings):
    """Compute cluster statistics: cluster center, cluster count,
    maximum distance between points inside clusters.

    Args:
        clusters (pd.Series): Cluster assignments for phrases.
        phrases (pd.Series): Phrases.
        embeddings (np.array): Embeddings for phrases.
    """

    df_clusters = clusters.value_counts().reset_index()
    df_clusters.columns = ['cluster_id', 'cluster_size']

    centers, max_distances, mean_distances = [], [], []
    for cluster_id in df_clusters.cluster_id:

        cluster_phrases = phrases[clusters == cluster_id].values
        cluster_embeddings = embeddings[clusters == cluster_id]

        dist_matrix = compute_dist_matrix(cluster_embeddings, metric='inner_product')
        mean_point_distance = dist_matrix.mean(axis=1)
        center_idx = mean_point_distance.argmin()
        upper_triang_dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

        centers.append(cluster_phrases[center_idx])
        max_distances.append(dist_matrix.max())
        mean_distances.append(upper_triang_dist.mean())

    df_clusters['cluster_center'] = centers
    df_clusters['max_distance'] = max_distances
    df_clusters['mean_distance'] = mean_distances

    return df_clusters


def compute_dist_matrix(X, metric='inner_product'):
    """
    Compute distance matrix between points.

    Args:
        X (np.array): 2D array with data points.
        metric (str): Distance metric for scipy.spatial.distance.pdist or 'inner_product'.
            If 'inner_product' then use np.inner instead of pdist which is much faster.
            np.inner could be used instead of cosine distance for normalized vectors.
    """

    if X.ndim == 1:
        X = X[None, :]

    if metric == 'inner_product':
        dist_matrix = 1 - np.inner(X, X)
    else:
        dist_matrix = squareform(pdist(X, metric=metric))

    return dist_matrix
