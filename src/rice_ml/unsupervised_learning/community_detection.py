"""
Community Detection (NumPy-only).

This module provides algorithms for detecting communities (clusters)
in graphs represented as adjacency matrices.

Classes
-------
SpectralClustering
    Spectral clustering using graph Laplacian eigenvectors.
LabelPropagation
    Label propagation algorithm for community detection.

Examples
--------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.community_detection import SpectralClustering
>>> # Adjacency matrix for two connected components
>>> A = np.array([[0, 1, 1, 0, 0],
...               [1, 0, 1, 0, 0],
...               [1, 1, 0, 0, 0],
...               [0, 0, 0, 0, 1],
...               [0, 0, 0, 1, 0]], dtype=float)
>>> sc = SpectralClustering(n_clusters=2, random_state=0)
>>> labels = sc.fit_predict(A)
>>> len(set(labels))
2
"""

from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np

__all__ = ['SpectralClustering', 'LabelPropagation']

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


class SpectralClustering:
    """
    Spectral Clustering for community detection.

    Uses the eigenvectors of the graph Laplacian to embed nodes
    in a lower-dimensional space, then applies K-Means.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters/communities.
    affinity : {"precomputed", "rbf"}, default="precomputed"
        How to construct the affinity matrix:
        - "precomputed": Input is already an affinity/adjacency matrix
        - "rbf": Compute RBF kernel from feature matrix
    gamma : float or None, default=None
        Kernel coefficient for RBF. If None, uses 1/n_features.
    n_init : int, default=10
        Number of K-Means initializations.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Affinity matrix used for clustering.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=float)
    >>> sc = SpectralClustering(n_clusters=2, random_state=0)
    >>> labels = sc.fit_predict(A)
    >>> isinstance(labels, np.ndarray)
    True
    """

    def __init__(
        self,
        n_clusters: int = 2,
        affinity: str = "precomputed",
        gamma: Optional[float] = None,
        n_init: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if affinity not in ("precomputed", "rbf"):
            raise ValueError("affinity must be 'precomputed' or 'rbf'.")

        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_init = n_init
        self.random_state = random_state

        self.labels_: Optional[np.ndarray] = None
        self.affinity_matrix_: Optional[np.ndarray] = None

    def _compute_affinity(self, X: np.ndarray) -> np.ndarray:
        """Compute affinity matrix."""
        if self.affinity == "precomputed":
            return X
        else:  # rbf
            gamma = self.gamma or (1.0 / X.shape[1])
            # RBF kernel: exp(-gamma * ||x_i - x_j||^2)
            sq_dists = np.sum(X ** 2, axis=1)[:, np.newaxis] + \
                       np.sum(X ** 2, axis=1)[np.newaxis, :] - \
                       2 * X @ X.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-gamma * sq_dists)

    def _compute_laplacian(self, W: np.ndarray) -> np.ndarray:
        """Compute normalized graph Laplacian."""
        # Degree matrix
        d = np.sum(W, axis=1)
        d_sqrt_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0)
        D_sqrt_inv = np.diag(d_sqrt_inv)

        # Normalized Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}
        I = np.eye(W.shape[0])
        L_norm = I - D_sqrt_inv @ W @ D_sqrt_inv
        return L_norm

    def _kmeans_simple(
        self, X: np.ndarray, n_clusters: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Simple K-Means for embedding."""
        n_samples = X.shape[0]

        best_labels = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            # Random initialization
            indices = rng.choice(n_samples, size=n_clusters, replace=False)
            centroids = X[indices].copy()

            for _ in range(100):
                # Assign labels
                dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
                labels = np.argmin(dists, axis=1)

                # Update centroids
                new_centroids = np.zeros_like(centroids)
                for k in range(n_clusters):
                    mask = labels == k
                    if mask.sum() > 0:
                        new_centroids[k] = X[mask].mean(axis=0)
                    else:
                        new_centroids[k] = centroids[k]

                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            # Compute inertia
            dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            inertia = np.sum(np.min(dists, axis=1))

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels

        return best_labels

    def fit(self, X: ArrayLike) -> "SpectralClustering":
        """
        Fit the spectral clustering model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_samples) or (n_samples, n_features)
            If affinity="precomputed": adjacency/affinity matrix
            If affinity="rbf": feature matrix

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")

        if self.affinity == "precomputed" and X_arr.shape[0] != X_arr.shape[1]:
            raise ValueError("For precomputed affinity, X must be square.")

        # Compute affinity matrix
        self.affinity_matrix_ = self._compute_affinity(X_arr)

        # Compute normalized Laplacian
        L = self._compute_laplacian(self.affinity_matrix_)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Take first k eigenvectors (smallest eigenvalues)
        # Skip the first if it's essentially zero (connected graph)
        embedding = eigenvectors[:, :self.n_clusters]

        # Normalize rows
        row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 0, row_norms, 1)
        embedding = embedding / row_norms

        # K-Means on embedding
        rng = np.random.default_rng(self.random_state)
        self.labels_ = self._kmeans_simple(embedding, self.n_clusters, rng)

        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class LabelPropagation:
    """
    Label Propagation Algorithm for community detection.

    An iterative algorithm where each node adopts the label
    that the maximum number of its neighbors has.

    Parameters
    ----------
    max_iter : int, default=30
        Maximum number of iterations.
    random_state : int or None, default=None
        Random seed for tie-breaking.

    Attributes
    ----------
    labels_ : ndarray of shape (n_nodes,)
        Community labels.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    >>> lp = LabelPropagation(random_state=0)
    >>> labels = lp.fit_predict(A)
    >>> isinstance(labels, np.ndarray)
    True
    """

    def __init__(
        self,
        max_iter: int = 30,
        random_state: Optional[int] = None,
    ) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.max_iter = max_iter
        self.random_state = random_state

        self.labels_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    def fit(self, X: ArrayLike) -> "LabelPropagation":
        """
        Fit the label propagation model.

        Parameters
        ----------
        X : array_like, shape (n_nodes, n_nodes)
            Adjacency matrix (symmetric, non-negative).

        Returns
        -------
        self
        """
        A = _ensure_2d_float(X, "X")

        if A.shape[0] != A.shape[1]:
            raise ValueError("X must be a square adjacency matrix.")

        n_nodes = A.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Initialize: each node in its own community
        labels = np.arange(n_nodes)

        for iteration in range(self.max_iter):
            self.n_iter_ = iteration + 1
            old_labels = labels.copy()

            # Process nodes in random order
            order = rng.permutation(n_nodes)

            for i in order:
                neighbors = np.where(A[i] > 0)[0]
                if len(neighbors) == 0:
                    continue

                # Count neighbor labels (weighted by edge weights)
                neighbor_labels = labels[neighbors]
                weights = A[i, neighbors]

                # Weighted vote
                unique_labels = np.unique(neighbor_labels)
                scores = np.zeros(len(unique_labels))
                for j, label in enumerate(unique_labels):
                    mask = neighbor_labels == label
                    scores[j] = np.sum(weights[mask])

                # Find max score(s)
                max_score = np.max(scores)
                candidates = unique_labels[scores == max_score]

                # Random tie-breaking
                if len(candidates) > 1:
                    labels[i] = rng.choice(candidates)
                else:
                    labels[i] = candidates[0]

            # Check convergence
            if np.array_equal(labels, old_labels):
                break

        # Relabel to consecutive integers
        unique, inverse = np.unique(labels, return_inverse=True)
        self.labels_ = inverse

        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """Fit and return community labels."""
        self.fit(X)
        return self.labels_