"""
K-Means Clustering (NumPy-only).

This module provides K-Means clustering with various initialization
methods and convergence options.

Classes
-------
KMeans
    K-Means clustering algorithm.

Examples
--------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.k_means_clustering import KMeans
>>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
>>> kmeans = KMeans(n_clusters=2, random_state=0)
>>> kmeans.fit(X)
>>> kmeans.labels_
array([0, 0, 0, 1, 1, 1])
"""

from __future__ import annotations
from typing import Literal, Optional, Union, Sequence

import numpy as np

__all__ = ['KMeans']

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got {arr.ndim}D.")
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


class KMeans:
    """
    K-Means clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    init : {"k-means++", "random"}, default="k-means++"
        Initialization method:
        - "k-means++": Smart initialization for better convergence
        - "random": Random selection of initial centroids
    n_init : int, default=10
        Number of times to run with different initializations.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centroids.
    labels_ : ndarray of shape (n_samples,)
        Cluster assignments for each sample.
    inertia_ : float
        Sum of squared distances to closest centroid.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
    >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([0, 1])
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'.")
        if n_init < 1:
            raise ValueError("n_init must be >= 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    def _init_centroids_random(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Initialize centroids by random selection."""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _init_centroids_kmeanspp(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Initialize centroids using k-means++ algorithm."""
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features))

        # First centroid: random
        idx = rng.integers(0, n_samples)
        centroids[0] = X[idx]

        # Remaining centroids: weighted by distance squared
        for k in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            dists = np.min(
                np.sum((X[:, np.newaxis, :] - centroids[:k, :]) ** 2, axis=2),
                axis=1
            )
            # Probability proportional to D^2
            probs = dists / dists.sum()
            idx = rng.choice(n_samples, p=probs)
            centroids[k] = X[idx]

        return centroids

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances from samples to centroids."""
        # X: (n_samples, n_features)
        # centroids: (n_clusters, n_features)
        # Result: (n_samples, n_clusters)
        return np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest centroid."""
        distances = self._compute_distances(X, centroids)
        return np.argmin(distances, axis=1)

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute sum of squared distances to assigned centroids."""
        distances = self._compute_distances(X, centroids)
        return float(np.sum(distances[np.arange(len(labels)), labels]))

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """Update centroids as mean of assigned samples."""
        centroids = np.zeros((n_clusters, X.shape[1]))
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(axis=0)
        return centroids

    def _single_run(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple:
        """Run K-Means once with given initialization."""
        # Initialize centroids
        if self.init == "k-means++":
            centroids = self._init_centroids_kmeanspp(X, rng)
        else:
            centroids = self._init_centroids_random(X, rng)

        n_iter = 0
        for i in range(self.max_iter):
            n_iter = i + 1

            # Assign labels
            labels = self._assign_labels(X, centroids)

            # Update centroids
            new_centroids = self._update_centroids(X, labels, self.n_clusters)

            # Check convergence
            centroid_shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids

            if centroid_shift < self.tol:
                break

        # Final assignment and inertia
        labels = self._assign_labels(X, centroids)
        inertia = self._compute_inertia(X, labels, centroids)

        return centroids, labels, inertia, n_iter

    def fit(self, X: ArrayLike) -> "KMeans":
        """
        Fit the K-Means model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")

        if X_arr.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X_arr.shape[0]} must be >= n_clusters={self.n_clusters}."
            )

        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._single_run(X_arr, rng)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict cluster labels for new samples.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            New samples.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster labels.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")

        if X_arr.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.cluster_centers_.shape[1]}."
            )

        return self._assign_labels(X_arr, self.cluster_centers_)

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Fit and return cluster labels.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X to cluster-distance space.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        distances : ndarray, shape (n_samples, n_clusters)
            Distances to each cluster center.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        return np.sqrt(self._compute_distances(X_arr, self.cluster_centers_))

    def score(self, X: ArrayLike) -> float:
        """
        Return negative inertia (for consistency with sklearn).

        Parameters
        ----------
        X : array_like
            Data to score.

        Returns
        -------
        float
            Negative of sum of squared distances.
        """
        X_arr = _ensure_2d_float(X, "X")
        labels = self.predict(X_arr)
        inertia = self._compute_inertia(X_arr, labels, self.cluster_centers_)
        return -inertia