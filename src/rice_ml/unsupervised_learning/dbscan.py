"""
DBSCAN Clustering (NumPy-only).

Density-Based Spatial Clustering of Applications with Noise.

Classes
-------
DBSCAN
    DBSCAN clustering algorithm.

Examples
--------
>>> import numpy as np
>>> from rice_ml.unsupervised_learning.dbscan import DBSCAN
>>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]], dtype=float)
>>> db = DBSCAN(eps=3, min_samples=2)
>>> db.fit(X)
>>> db.labels_
array([ 0,  0,  0,  1,  1, -1])
"""

from __future__ import annotations
from typing import Literal, Optional, Union, Sequence

import numpy as np

__all__ = ['DBSCAN']

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


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    A density-based clustering algorithm that groups together points
    that are closely packed together, marking points in low-density
    regions as outliers.

    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples for one to be considered
        in the neighborhood of the other.
    min_samples : int, default=5
        Minimum number of samples in a neighborhood to form a core point.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels. Noisy samples are given the label -1.
    core_sample_indices_ : ndarray
        Indices of core samples.
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.
    n_features_ : int
        Number of features.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]], dtype=float)
    >>> db = DBSCAN(eps=3, min_samples=2).fit(X)
    >>> set(db.labels_)
    {0, 1, -1}
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between all samples."""
        n = X.shape[0]

        if self.metric == "euclidean":
            # Efficient computation using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            sq_norms = np.sum(X ** 2, axis=1)
            D2 = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T
            D2 = np.maximum(D2, 0)  # Numerical stability
            return np.sqrt(D2)
        else:  # manhattan
            # Use broadcasting: |X[i] - X[j]|
            D = np.zeros((n, n))
            for i in range(n):
                D[i] = np.sum(np.abs(X - X[i]), axis=1)
            return D

    def _get_neighbors(self, D: np.ndarray, i: int) -> np.ndarray:
        """Get indices of neighbors within eps distance."""
        return np.where(D[i] <= self.eps)[0]

    def fit(self, X: ArrayLike) -> "DBSCAN":
        """
        Fit the DBSCAN model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")
        n_samples = X_arr.shape[0]
        self.n_features_ = X_arr.shape[1]

        # Compute pairwise distances
        D = self._pairwise_distances(X_arr)

        # Find core samples
        n_neighbors = np.sum(D <= self.eps, axis=1)
        core_mask = n_neighbors >= self.min_samples
        self.core_sample_indices_ = np.where(core_mask)[0]
        self.components_ = X_arr[self.core_sample_indices_].copy()

        # Initialize labels (-1 = undefined/noise)
        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        # Process each point
        for i in range(n_samples):
            # Skip if already processed
            if labels[i] != -1:
                continue

            # Get neighbors
            neighbors = self._get_neighbors(D, i)

            # Check if core point
            if len(neighbors) < self.min_samples:
                # Noise point (may be reassigned later if reachable from core)
                continue

            # Start new cluster
            labels[i] = cluster_id

            # Process neighbors (seed set)
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]

                if labels[q] == -1:
                    # Was noise, now border point
                    labels[q] = cluster_id
                elif labels[q] != -1:
                    # Already processed
                    j += 1
                    continue

                # If not labeled yet, label it
                if labels[q] == -1:
                    labels[q] = cluster_id

                # Check if q is a core point
                q_neighbors = self._get_neighbors(D, q)
                if len(q_neighbors) >= self.min_samples:
                    # Add new neighbors to seed set
                    for neighbor in q_neighbors:
                        if labels[neighbor] == -1:
                            seed_set.append(neighbor)

                j += 1

            cluster_id += 1

        self.labels_ = labels
        return self

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
            Cluster labels. -1 indicates noise.
        """
        self.fit(X)
        return self.labels_