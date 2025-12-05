"""
Decision Tree Regressor (NumPy-only).

This module provides a CART-style decision tree regressor using
mean squared error as the splitting criterion.

Classes
-------
DecisionTreeRegressor
    Decision tree for regression tasks.

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.regression_trees import DecisionTreeRegressor
>>> X = np.array([[1], [2], [3], [4], [5]], dtype=float)
>>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
>>> reg = DecisionTreeRegressor(max_depth=2)
>>> reg.fit(X, y)
>>> reg.predict([[2.5]])[0] > 0
True
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Sequence

import numpy as np

__all__ = ['DecisionTreeRegressor']

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


def _ensure_1d_float(y: ArrayLike, name: str = "y") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
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


@dataclass
class _RegressionNode:
    """Internal node representation for regression tree."""
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_RegressionNode"] = None
    right: Optional["_RegressionNode"] = None
    value: Optional[float] = None  # Prediction value for leaf nodes

    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTreeRegressor:
    """
    Decision Tree Regressor using CART algorithm with MSE criterion.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf.
    max_features : int or float or None, default=None
        Number of features to consider at each split.
    random_state : int or None, default=None
        Random seed for feature selection.

    Attributes
    ----------
    n_features_ : int
        Number of features.
    tree_ : _RegressionNode
        Root node of the fitted tree.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4]], dtype=float)
    >>> y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> reg = DecisionTreeRegressor(max_depth=2).fit(X, y)
    >>> pred = reg.predict([[2.5]])
    >>> 1.5 < pred[0] < 3.5
    True
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.n_features_: Optional[int] = None
        self.tree_: Optional[_RegressionNode] = None
        self._rng: Optional[np.random.Generator] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DecisionTreeRegressor":
        """
        Fit the decision tree regressor.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : array_like, shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_float(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        self.n_features_ = X_arr.shape[1]
        self._rng = np.random.default_rng(self.random_state)

        self.tree_ = self._grow_tree(X_arr, y_arr, depth=0)
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _RegressionNode:
        """Recursively grow the tree."""
        n_samples = X.shape[0]

        # Compute mean value for this node
        mean_value = float(np.mean(y))

        # Stopping criteria
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.var(y) == 0
        ):
            return _RegressionNode(value=mean_value)

        # Find best split
        feat_idx, threshold, left_mask, right_mask = self._best_split(X, y)

        if feat_idx is None:
            return _RegressionNode(value=mean_value)

        # Recursively grow children
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _RegressionNode(
            feature_index=feat_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
            value=mean_value,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], np.ndarray, np.ndarray]:
        """Find the best split using MSE reduction."""
        n_samples, n_features = X.shape

        if n_samples < 2 * self.min_samples_leaf:
            return None, None, np.array([]), np.array([])

        # Determine features to consider
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
            feature_indices = self._rng.choice(n_features, k, replace=False)
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
            feature_indices = self._rng.choice(n_features, k, replace=False)
        else:
            feature_indices = np.arange(n_features)

        best_mse = np.inf
        best_feat = None
        best_thresh = None
        best_left_mask = np.array([], dtype=bool)
        best_right_mask = np.array([], dtype=bool)

        # Current MSE
        current_var = np.var(y)

        for feat in feature_indices:
            x_col = X[:, feat]
            thresholds = np.unique(x_col)

            if len(thresholds) == 1:
                continue

            for thresh in thresholds:
                left_mask = x_col <= thresh
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # Compute weighted MSE
                mse_left = np.var(y[left_mask]) if n_left > 0 else 0
                mse_right = np.var(y[right_mask]) if n_right > 0 else 0

                weighted_mse = (n_left * mse_left + n_right * mse_right) / n_samples

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feat = feat
                    best_thresh = float(thresh)
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        # Only split if it reduces impurity
        if best_feat is None or best_mse >= current_var:
            return None, None, np.array([]), np.array([])

        return best_feat, best_thresh, best_left_mask, best_right_mask

    def _traverse(self, x: np.ndarray, node: _RegressionNode) -> float:
        """Traverse tree to get prediction for a single sample."""
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict target values.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted values.
        """
        if self.tree_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")

        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_}."
            )

        return np.array([self._traverse(x, self.tree_) for x in X_arr])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return R^2 score.

        Parameters
        ----------
        X : array_like
            Test samples.
        y : array_like
            True values.

        Returns
        -------
        float
            R^2 score.
        """
        y_arr = _ensure_1d_float(y, "y")
        y_pred = self.predict(X)

        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return float(1.0 - ss_res / ss_tot)