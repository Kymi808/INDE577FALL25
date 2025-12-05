"""
Perceptron classifier (NumPy-only).

This module provides a simple Perceptron implementation for binary
classification.

Classes
-------
Perceptron
    Single-layer perceptron for binary classification.

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.perceptron import Perceptron
>>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
>>> y = np.array([0, 0, 0, 1])  # AND gate
>>> clf = Perceptron(max_iter=100).fit(X, y)
>>> clf.predict([[1, 1]])
array([1])
"""

from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np

__all__ = ['Perceptron']

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric ndarray of dtype float."""
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


def _ensure_1d(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Ensure y is a 1D array."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


class Perceptron:
    """
    Perceptron binary classifier.

    The perceptron is a simple linear classifier that learns a hyperplane
    to separate two classes. It uses the update rule:
        w = w + eta * y * x  (when misclassified)

    Parameters
    ----------
    learning_rate : float, default=1.0
        Learning rate (eta).
    max_iter : int, default=1000
        Maximum number of passes over the training data.
    tol : float or None, default=1e-3
        Stopping criterion. Training stops when the loss doesn't improve
        by at least tol for two consecutive epochs. Set to None to disable.
    fit_intercept : bool, default=True
        Whether to fit an intercept (bias) term.
    random_state : int or None, default=None
        Seed for shuffling the data.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Unique class labels.
    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features.
    intercept_ : float
        Bias term.
    n_iter_ : int
        Actual number of iterations.
    errors_ : list of int
        Number of misclassifications in each epoch.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[2, 1], [3, 2], [1, 4], [2, 5]], dtype=float)
    >>> y = np.array([1, 1, -1, -1])
    >>> clf = Perceptron(max_iter=100, learning_rate=0.1).fit(X, y)
    >>> clf.predict([[2.5, 1.5]])
    array([1])
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 1000,
        tol: Optional[float] = 1e-3,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0
        self.errors_: list = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> "Perceptron":
        """
        Fit the perceptron model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : array_like, shape (n_samples,)
            Target labels.

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError("Perceptron is a binary classifier; y must have exactly 2 classes.")

        n_samples, n_features = X_arr.shape

        # Map labels to -1, +1
        pos_label = self.classes_[1]
        y_binary = np.where(y_arr == pos_label, 1, -1).astype(float)

        # Initialize weights
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        rng = np.random.default_rng(self.random_state)
        self.errors_ = []

        prev_errors = float('inf')

        for epoch in range(self.max_iter):
            # Shuffle data
            indices = rng.permutation(n_samples)
            X_shuffled = X_arr[indices]
            y_shuffled = y_binary[indices]

            errors = 0
            for xi, yi in zip(X_shuffled, y_shuffled):
                # Compute prediction
                linear_output = xi @ self.coef_ + self.intercept_
                y_pred = 1 if linear_output >= 0 else -1

                # Update if misclassified
                if yi != y_pred:
                    update = self.learning_rate * yi
                    self.coef_ += update * xi
                    if self.fit_intercept:
                        self.intercept_ += update
                    errors += 1

            self.errors_.append(errors)
            self.n_iter_ = epoch + 1

            # Early stopping
            if errors == 0:
                break

            if self.tol is not None and prev_errors - errors < self.tol:
                break

            prev_errors = errors

        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """
        Compute the raw decision scores.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray, shape (n_samples,)
            Decision function values (positive = class 1).
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != len(self.coef_):
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {len(self.coef_)}."
            )

        return X_arr @ self.coef_ + self.intercept_

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted labels.
        """
        scores = self.decision_function(X)
        # Map back to original labels
        indices = (scores >= 0).astype(int)
        return self.classes_[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return accuracy score.

        Parameters
        ----------
        X : array_like
            Test samples.
        y : array_like
            True labels.

        Returns
        -------
        float
            Accuracy.
        """
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))