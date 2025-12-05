"""
Logistic Regression (NumPy-only).

This module provides logistic regression for binary and multiclass
classification using gradient descent optimization.

Classes
-------
LogisticRegression
    Binary and multiclass logistic regression classifier.

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.logistic_regression import LogisticRegression
>>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
>>> y = np.array([0, 0, 1, 1])
>>> clf = LogisticRegression(max_iter=1000).fit(X, y)
>>> clf.predict([[2.5, 3.5]])
array([0])
"""

from __future__ import annotations
from typing import Literal, Optional, Union, Sequence

import numpy as np

__all__ = ['LogisticRegression']

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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    # Clip to avoid overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax function."""
    # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class LogisticRegression:
    """
    Logistic Regression classifier.

    Supports binary classification (sigmoid) and multiclass classification
    (softmax/multinomial) via gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criterion.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    penalty : {"none", "l2"}, default="none"
        Regularization penalty.
    C : float, default=1.0
        Inverse of regularization strength (higher = less regularization).
    multi_class : {"auto", "ovr", "multinomial"}, default="auto"
        Multiclass strategy:
        - "auto": "ovr" for binary, "multinomial" for multiclass
        - "ovr": One-vs-Rest
        - "multinomial": Softmax regression

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    coef_ : ndarray of shape (n_classes, n_features) or (1, n_features)
        Coefficients.
    intercept_ : ndarray of shape (n_classes,) or (1,)
        Intercept terms.
    n_iter_ : int
        Actual number of iterations.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = LogisticRegression(learning_rate=0.5, max_iter=500).fit(X, y)
    >>> clf.predict([[1.5, 1.5]])
    array([0])
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        penalty: Literal["none", "l2"] = "none",
        C: float = 1.0,
        multi_class: Literal["auto", "ovr", "multinomial"] = "auto",
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if C <= 0:
            raise ValueError("C must be positive.")
        if penalty not in ("none", "l2"):
            raise ValueError("penalty must be 'none' or 'l2'.")
        if multi_class not in ("auto", "ovr", "multinomial"):
            raise ValueError("multi_class must be 'auto', 'ovr', or 'multinomial'.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.C = C
        self.multi_class = multi_class

        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """
        Fit the logistic regression model.

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
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("y must contain at least 2 classes.")

        # Determine strategy
        if self.multi_class == "auto":
            strategy = "binary" if n_classes == 2 else "multinomial"
        elif self.multi_class == "ovr":
            strategy = "ovr"
        else:
            strategy = "multinomial"

        if n_classes == 2 and strategy != "ovr":
            strategy = "binary"

        if strategy == "binary":
            self._fit_binary(X_arr, y_arr)
        elif strategy == "multinomial":
            self._fit_multinomial(X_arr, y_arr)
        else:  # ovr
            self._fit_ovr(X_arr, y_arr)

        return self

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit binary logistic regression using sigmoid."""
        n_samples, n_features = X.shape

        # Convert labels to 0/1
        y_binary = (y == self.classes_[1]).astype(float)

        # Initialize weights
        w = np.zeros(n_features)
        b = 0.0

        reg_strength = 1.0 / self.C if self.penalty == "l2" else 0.0

        for i in range(self.max_iter):
            # Forward pass
            z = X @ w + b
            h = _sigmoid(z)

            # Compute gradients
            error = h - y_binary
            grad_w = (1 / n_samples) * (X.T @ error)
            grad_b = (1 / n_samples) * np.sum(error)

            # Add L2 regularization
            if reg_strength > 0:
                grad_w += reg_strength * w

            # Update
            w_old = w.copy()
            w -= self.learning_rate * grad_w
            if self.fit_intercept:
                b -= self.learning_rate * grad_b

            # Check convergence
            if np.max(np.abs(w - w_old)) < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])

    def _fit_multinomial(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit multiclass logistic regression using softmax."""
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # One-hot encode y
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[label] for label in y])
        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y_idx] = 1

        # Initialize weights
        W = np.zeros((n_features, n_classes))
        b = np.zeros(n_classes)

        reg_strength = 1.0 / self.C if self.penalty == "l2" else 0.0

        for i in range(self.max_iter):
            # Forward pass
            z = X @ W + b
            probs = _softmax(z)

            # Compute gradients
            error = probs - Y_onehot
            grad_W = (1 / n_samples) * (X.T @ error)
            grad_b = (1 / n_samples) * np.sum(error, axis=0)

            # Add L2 regularization
            if reg_strength > 0:
                grad_W += reg_strength * W

            # Update
            W_old = W.copy()
            W -= self.learning_rate * grad_W
            if self.fit_intercept:
                b -= self.learning_rate * grad_b

            # Check convergence
            if np.max(np.abs(W - W_old)) < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = W.T  # (n_classes, n_features)
        self.intercept_ = b

    def _fit_ovr(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using One-vs-Rest strategy."""
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)

        for idx, cls in enumerate(self.classes_):
            # Binary problem: current class vs rest
            y_binary = (y == cls).astype(float)

            w = np.zeros(n_features)
            b = 0.0

            reg_strength = 1.0 / self.C if self.penalty == "l2" else 0.0

            for i in range(self.max_iter):
                z = X @ w + b
                h = _sigmoid(z)

                error = h - y_binary
                grad_w = (1 / n_samples) * (X.T @ error)
                grad_b = (1 / n_samples) * np.sum(error)

                if reg_strength > 0:
                    grad_w += reg_strength * w

                w -= self.learning_rate * grad_w
                if self.fit_intercept:
                    b -= self.learning_rate * grad_b

            self.coef_[idx] = w
            self.intercept_[idx] = b

        self.n_iter_ = self.max_iter

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            Probability of each class.
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != self.coef_.shape[1]:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.coef_.shape[1]}."
            )

        n_classes = len(self.classes_)

        if n_classes == 2 and self.coef_.shape[0] == 1:
            # Binary classification
            z = X_arr @ self.coef_[0] + self.intercept_[0]
            p1 = _sigmoid(z)
            return np.column_stack([1 - p1, p1])
        else:
            # Multiclass or OvR
            z = X_arr @ self.coef_.T + self.intercept_
            return _softmax(z)

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
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
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