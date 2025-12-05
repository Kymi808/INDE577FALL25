"""
Linear Regression models (NumPy-only).

This module provides linear regression implementations with support for
ordinary least squares (closed-form and gradient descent), Ridge (L2),
and Lasso (L1) regularization.

Classes
-------
LinearRegression
    Ordinary least squares linear regression.
RidgeRegression
    Linear regression with L2 regularization.
LassoRegression
    Linear regression with L1 regularization (coordinate descent).

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.linear_regression import LinearRegression
>>> X = np.array([[1], [2], [3], [4]], dtype=float)
>>> y = np.array([2, 4, 6, 8], dtype=float)
>>> model = LinearRegression().fit(X, y)
>>> model.predict([[5]])[0]
10.0
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple, Union, Sequence

import numpy as np

__all__ = [
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
]

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


def _ensure_1d_float(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Ensure y is a 1D numeric array."""
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


class LinearRegression:
    """
    Ordinary Least Squares Linear Regression.

    Supports closed-form (normal equation) and gradient descent solvers.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    solver : {"closed", "gd"}, default="closed"
        Solver to use:
        - "closed": Normal equation (X^T X)^{-1} X^T y
        - "gd": Gradient descent
    learning_rate : float, default=0.01
        Learning rate for gradient descent (ignored if solver="closed").
    max_iter : int, default=1000
        Maximum iterations for gradient descent.
    tol : float, default=1e-6
        Convergence tolerance for gradient descent.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Intercept term (0.0 if fit_intercept=False).
    n_iter_ : int
        Number of iterations (only for solver="gd").

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4]], dtype=float)
    >>> y = np.array([2, 4, 6, 8], dtype=float)
    >>> model = LinearRegression().fit(X, y)
    >>> round(model.coef_[0], 4)
    2.0
    >>> round(model.intercept_, 4)
    0.0
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        solver: Literal["closed", "gd"] = "closed",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        if solver not in ("closed", "gd"):
            raise ValueError("solver must be 'closed' or 'gd'.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """
        Fit the linear regression model.

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
            raise ValueError(
                f"X and y must have same number of samples; "
                f"got {X_arr.shape[0]} vs {y_arr.shape[0]}."
            )

        n_samples, n_features = X_arr.shape

        if self.solver == "closed":
            self._fit_closed(X_arr, y_arr)
        else:
            self._fit_gd(X_arr, y_arr)

        return self

    def _fit_closed(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using the normal equation."""
        if self.fit_intercept:
            X_b = np.c_[np.ones(X.shape[0]), X]
        else:
            X_b = X

        # theta = (X^T X)^{-1} X^T y
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

        self.n_iter_ = 1

    def _fit_gd(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using gradient descent."""
        n_samples, n_features = X.shape

        # Initialize
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0 if self.fit_intercept else 0.0

        for i in range(self.max_iter):
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y

            # Gradients
            grad_coef = (1 / n_samples) * (X.T @ error)
            grad_intercept = (1 / n_samples) * np.sum(error) if self.fit_intercept else 0.0

            # Update
            self.coef_ -= self.learning_rate * grad_coef
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_intercept

            # Convergence check
            if np.linalg.norm(grad_coef) < self.tol:
                self.n_iter_ = i + 1
                return

        self.n_iter_ = self.max_iter

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted values.
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != len(self.coef_):
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {len(self.coef_)}."
            )

        return X_arr @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return the coefficient of determination R^2.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Test samples.
        y : array_like, shape (n_samples,)
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


class RidgeRegression:
    """
    Linear regression with L2 regularization (Ridge).

    Minimizes: ||y - Xw||^2 + alpha * ||w||^2

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Must be positive.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Intercept term.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=float)
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> model = RidgeRegression(alpha=0.5).fit(X, y)
    >>> model.predict([[3, 5]])[0] > 0
    True
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RidgeRegression":
        """Fit the Ridge regression model."""
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_float(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        n_samples, n_features = X_arr.shape

        if self.fit_intercept:
            X_mean = X_arr.mean(axis=0)
            y_mean = y_arr.mean()
            X_centered = X_arr - X_mean
            y_centered = y_arr - y_mean
        else:
            X_centered = X_arr
            y_centered = y_arr
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        # Closed-form: w = (X^T X + alpha * I)^{-1} X^T y
        A = X_centered.T @ X_centered + self.alpha * np.eye(n_features)
        b = X_centered.T @ y_centered
        self.coef_ = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict using the Ridge model."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != len(self.coef_):
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {len(self.coef_)}.")
        return X_arr @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R^2 score."""
        y_arr = _ensure_1d_float(y, "y")
        y_pred = self.predict(X)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)


class LassoRegression:
    """
    Linear regression with L1 regularization (Lasso).

    Uses coordinate descent to minimize: ||y - Xw||^2 / (2n) + alpha * ||w||_1

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Intercept term.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3]], dtype=float)
    >>> y = np.array([1, 2, 3], dtype=float)
    >>> model = LassoRegression(alpha=0.1).fit(X, y)
    >>> len(model.coef_)
    1
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def _soft_threshold(self, x: float, lam: float) -> float:
        """Soft thresholding operator."""
        if x > lam:
            return x - lam
        elif x < -lam:
            return x + lam
        else:
            return 0.0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LassoRegression":
        """Fit the Lasso model using coordinate descent."""
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_float(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        n_samples, n_features = X_arr.shape

        # Center data if fitting intercept
        if self.fit_intercept:
            X_mean = X_arr.mean(axis=0)
            y_mean = y_arr.mean()
            X_c = X_arr - X_mean
            y_c = y_arr - y_mean
        else:
            X_c = X_arr
            y_c = y_arr
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        # Precompute X^T X diagonal and X^T y
        X_col_norms_sq = np.sum(X_c ** 2, axis=0)

        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        residual = y_c.copy()

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                if X_col_norms_sq[j] == 0:
                    continue

                # Add back contribution of feature j
                residual += X_c[:, j] * self.coef_[j]

                # Compute rho_j = X_j^T * residual
                rho_j = X_c[:, j] @ residual

                # Soft threshold
                self.coef_[j] = self._soft_threshold(
                    rho_j / n_samples,
                    self.alpha
                ) / (X_col_norms_sq[j] / n_samples)

                # Update residual
                residual -= X_c[:, j] * self.coef_[j]

            # Check convergence
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict using the Lasso model."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != len(self.coef_):
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {len(self.coef_)}.")
        return X_arr @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R^2 score."""
        y_arr = _ensure_1d_float(y, "y")
        y_pred = self.predict(X)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)