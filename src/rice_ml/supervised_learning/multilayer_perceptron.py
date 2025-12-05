"""
Multilayer Perceptron (MLP) neural network (NumPy-only).

This module provides a simple feedforward neural network implementation
with backpropagation for classification and regression tasks.

Classes
-------
MLPClassifier
    Multilayer perceptron classifier.
MLPRegressor
    Multilayer perceptron regressor.

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier
>>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
>>> y = np.array([0, 1, 1, 0])  # XOR
>>> clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
>>> clf.fit(X, y)
>>> clf.score(X, y) > 0.5
True
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple, Union, Sequence

import numpy as np

__all__ = ['MLPClassifier', 'MLPRegressor']

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


def _ensure_1d(y: ArrayLike, name: str = "y") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


# Activation functions and their derivatives

def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s * (1 - s)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z) ** 2


def _softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _identity(z: np.ndarray) -> np.ndarray:
    return z


def _identity_derivative(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


ACTIVATIONS = {
    'relu': (_relu, _relu_derivative),
    'sigmoid': (_sigmoid, _sigmoid_derivative),
    'tanh': (_tanh, _tanh_derivative),
    'identity': (_identity, _identity_derivative),
}


class _BaseMLP:
    """Base class for MLP classifiers and regressors."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: Literal['relu', 'sigmoid', 'tanh'] = 'relu',
        learning_rate: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
        momentum: float = 0.9,
    ) -> None:
        if not hidden_layer_sizes or not all(h > 0 for h in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must be a tuple of positive integers.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS.keys())}.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.batch_size = batch_size
        self.momentum = momentum

        self.weights_: Optional[list] = None
        self.biases_: Optional[list] = None
        self.n_iter_: int = 0
        self.loss_curve_: list = []

    def _initialize_weights(self, layer_sizes: list, rng: np.random.Generator) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier initialization
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = rng.normal(0, std, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.weights_.append(W)
            self.biases_.append(b)

    def _forward(self, X: np.ndarray) -> Tuple[list, list]:
        """Forward pass through the network."""
        activations = [X]
        z_values = []

        act_fn, _ = ACTIVATIONS[self.activation]

        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = activations[-1] @ W + b
            z_values.append(z)

            # Use specified activation for hidden layers
            if i < len(self.weights_) - 1:
                a = act_fn(z)
            else:
                # Output layer handled by subclass
                a = self._output_activation(z)
            activations.append(a)

        return activations, z_values

    def _output_activation(self, z: np.ndarray) -> np.ndarray:
        """Output activation (to be overridden by subclass)."""
        raise NotImplementedError

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss (to be overridden by subclass)."""
        raise NotImplementedError

    def _output_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute output layer gradient (to be overridden by subclass)."""
        raise NotImplementedError


class MLPClassifier(_BaseMLP):
    """
    Multilayer Perceptron classifier.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        Number of neurons in each hidden layer.
    activation : {"relu", "sigmoid", "tanh"}, default="relu"
        Activation function for hidden layers.
    learning_rate : float, default=0.001
        Learning rate.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for early stopping.
    random_state : int or None, default=None
        Random seed.
    batch_size : int or None, default=None
        Mini-batch size. If None, use full batch.
    momentum : float, default=0.9
        Momentum for gradient descent.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_iter_ : int
        Number of iterations run.
    loss_curve_ : list
        Loss at each iteration.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    >>> y = np.array([0, 1, 1, 0])  # XOR
    >>> clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=1)
    >>> clf.fit(X, y)
    >>> clf.predict(X).tolist()
    [0, 1, 1, 0]
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: Literal['relu', 'sigmoid', 'tanh'] = 'relu',
        learning_rate: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            batch_size=batch_size,
            momentum=momentum,
        )
        self.classes_: Optional[np.ndarray] = None
        self._n_outputs: int = 0

    def _output_activation(self, z: np.ndarray) -> np.ndarray:
        if self._n_outputs == 1:
            return _sigmoid(z)
        return _softmax(z)

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _output_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPClassifier":
        """
        Fit the MLP classifier.

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

        n_samples, n_features = X_arr.shape

        # One-hot encode for multiclass, keep 1D for binary
        if n_classes == 2:
            self._n_outputs = 1
            y_encoded = (y_arr == self.classes_[1]).astype(float).reshape(-1, 1)
        else:
            self._n_outputs = n_classes
            class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_idx = np.array([class_to_idx[label] for label in y_arr])
            y_encoded = np.zeros((n_samples, n_classes))
            y_encoded[np.arange(n_samples), y_idx] = 1

        # Build layer sizes
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [self._n_outputs]

        # Initialize
        rng = np.random.default_rng(self.random_state)
        self._initialize_weights(layer_sizes, rng)

        # Velocity for momentum
        v_weights = [np.zeros_like(W) for W in self.weights_]
        v_biases = [np.zeros_like(b) for b in self.biases_]

        batch_size = self.batch_size or n_samples
        _, act_derivative = ACTIVATIONS[self.activation]

        self.loss_curve_ = []

        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X_arr[batch_idx]
                y_batch = y_encoded[batch_idx]

                # Forward
                activations, z_values = self._forward(X_batch)

                # Backward
                m = len(batch_idx)
                delta = self._output_gradient(y_batch, activations[-1])

                for i in range(len(self.weights_) - 1, -1, -1):
                    grad_W = (1 / m) * activations[i].T @ delta
                    grad_b = (1 / m) * np.sum(delta, axis=0)

                    # Update with momentum
                    v_weights[i] = self.momentum * v_weights[i] - self.learning_rate * grad_W
                    v_biases[i] = self.momentum * v_biases[i] - self.learning_rate * grad_b

                    self.weights_[i] += v_weights[i]
                    self.biases_[i] += v_biases[i]

                    # Propagate error
                    if i > 0:
                        delta = (delta @ self.weights_[i].T) * act_derivative(z_values[i - 1])

            # Compute loss
            _, z_vals = self._forward(X_arr)
            y_pred = self._output_activation(z_vals[-1])
            loss = self._compute_loss(y_encoded, y_pred)
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

            # Early stopping
            if len(self.loss_curve_) > 1:
                if abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < self.tol:
                    break

        return self

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
        """
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        activations, _ = self._forward(X_arr)
        output = activations[-1]

        if self._n_outputs == 1:
            return np.column_stack([1 - output.ravel(), output.ravel()])
        return output

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return accuracy."""
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))


class MLPRegressor(_BaseMLP):
    """
    Multilayer Perceptron regressor.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        Number of neurons in each hidden layer.
    activation : {"relu", "sigmoid", "tanh"}, default="relu"
        Activation function for hidden layers.
    learning_rate : float, default=0.001
        Learning rate.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for early stopping.
    random_state : int or None, default=None
        Random seed.
    batch_size : int or None, default=None
        Mini-batch size.
    momentum : float, default=0.9
        Momentum for gradient descent.

    Attributes
    ----------
    n_iter_ : int
        Number of iterations run.
    loss_curve_ : list
        Loss at each iteration.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4]], dtype=float)
    >>> y = np.array([1, 2, 3, 4], dtype=float)
    >>> reg = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=0)
    >>> reg.fit(X, y)
    >>> reg.predict([[2.5]])[0] > 0
    True
    """

    def _output_activation(self, z: np.ndarray) -> np.ndarray:
        return z  # Identity for regression

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def _output_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.shape[0]

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPRegressor":
        """
        Fit the MLP regressor.

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
        y_arr = _ensure_1d(y, "y").astype(float).reshape(-1, 1)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        n_samples, n_features = X_arr.shape

        # Build layer sizes
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]

        # Initialize
        rng = np.random.default_rng(self.random_state)
        self._initialize_weights(layer_sizes, rng)

        v_weights = [np.zeros_like(W) for W in self.weights_]
        v_biases = [np.zeros_like(b) for b in self.biases_]

        batch_size = self.batch_size or n_samples
        _, act_derivative = ACTIVATIONS[self.activation]

        self.loss_curve_ = []

        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X_arr[batch_idx]
                y_batch = y_arr[batch_idx]

                activations, z_values = self._forward(X_batch)

                m = len(batch_idx)
                delta = self._output_gradient(y_batch, activations[-1])

                for i in range(len(self.weights_) - 1, -1, -1):
                    grad_W = activations[i].T @ delta
                    grad_b = np.sum(delta, axis=0)

                    v_weights[i] = self.momentum * v_weights[i] - self.learning_rate * grad_W
                    v_biases[i] = self.momentum * v_biases[i] - self.learning_rate * grad_b

                    self.weights_[i] += v_weights[i]
                    self.biases_[i] += v_biases[i]

                    if i > 0:
                        delta = (delta @ self.weights_[i].T) * act_derivative(z_values[i - 1])

            # Compute loss
            activations, _ = self._forward(X_arr)
            loss = self._compute_loss(y_arr, activations[-1])
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

            if len(self.loss_curve_) > 1:
                if abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < self.tol:
                    break

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict target values."""
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        activations, _ = self._forward(X_arr)
        return activations[-1].ravel()

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R^2 score."""
        y_arr = _ensure_1d(y, "y").astype(float)
        y_pred = self.predict(X)

        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)