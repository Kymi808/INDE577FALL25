"""
Ensemble learning methods (NumPy-only).

This module provides ensemble methods including Random Forest and
AdaBoost classifiers built on decision trees.

Classes
-------
RandomForestClassifier
    Random forest ensemble classifier.
AdaBoostClassifier
    Adaptive boosting classifier.
BaggingClassifier
    Bootstrap aggregating classifier.

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.ensemble_methods import RandomForestClassifier
>>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=float)
>>> y = np.array([0, 0, 1, 1, 1])
>>> clf = RandomForestClassifier(n_estimators=10, random_state=42)
>>> clf.fit(X, y)
>>> clf.predict([[3.5, 4.5]])
array([1])
"""

from __future__ import annotations
from typing import Literal, Optional, Union, Sequence

import numpy as np

from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier

__all__ = [
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'BaggingClassifier',
]

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


class RandomForestClassifier:
    """
    Random Forest classifier.

    A random forest is an ensemble of decision trees trained on bootstrap
    samples with random feature subsets at each split.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf.
    max_features : {"sqrt", "log2"} or int or float or None, default="sqrt"
        Number of features to consider at each split:
        - "sqrt": sqrt(n_features)
        - "log2": log2(n_features)
        - int: exact number
        - float: fraction of features
        - None: all features
    bootstrap : bool, default=True
        Whether to use bootstrap samples.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        Fitted trees.
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    feature_importances_ : ndarray
        Feature importances (mean over trees).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 1], [4, 2], [5, 3]], dtype=float)
    >>> y = np.array([0, 0, 1, 1, 1])
    >>> clf = RandomForestClassifier(n_estimators=5, random_state=0)
    >>> clf.fit(X, y)
    >>> clf.predict([[3, 2]])
    array([1])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float, None] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: list = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def _get_max_features(self, n_features: int) -> int:
        """Compute the number of features to consider at each split."""
        if self.max_features is None:
            return n_features
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RandomForestClassifier":
        """
        Fit the random forest.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : array_like, shape (n_samples,)
            Target labels (must be integer-encoded 0, 1, ..., K-1).

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        if not np.issubdtype(y_arr.dtype, np.integer):
            raise ValueError("y must be integer-encoded (0, 1, 2, ...).")

        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)

        n_samples, n_features = X_arr.shape
        max_features = self._get_max_features(n_features)

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for i in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                indices = rng.integers(0, n_samples, size=n_samples)
            else:
                indices = np.arange(n_samples)

            X_boot = X_arr[indices]
            y_boot = y_arr[indices]

            # Create and fit tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=rng.integers(0, 2**31),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

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
            Mean probability across all trees.
        """
        if not self.estimators_:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")

        # Average probabilities across trees
        all_proba = np.array([tree.predict_proba(X_arr) for tree in self.estimators_])
        return np.mean(all_proba, axis=0)

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
        return np.argmax(proba, axis=1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return accuracy."""
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))


class BaggingClassifier:
    """
    Bagging (Bootstrap Aggregating) classifier.

    Fits multiple base classifiers on bootstrap samples and aggregates
    predictions via majority voting.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of base estimators.
    max_samples : float, default=1.0
        Fraction of samples to draw for each estimator.
    max_features : float, default=1.0
        Fraction of features to use for each estimator.
    bootstrap : bool, default=True
        Whether to sample with replacement.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    estimators_ : list
        Fitted base estimators.
    classes_ : ndarray
        Unique class labels.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = BaggingClassifier(n_estimators=5, random_state=0)
    >>> clf.fit(X, y)
    >>> clf.predict([[2.5, 3.5]])
    array([0])
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if not 0 < max_samples <= 1.0:
            raise ValueError("max_samples must be in (0, 1].")
        if not 0 < max_features <= 1.0:
            raise ValueError("max_features must be in (0, 1].")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: list = []
        self.estimator_features_: list = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "BaggingClassifier":
        """Fit the bagging classifier."""
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        if not np.issubdtype(y_arr.dtype, np.integer):
            raise ValueError("y must be integer-encoded.")

        self.classes_ = np.unique(y_arr)
        n_samples, n_features = X_arr.shape

        n_samples_draw = max(1, int(self.max_samples * n_samples))
        n_features_draw = max(1, int(self.max_features * n_features))

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.estimator_features_ = []

        for _ in range(self.n_estimators):
            # Sample selection
            if self.bootstrap:
                sample_idx = rng.integers(0, n_samples, size=n_samples_draw)
            else:
                sample_idx = rng.choice(n_samples, size=n_samples_draw, replace=False)

            # Feature selection
            feature_idx = rng.choice(n_features, size=n_features_draw, replace=False)
            feature_idx = np.sort(feature_idx)

            X_subset = X_arr[sample_idx][:, feature_idx]
            y_subset = y_arr[sample_idx]

            tree = DecisionTreeClassifier(random_state=rng.integers(0, 2**31))
            tree.fit(X_subset, y_subset)

            self.estimators_.append(tree)
            self.estimator_features_.append(feature_idx)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels via majority voting."""
        if not self.estimators_:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        n_samples = X_arr.shape[0]

        # Collect votes
        votes = np.zeros((n_samples, len(self.classes_)), dtype=int)

        for tree, features in zip(self.estimators_, self.estimator_features_):
            X_subset = X_arr[:, features]
            preds = tree.predict(X_subset)
            for i, p in enumerate(preds):
                votes[i, p] += 1

        return np.argmax(votes, axis=1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return accuracy."""
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))


class AdaBoostClassifier:
    """
    AdaBoost (Adaptive Boosting) classifier.

    Uses decision stumps (depth-1 trees) as weak learners.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of boosting rounds.
    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        Fitted weak learners.
    estimator_weights_ : ndarray
        Weights for each estimator.
    classes_ : ndarray
        Unique class labels.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 1], [4, 2]], dtype=float)
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = AdaBoostClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    >>> clf.predict([[2.5, 2]])
    array([0])
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.estimators_: list = []
        self.estimator_weights_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AdaBoostClassifier":
        """
        Fit the AdaBoost classifier.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : array_like, shape (n_samples,)
            Target labels (must be integer-encoded 0, 1, ..., K-1).

        Returns
        -------
        self
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        if not np.issubdtype(y_arr.dtype, np.integer):
            raise ValueError("y must be integer-encoded.")

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        n_samples = X_arr.shape[0]

        # Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        estimator_weights = []

        for _ in range(self.n_estimators):
            # Fit a weak learner (decision stump)
            tree = DecisionTreeClassifier(
                max_depth=1,
                random_state=rng.integers(0, 2**31),
            )

            # Use weighted sampling
            indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights,
            )
            tree.fit(X_arr[indices], y_arr[indices])

            # Predictions on full dataset
            y_pred = tree.predict(X_arr)

            # Compute weighted error
            incorrect = (y_pred != y_arr).astype(float)
            error = np.sum(sample_weights * incorrect)

            # Avoid division by zero
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # SAMME algorithm for multiclass
            alpha = self.learning_rate * (
                np.log((1 - error) / error) + np.log(n_classes - 1)
            )

            # Update sample weights
            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= np.sum(sample_weights)

            self.estimators_.append(tree)
            estimator_weights.append(alpha)

        self.estimator_weights_ = np.array(estimator_weights)
        return self

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
        if not self.estimators_:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)

        # Weighted vote
        class_weights = np.zeros((n_samples, n_classes))

        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X_arr)
            for i, p in enumerate(preds):
                class_weights[i, p] += alpha

        return np.argmax(class_weights, axis=1)

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
            Probabilities.
        """
        if not self.estimators_:
            raise RuntimeError("Model is not fitted.")

        X_arr = _ensure_2d_float(X, "X")
        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)

        class_weights = np.zeros((n_samples, n_classes))

        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X_arr)
            for i, p in enumerate(preds):
                class_weights[i, p] += alpha

        # Normalize to probabilities
        proba = class_weights / class_weights.sum(axis=1, keepdims=True)
        return proba

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return accuracy."""
        y_arr = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))