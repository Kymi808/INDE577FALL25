"""
Example 1: MNIST Digit Classification
=====================================

This example demonstrates using the rice_ml package to classify
handwritten digits from the MNIST dataset using various algorithms.

Algorithms demonstrated:
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Logistic Regression
- MLP (Neural Network)

Prerequisites:
- mnist.npz file in the data/ directory
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rice_ml import (
    # Preprocessing
    standardize,
    train_test_split,
    # Classifiers
    KNNClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    LogisticRegression,
    MLPClassifier,
    # Metrics
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    # Dimensionality reduction
    PCA,
)


def load_mnist(data_path="data/mnist.npz"):
    """Load MNIST dataset from npz file."""
    # Try multiple paths
    paths_to_try = [
        data_path,
        os.path.join(os.path.dirname(__file__), data_path),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist.npz'),
        os.path.join(os.path.dirname(__file__), 'data', 'mnist.npz'),
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            data_path = path
            break
    else:
        raise FileNotFoundError(
            f"Could not find mnist.npz. Tried: {paths_to_try}"
        )
    
    with np.load(data_path) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        test_images = data["test_images"]
        test_labels = data["test_labels"]
    
    print(f"Loaded MNIST from {data_path}")
    print(f"  Training: {train_images.shape[0]} samples")
    print(f"  Test: {test_images.shape[0]} samples")
    print(f"  Image shape: {train_images.shape[1]} features (28x28 flattened)")
    
    return train_images, train_labels, test_images, test_labels


def subsample_data(X_train, y_train, X_test, y_test, train_size=5000, test_size=1000):
    """Subsample data for faster experimentation."""
    np.random.seed(42)
    
    train_idx = np.random.choice(len(X_train), size=train_size, replace=False)
    test_idx = np.random.choice(len(X_test), size=test_size, replace=False)
    
    return (
        X_train[train_idx],
        y_train[train_idx],
        X_test[test_idx],
        y_test[test_idx]
    )


def print_metrics(y_true, y_pred, model_name):
    """Print classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f} (macro)")
    print(f"  Recall:    {rec:.4f} (macro)")
    print(f"  F1 Score:  {f1:.4f} (macro)")
    
    return acc


def main():
    print("=" * 60)
    print("MNIST Digit Classification with rice_ml")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Subsample for faster training (use full data for better accuracy)
    print("\nSubsampling data for faster experimentation...")
    X_train, y_train, X_test, y_test = subsample_data(
        X_train, y_train, X_test, y_test,
        train_size=5000, test_size=1000
    )
    print(f"  Using {len(X_train)} training and {len(y_test)} test samples")
    
    # Standardize features
    print("\nStandardizing features...")
    X_train_std, params = standardize(X_train, return_params=True)
    X_test_std = (X_test - params["mean"]) / params["scale"]
    
    # Store results
    results = {}
    
    # =========================================================================
    # 1. K-Nearest Neighbors
    # =========================================================================
    print("\n" + "-" * 40)
    print("Training KNN Classifier (k=5)...")
    
    knn = KNNClassifier(n_neighbors=5, metric="euclidean", weights="distance")
    knn.fit(X_train_std, y_train)
    y_pred_knn = knn.predict(X_test_std)
    results["KNN (k=5)"] = print_metrics(y_test, y_pred_knn, "KNN (k=5)")
    
    # =========================================================================
    # 2. Decision Tree
    # =========================================================================
    print("\n" + "-" * 40)
    print("Training Decision Tree...")
    
    dt = DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=42)
    dt.fit(X_train_std, y_train)
    y_pred_dt = dt.predict(X_test_std)
    results["Decision Tree"] = print_metrics(y_test, y_pred_dt, "Decision Tree")
    
    # =========================================================================
    # 3. Random Forest
    # =========================================================================
    print("\n" + "-" * 40)
    print("Training Random Forest (10 trees)...")
    
    rf = RandomForestClassifier(
        n_estimators=10,
        max_depth=15,
        max_features="sqrt",
        random_state=42
    )
    rf.fit(X_train_std, y_train)
    y_pred_rf = rf.predict(X_test_std)
    results["Random Forest"] = print_metrics(y_test, y_pred_rf, "Random Forest")
    
    # =========================================================================
    # 4. Logistic Regression
    # =========================================================================
    print("\n" + "-" * 40)
    print("Training Logistic Regression...")
    
    lr = LogisticRegression(
        max_iter=500,
        learning_rate=0.1,
        penalty="l2",
        C=1.0
    )
    lr.fit(X_train_std, y_train)
    y_pred_lr = lr.predict(X_test_std)
    results["Logistic Regression"] = print_metrics(y_test, y_pred_lr, "Logistic Regression")
    
    # =========================================================================
    # 5. MLP Neural Network
    # =========================================================================
    print("\n" + "-" * 40)
    print("Training MLP Neural Network...")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate=0.01,
        max_iter=100,
        batch_size=32,
        random_state=42
    )
    mlp.fit(X_train_std, y_train)
    y_pred_mlp = mlp.predict(X_test_std)
    results["MLP"] = print_metrics(y_test, y_pred_mlp, "MLP Neural Network")
    
    # =========================================================================
    # 6. PCA + KNN (Dimensionality Reduction)
    # =========================================================================
    print("\n" + "-" * 40)
    print("Training PCA (50 components) + KNN...")
    
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    
    print(f"  Explained variance: {sum(pca.explained_variance_ratio_):.2%}")
    
    knn_pca = KNNClassifier(n_neighbors=5, weights="distance")
    knn_pca.fit(X_train_pca, y_train)
    y_pred_knn_pca = knn_pca.predict(X_test_pca)
    results["PCA + KNN"] = print_metrics(y_test, y_pred_knn_pca, "PCA (50) + KNN")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY - Model Comparison")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10}")
    print("-" * 35)
    for model, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{model:<25} {acc:>10.4f}")
    
    # Show confusion matrix for best model
    best_model = max(results, key=results.get)
    print(f"\nConfusion Matrix for {best_model}:")
    
    # Get predictions for best model
    if best_model == "Random Forest":
        y_pred_best = y_pred_rf
    elif best_model == "MLP":
        y_pred_best = y_pred_mlp
    elif best_model == "KNN (k=5)":
        y_pred_best = y_pred_knn
    elif best_model == "Logistic Regression":
        y_pred_best = y_pred_lr
    elif best_model == "PCA + KNN":
        y_pred_best = y_pred_knn_pca
    else:
        y_pred_best = y_pred_dt
    
    cm = confusion_matrix(y_test, y_pred_best)
    print("     " + " ".join(f"{i:>4}" for i in range(10)))
    for i, row in enumerate(cm):
        print(f"  {i}: " + " ".join(f"{v:>4}" for v in row))


if __name__ == "__main__":
    main()