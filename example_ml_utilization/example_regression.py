"""
Example 3: Regression Models Comparison
========================================

This example demonstrates regression algorithms from the rice_ml package
on synthetic and real-world style datasets.

Algorithms demonstrated:
- Linear Regression (closed-form and gradient descent)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- MLP Regressor (Neural Network)

"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rice_ml import (
    # Preprocessing
    standardize,
    train_test_split,
    # Regressors
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    DecisionTreeRegressor,
    KNNRegressor,
    MLPRegressor,
    # Metrics
    mse,
    rmse,
    mae,
    r2_score,
)


def generate_linear_data(n_samples=200, n_features=5, noise=0.5, seed=42):
    """Generate linear regression data with known coefficients."""
    np.random.seed(seed)
    
    # True coefficients
    true_coef = np.array([3.0, -2.0, 1.5, 0.0, -0.5])[:n_features]
    true_intercept = 5.0
    
    X = np.random.randn(n_samples, n_features)
    y = X @ true_coef + true_intercept + noise * np.random.randn(n_samples)
    
    return X, y, true_coef, true_intercept


def generate_nonlinear_data(n_samples=300, seed=42):
    """Generate nonlinear regression data (sin function)."""
    np.random.seed(seed)
    
    X = np.linspace(-2 * np.pi, 2 * np.pi, n_samples).reshape(-1, 1)
    np.random.shuffle(X)
    y = np.sin(X.ravel()) + 0.2 * np.random.randn(n_samples)
    
    return X, y


def generate_sparse_data(n_samples=200, n_features=20, n_informative=5, noise=0.3, seed=42):
    """Generate data with sparse true coefficients (for Lasso demo)."""
    np.random.seed(seed)
    
    # Only first n_informative features are relevant
    true_coef = np.zeros(n_features)
    true_coef[:n_informative] = np.random.randn(n_informative) * 2
    
    X = np.random.randn(n_samples, n_features)
    y = X @ true_coef + noise * np.random.randn(n_samples)
    
    return X, y, true_coef


def print_metrics(y_true, y_pred, model_name):
    """Print regression metrics."""
    print(f"\n{model_name}:")
    print(f"  MSE:  {mse(y_true, y_pred):.4f}")
    print(f"  RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"  MAE:  {mae(y_true, y_pred):.4f}")
    print(f"  R²:   {r2_score(y_true, y_pred):.4f}")
    return r2_score(y_true, y_pred)


def main():
    print("=" * 60)
    print("Regression Models Comparison with rice_ml")
    print("=" * 60)
    
    # =========================================================================
    # Part 1: Linear Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: Linear Regression on Linear Data")
    print("=" * 60)
    
    X, y, true_coef, true_intercept = generate_linear_data(n_samples=300, n_features=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"True coefficients: {true_coef}")
    print(f"True intercept: {true_intercept}")
    
    results_linear = {}
    
    # Linear Regression (closed-form)
    print("\n" + "-" * 40)
    lr = LinearRegression(solver="closed")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results_linear["Linear (closed)"] = print_metrics(y_test, y_pred, "Linear Regression (closed-form)")
    print(f"  Learned coef: {lr.coef_.round(2)}")
    print(f"  Learned intercept: {lr.intercept_:.2f}")
    
    # Linear Regression (gradient descent)
    lr_gd = LinearRegression(solver="gd", learning_rate=0.01, max_iter=1000)
    lr_gd.fit(X_train, y_train)
    y_pred = lr_gd.predict(X_test)
    results_linear["Linear (GD)"] = print_metrics(y_test, y_pred, "Linear Regression (gradient descent)")
    
    # Ridge Regression
    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results_linear["Ridge"] = print_metrics(y_test, y_pred, "Ridge Regression (α=1.0)")
    
    # =========================================================================
    # Part 2: Sparse Data (Lasso vs Ridge)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: Sparse Coefficients - Lasso vs Ridge")
    print("=" * 60)
    
    X_sparse, y_sparse, true_sparse_coef = generate_sparse_data(
        n_samples=300, n_features=20, n_informative=5
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_sparse, y_sparse, test_size=0.2, random_state=42
    )
    
    # Standardize for fair comparison
    X_train_std, params = standardize(X_train, return_params=True)
    X_test_std = (X_test - params["mean"]) / params["scale"]
    
    print(f"\nDataset: {len(X_sparse)} samples, {X_sparse.shape[1]} features")
    print(f"True non-zero coefficients: {np.sum(true_sparse_coef != 0)}")
    print(f"True coefficients: {true_sparse_coef.round(2)}")
    
    # Ridge
    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train_std, y_train)
    y_pred = ridge.predict(X_test_std)
    print_metrics(y_test, y_pred, "Ridge Regression")
    ridge_zeros = np.sum(np.abs(ridge.coef_) < 0.1)
    print(f"  Near-zero coefficients: {ridge_zeros}/{len(ridge.coef_)}")
    
    # Lasso
    lasso = LassoRegression(alpha=0.1, max_iter=2000)
    lasso.fit(X_train_std, y_train)
    y_pred = lasso.predict(X_test_std)
    print_metrics(y_test, y_pred, "Lasso Regression")
    lasso_zeros = np.sum(np.abs(lasso.coef_) < 0.01)
    print(f"  Zero coefficients: {lasso_zeros}/{len(lasso.coef_)}")
    print(f"  Lasso coef: {lasso.coef_.round(2)}")
    
    # =========================================================================
    # Part 3: Nonlinear Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 3: Nonlinear Data (sin function)")
    print("=" * 60)
    
    X_nl, y_nl = generate_nonlinear_data(n_samples=400)
    X_train, X_test, y_train, y_test = train_test_split(
        X_nl, y_nl, test_size=0.2, random_state=42
    )
    
    # Standardize
    X_train_std, params = standardize(X_train, return_params=True)
    X_test_std = (X_test - params["mean"]) / params["scale"]
    
    print(f"\nDataset: y = sin(x) + noise")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    results_nonlinear = {}
    
    # Linear Regression (will fail)
    lr = LinearRegression()
    lr.fit(X_train_std, y_train)
    y_pred = lr.predict(X_test_std)
    results_nonlinear["Linear"] = print_metrics(y_test, y_pred, "Linear Regression")
    
    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train_std, y_train)
    y_pred = dt.predict(X_test_std)
    results_nonlinear["Decision Tree"] = print_metrics(y_test, y_pred, "Decision Tree")
    
    # KNN
    knn = KNNRegressor(n_neighbors=5, weights="distance")
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    results_nonlinear["KNN"] = print_metrics(y_test, y_pred, "KNN Regressor")
    
    # MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        learning_rate=0.01,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train_std, y_train)
    y_pred = mlp.predict(X_test_std)
    results_nonlinear["MLP"] = print_metrics(y_test, y_pred, "MLP Regressor")
    
    # =========================================================================
    # Part 4: Regularization Effect
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 4: Effect of Regularization Strength")
    print("=" * 60)
    
    # Generate data with many features
    np.random.seed(42)
    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + 0.5 * np.random.randn(n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_std, params = standardize(X_train, return_params=True)
    X_test_std = (X_test - params["mean"]) / params["scale"]
    
    print(f"\nHigh-dimensional data: {n_samples} samples, {n_features} features")
    print("\nRidge Regression with different alpha values:")
    print(f"{'Alpha':<10} {'Train R²':>12} {'Test R²':>12} {'Coef Norm':>12}")
    print("-" * 48)
    
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = RidgeRegression(alpha=alpha)
        ridge.fit(X_train_std, y_train)
        
        train_r2 = r2_score(y_train, ridge.predict(X_train_std))
        test_r2 = r2_score(y_test, ridge.predict(X_test_std))
        coef_norm = np.linalg.norm(ridge.coef_)
        
        print(f"{alpha:<10.2f} {train_r2:>12.4f} {test_r2:>12.4f} {coef_norm:>12.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nNonlinear Data Results (R² scores):")
    print(f"{'Model':<20} {'R²':>10}")
    print("-" * 32)
    for model, r2 in sorted(results_nonlinear.items(), key=lambda x: -x[1]):
        print(f"{model:<20} {r2:>10.4f}")
    



if __name__ == "__main__":
    main()