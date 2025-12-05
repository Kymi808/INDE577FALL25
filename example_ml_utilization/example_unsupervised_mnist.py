"""
Example 2: Unsupervised Learning on MNIST
==========================================

This example demonstrates unsupervised learning techniques from
the rice_ml package on the MNIST dataset.

Algorithms demonstrated:
- PCA (Principal Component Analysis) for visualization
- K-Means Clustering
- DBSCAN Clustering

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
    # Unsupervised Learning
    PCA,
    KMeans,
    DBSCAN,
    # Metrics (for evaluating clustering with ground truth)
    accuracy_score,
)


def load_mnist(data_path="data/mnist.npz"):
    """Load MNIST dataset from npz file."""
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
        raise FileNotFoundError(f"Could not find mnist.npz. Tried: {paths_to_try}")
    
    with np.load(data_path) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        test_images = data["test_images"]
        test_labels = data["test_labels"]
    
    return train_images, train_labels, test_images, test_labels


def cluster_accuracy(y_true, cluster_labels):
    """
    Compute clustering accuracy by finding best mapping from clusters to true labels.
    
    This finds the permutation of cluster labels that maximizes accuracy.
    """
    from collections import Counter
    
    # For each cluster, find the most common true label
    cluster_to_label = {}
    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])  # Ignore noise (-1)
    
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        true_labels_in_cluster = y_true[mask]
        if len(true_labels_in_cluster) > 0:
            most_common = Counter(true_labels_in_cluster).most_common(1)[0][0]
            cluster_to_label[cluster] = most_common
    
    # Map predictions
    y_pred_mapped = np.array([
        cluster_to_label.get(c, -1) for c in cluster_labels
    ])
    
    # Only evaluate non-noise points
    valid = cluster_labels >= 0
    if valid.sum() == 0:
        return 0.0
    
    return accuracy_score(y_true[valid], y_pred_mapped[valid])


def main():
    print("=" * 60)
    print("Unsupervised Learning on MNIST with rice_ml")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Use a subset for faster computation
    print("\nSubsampling 3000 samples for experimentation...")
    np.random.seed(42)
    idx = np.random.choice(len(X_train), size=3000, replace=False)
    X = X_train[idx]
    y = y_train[idx]
    print(f"Using {len(X)} samples")
    
    # Standardize
    print("\nStandardizing features...")
    X_std = standardize(X)
    
    # =========================================================================
    # 1. PCA Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. PCA - Principal Component Analysis")
    print("=" * 60)
    
    # Fit PCA with all components first
    pca_full = PCA(n_components=None)
    pca_full.fit(X_std)
    
    # Cumulative explained variance
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    
    print("\nExplained Variance by Number of Components:")
    for n in [2, 10, 50, 100, 200]:
        if n <= len(cumsum):
            print(f"  {n:>3} components: {cumsum[n-1]:>6.2%}")
    
    # Find number of components for 95% variance
    n_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"\nComponents needed for 95% variance: {n_95}")
    
    # Reduce to 2D for visualization info
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_std)
    
    print(f"\n2D Projection (first 2 PCs):")
    print(f"  Explained variance: {sum(pca_2d.explained_variance_ratio_):.2%}")
    print(f"  X_2d shape: {X_2d.shape}")
    
    # Show spread of each digit in 2D space
    print("\n  Digit centroids in 2D PCA space:")
    print(f"  {'Digit':<6} {'PC1':>10} {'PC2':>10}")
    print("  " + "-" * 28)
    for digit in range(10):
        mask = y == digit
        centroid = X_2d[mask].mean(axis=0)
        print(f"  {digit:<6} {centroid[0]:>10.2f} {centroid[1]:>10.2f}")
    
    # =========================================================================
    # 2. K-Means Clustering
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. K-Means Clustering")
    print("=" * 60)
    
    # Use PCA-reduced data for faster clustering
    pca_50 = PCA(n_components=50)
    X_pca = pca_50.fit_transform(X_std)
    print(f"\nUsing 50 PCA components ({sum(pca_50.explained_variance_ratio_):.1%} variance)")
    
    # Try K-Means with 10 clusters (one per digit)
    print("\nFitting K-Means with k=10...")
    kmeans = KMeans(n_clusters=10, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(X_pca)
    
    print(f"  Iterations: {kmeans.n_iter_}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    
    # Evaluate clustering quality
    cluster_acc = cluster_accuracy(y, kmeans.labels_)
    print(f"  Cluster Accuracy: {cluster_acc:.2%}")
    
    # Show cluster distribution
    print("\n  Cluster sizes:")
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    for cluster, count in zip(unique, counts):
        # Find dominant digit in this cluster
        mask = kmeans.labels_ == cluster
        digit_counts = np.bincount(y[mask], minlength=10)
        dominant = np.argmax(digit_counts)
        purity = digit_counts[dominant] / count
        print(f"    Cluster {cluster}: {count:>4} samples, "
              f"dominant digit={dominant}, purity={purity:.1%}")
    
    # Try different k values
    print("\n  Trying different k values:")
    print(f"  {'k':<5} {'Inertia':>12} {'Cluster Acc':>12}")
    print("  " + "-" * 30)
    for k in [5, 10, 15, 20]:
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        km.fit(X_pca)
        acc = cluster_accuracy(y, km.labels_)
        print(f"  {k:<5} {km.inertia_:>12.0f} {acc:>12.2%}")
    
    # =========================================================================
    # 3. DBSCAN Clustering
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. DBSCAN Clustering")
    print("=" * 60)
    
    # DBSCAN works better on lower-dimensional data
    pca_10 = PCA(n_components=10)
    X_pca_10 = pca_10.fit_transform(X_std)
    print(f"\nUsing 10 PCA components ({sum(pca_10.explained_variance_ratio_):.1%} variance)")
    
    # Use smaller subset for DBSCAN (it's O(n²) in worst case)
    X_db = X_pca_10[:1000]
    y_db = y[:1000]
    
    print(f"Using {len(X_db)} samples for DBSCAN")
    
    # Try different eps values
    print("\n  Trying different eps values (min_samples=5):")
    print(f"  {'eps':<6} {'Clusters':>10} {'Noise %':>10} {'Coverage':>10}")
    print("  " + "-" * 38)
    
    for eps in [1.0, 2.0, 3.0, 4.0, 5.0]:
        db = DBSCAN(eps=eps, min_samples=5)
        db.fit(X_db)
        
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_noise = np.sum(db.labels_ == -1)
        noise_pct = n_noise / len(X_db)
        coverage = 1 - noise_pct
        
        print(f"  {eps:<6.1f} {n_clusters:>10} {noise_pct:>10.1%} {coverage:>10.1%}")
    
    # Detailed analysis with best eps
    print("\n  Detailed DBSCAN (eps=3.0, min_samples=5):")
    db = DBSCAN(eps=3.0, min_samples=5)
    db.fit(X_db)
    
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = np.sum(db.labels_ == -1)
    
    print(f"    Clusters found: {n_clusters}")
    print(f"    Noise points: {n_noise} ({n_noise/len(X_db):.1%})")
    print(f"    Core samples: {len(db.core_sample_indices_)}")
    
    if n_clusters > 0:
        acc = cluster_accuracy(y_db, db.labels_)
        print(f"    Cluster Accuracy (non-noise): {acc:.2%}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)


if __name__ == "__main__":
    main()