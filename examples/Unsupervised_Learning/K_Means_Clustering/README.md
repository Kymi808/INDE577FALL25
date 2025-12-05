# K Means Clustering

This directory contains example code and notes for the K Means Clustering algorithm
in unsupervised learning.

## Algorithm

K-Means partitions data into k clusters by iterating two steps:
(1) assign each point to its nearest centroid; (2) update each centroid to the mean of its assigned points.
Objective: minimize total inertia (sum of squared distances to centroids).
Key hyperparameters: n_clusters (k), init (k-means++ recommended), n_init, max_iter, tol, random_state.

## Data

Input is a numeric feature matrix. In the notebook we standardize with StandardScaler and use a synthetic 2-D blobs dataset (with a few outliers) to visualize clusters, compare Elbow and Silhouette, and validate with ARI/NMI against known labels.
