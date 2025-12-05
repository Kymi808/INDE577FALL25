# DBSCAN

This directory contains example code and notes for the DBSCAN algorithm
in unsupervised learning.

## Algorithm

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points whose neighborhoods (radius eps) contain at least min_samples points, forming core points; points reachable from cores are clustered; others are labeled noise.
Objective: uncover dense regions without specifying the number of clusters; treat sparse regions as outliers.
Key hyperparameters: eps, min_samples, metric (e.g., Euclidean), and leaf_size (k-NN efficiency).

## Data

Input is a numeric feature matrix we standardized with StandardScaler.
This notebook uses a synthetic 2-D dataset (moons + blob + noise) to demonstrate DBSCAN’s strength on non-convex shapes and outlier detection.
