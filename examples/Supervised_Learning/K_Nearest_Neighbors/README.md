# K Nearest Neighbors

This directory contains example code and notes for the K Nearest Neighbors algorithm
in supervised learning.

## Algorithm

K Nearest Neighbors (KNN) classifies a point by the majority (or distance-weighted) vote among its k nearest training samples under a distance metric (e.g., Euclidean).
Objective: approximate class boundaries by local neighborhoods in feature space.
Key hyperparameters: n_neighbors (k), weights (uniform, distance), metric (minkowski with p=1/2, or others), and leaf_size / algorithm (for large datasets).

## Data

This example uses the Wine dataset (13 numeric features; 3 classes). Data are loaded from sklearn.datasets.load_wine. Feature scaling with StandardScaler is applied before KNN.
