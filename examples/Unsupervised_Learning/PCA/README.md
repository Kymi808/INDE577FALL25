# PCA

This directory contains example code and notes for the PCA algorithm
in unsupervised learning.

## Algorithm

Principal Component Analysis (PCA) finds orthogonal directions (principal components) capturing the greatest variance in the data.
Objective: minimize reconstruction error in a k-dimensional subspace (equivalently, maximize retained variance).
Key hyperparameters: n_components (int or variance fraction like 0.95), svd_solver (full, randomized), whiten (for unit-variance PCs).

## Data

Input is a numeric feature matrix. This example uses the Wine dataset (13 features), standardized with StandardScaler.
We report explained variance, cumulative EVR, 2D PC scatter, loadings/biplot, and reconstruction error vs. components.
