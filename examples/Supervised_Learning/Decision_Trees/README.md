# Decision Trees

This directory contains example code and notes for the Decision Trees algorithm
in supervised learning.

## Algorithm

Algorithm
Decision Trees recursively partition the feature space using axis-aligned thresholds to minimize node impurity (e.g., Gini, entropy, log loss). Objective: produce a tree that generalizes while controlling complexity. Key hyperparameters: criterion, max_depth, min_samples_leaf, min_samples_split, max_features, and pruning via ccp_alpha.

## Data

This example uses Iris (4 numeric features: sepal length/width, petal length/width; labels: setosa, versicolor, virginica). Data loaded with sklearn.datasets.load_iris. No scaling required; imputation included for robustness.
