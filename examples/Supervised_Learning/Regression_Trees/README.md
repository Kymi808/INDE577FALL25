# Regression Trees

This directory contains example code and notes for the Regression Trees algorithm
in supervised learning.

## Algorithm

Core idea: Recursively partition the feature space with binary splits. At each node, choose the feature and threshold that minimize the weighted mean squared error (MSE)** of the two child nodes. Each leaf predicts the mean target value of the samples in that leaf.
Objective: Minimize post-split impurity (variance/MSE), i.e., maximize variance reduction.
Key hyperparameters:
- `max_depth` — maximum depth of the tree (capacity/overfitting control)
- `min_samples_split` — minimum samples required to consider a split
- `min_samples_leaf` — minimum samples required in a leaf
- `max_features` — number of features to evaluate per split (if `None`, use all)
- `min_impurity_decrease` — minimum impurity reduction to accept a split
- `ccp_alpha` — cost-complexity pruning strength (post-pruning)
- `random_state` — reproducibility for tie-breaks/heuristics


## Data

This example notebook uses:
- California Housing: median house value prediction with 8 numeric features  
  and fallbacks to Diabetes regression if fetch is unavailable

Loading and preprocessing:
- Loaded via `sklearn.datasets.fetch_california_housing(as_frame=True)`  
- Split with `train_test_split`.
- Imputation: `SimpleImputer(strategy="median")` for robustness.  

