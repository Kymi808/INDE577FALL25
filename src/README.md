# The Source Code for Rice ML

What’s included

rice_ml.processing/

Common helpers for data prep (e.g., scaling, splitting, encoding, batching).

- Put reusable, non-model-specific utilities here.

rice_ml.supervised_learning/

Implementations of supervised ML models and supporting utilities:
- linear_regression.py — Linear regression models (typically fit via normal equation or gradient descent).
- logistic_regression.py — Logistic regression for binary (and possibly multiclass) classification.
- perceptron.py — Classic perceptron classifier.
- multilayer_perceptron.py — Feed-forward neural network (MLP).
- rnn.py — Recurrent neural network components/models.
- knn.py — k-Nearest Neighbors classifier/regressor.
- distance_metrics.py — Distance functions (e.g., Euclidean, Manhattan, cosine) used by KNN and clustering.
- decision_trees.py — Decision tree classifier (and/or general tree logic).
- regression_trees.py — Regression tree variant.
- ensemble_methods.py — Ensemble techniques (e.g., bagging, boosting, random forests—depending on implementation).

rice_ml.unsupervised_learning/

Unsupervised methods for clustering, dimensionality reduction, and graph structure:
- k_means_clustering.py — K-Means clustering.
- dbscan.py — DBSCAN clustering.
- pca.py — Principal Component Analysis for dimensionality reduction.
- community_detection.py — Graph community detection utilities/algorithms.