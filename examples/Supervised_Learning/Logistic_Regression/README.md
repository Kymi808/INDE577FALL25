# Logistic Regression

This directory contains example code and notes for the Logistic Regression algorithm
in supervised learning.

## Algorithm

Logistic Regression predicts the probability of the positive class using a logistic/sigmoid link on a linear combination of features.
Objective: minimize log-loss (negative log-likelihood) with optional L1/L2 penalties.
Key hyperparameters: penalty (l2, l1), C (inverse regularization), solver (liblinear, lbfgs, saga), class_weight, and max_iter.

## Data

This example uses Breast Cancer Wisconsin (30 numeric features; label: malignant/benign).
Loaded via sklearn.datasets.load_breast_cancer. We apply a StandardScaler before Logistic Regression.
