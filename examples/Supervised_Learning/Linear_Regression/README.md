# Linear Regression

This directory contains example code and notes for the Linear Regression algorithm
in supervised learning.

## Algorithm

Linear Regression models the target y as a linear combination of features X:
$$\hat{y} = \beta_0 + \beta^\top X$$
Objective: minimize the sum of squared errors (OLS).
Key hyperparameters (when regularized):
- Ridge (L2): alpha (penalty strength)
- Lasso (L1): alpha (sparsity level)
- (With polynomial basis) degree and interaction options.

## Data

This example uses scikit-learn’s Diabetes dataset (10 numeric features, continuous target).
Data are loaded via sklearn.datasets.load_diabetes. We split into train/test; OLS is fit on raw features; Ridge/Lasso use a StandardScaler inside Pipelines.
