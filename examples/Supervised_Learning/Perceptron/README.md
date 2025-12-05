# Perceptron

This directory contains example code and notes for the Perceptron algorithm
in supervised learning.

## Algorithm

The Perceptron classifies using a linear score $$z = w^\top x + b$$ and a step function $$\hat{y} = \mathrm{sign}(z)$$.
Objective (implicit): find w, b that separate the classes; update on mistakes:
$$w \leftarrow w + \eta\, y_i x_i,\quad b \leftarrow b + \eta\, y_i$$
where $y_i \in \{-1,+1\}$ and $\eta$ is the learning rate.
Key hyperparameters: learning rate $(eta0)$, max_iter, fit_intercept, regularization (penalty, alpha) in sklearn variants.

## Data

This example uses a synthetic 2D dataset created with make_classification for clear visualization.
Data are split via train_test_split; we include a StandardScaler in sklearn pipelines for faster convergence.
