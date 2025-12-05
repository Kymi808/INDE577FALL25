# Multilayer Perceptron

This directory contains example code and notes for the Multilayer Perceptron algorithm
in supervised learning.

## Algorithm

A Multilayer Perceptron (MLP) is a feed-forward neural network with one or more hidden layers. Each layer computes $a^{(l)}=\phi(W^{(l)} a^{(l-1)}+b^{(l)})$ with nonlinear activation $\phi$ (e.g., ReLU), and the output layer uses softmax for multiclass.
Objective: minimize cross-entropy between predicted and true labels via backpropagation and gradient-based optimization.
Key hyperparameters: number/size of hidden layers (hidden_layer_sizes), activation (relu, tanh), regularization $(alpha/L2)$, learning rate (learning_rate_init), batch size, and max_iter.

## Data

This example uses scikit-learn’s Digits dataset (64 pixel features, labels 0–9). Data are loaded via sklearn.datasets.load_digits. We standardize features and evaluate accuracy, macro ROC-AUC, and confusion matrices.
