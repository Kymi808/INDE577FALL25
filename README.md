# CMOR 438 Data Science & Machine Learning 
# Netid: kzz1

## Introduction
This repo consists of everything required in the final project of course INDE577. It consists of a NumPy-first ML package with clear reference implementations and runnable examples for supervised, unsupervised, and sequence models. The repository includes reproducible scripts and notebooks on MNIST and language identification, plus unit tests.

Examples of data ran on MNIST digits and 5 Languages datasets can be found in the example_ml_utilization module. Within it are 2 jupiter notebooks showing analysis on different ml models trained by the data. 

## Repo Structure

data/                          
data/mnist.npz                      MNIST digits 
data/lang_id.npz                    5 Language Dataset

example_ml_utilization/             end-to-end demos & notebooks of ml package
example_ml_utilization/example_language_id.py
example_ml_utilization/example_mnist_classification.py
example_ml_utilization/example_regression.py
example_ml_utilization/example_unsupervised_mnist.py
example_ml_utilization/Language_ID_RNN_GRU_with_Baselines.ipynb
example_ml_utilization/MNIST_Supervised_Unsupervised_Regression.ipynb


examples/                           concept-by-concept snippets compiled of jupiter notebooks on topics covered in class
examples/Supervised_Learning/       KNN, Trees, Ensembles, Linear/Logistic, MLP
examples/Unsupervised_Learning/     PCA, KMeans, DBSCAN, community detection

src/
src/rice_ml/
src/rice_ml/processing/             preprocessing & post-processing
src/rice_ml/supervised_learning/    KNN, Trees, Ensembles, Linear/Logistic, MLP, RNN/GRU
src/rice_ml/unsupervised_learning/  PCA, KMeans, DBSCAN, community detection


tests/
tests/unit/                         pytest unit tests per module consists of unit tests for all models in ml package

pyproject.toml                      build config 
Makefile                        
README.md

# Unit Testing
To perform unit tests:
Make test:
PYTHONPATH=src pytest tests/unit/ -v

# Cache Management
To clean cache 
Run make clean-pyc

# Python & Dependencies
- Python 3.9+ recommended (NumPy-first implementations).
- Core: numpy
- Examples/plots/notebooks: matplotlib, jupyter (optional)
- Tests: pytest