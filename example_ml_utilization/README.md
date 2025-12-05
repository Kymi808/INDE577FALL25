## APPLIED EXAMPLES

Two end-to-end, classroom-friendly notebooks that exercise your rice_ml package across supervised, unsupervised, and regression workflows—complete with notes and plots.
- Language_ID_RNN_GRU_with_Baselines_plots.ipynb
- MNIST_Supervised_Unsupervised_Regression_rice_ml.ipynb

These mirror the Python scripts you shared, but add rich Markdown, diagnostics, and Matplotlib visualizations so you can demo, grade, or iterate quickly.

⸻

1) What’s inside each notebook?

A. Language_ID_RNN_GRU_with_Baselines_plots.ipynb

Sequence classification on a 5-language character-level dataset.

Models
- Baselines: Bag-of-Characters → LogisticRegression, RandomForestClassifier
- Sequence models: RNNClassifier, GRUClassifier (character one-hot inputs)

Visuals & Diagnostics
- Bucket/sequence-length distribution
- Top-k character frequency bar chart
- Training curves (loss vs. epoch) for RNN & GRU
- Accuracy comparison bar chart (baselines vs. sequence models)
- Confusion matrix heatmap for the best seq model


B. MNIST_Supervised_Unsupervised_Regression_rice_ml.ipynb

A single notebook combining classification, unsupervised analysis, and regression demos on MNIST + synthetic data.

Supervised (subsampled to 5k train / 1k test for speed)
- KNNClassifier, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, MLPClassifier
- Standardization + results table + confusion matrix for the top model
- Accuracy bar chart

Unsupervised
- PCA: explained variance curve + 2D scatter (centroids per digit)
- KMeans: k-sweep (k∈{5,10,15,20}) inertia & cluster-accuracy plots
- DBSCAN: eps-sweep table/plot + summary with noise/coverage

Regression
- Linear data: OLS (closed-form & GD), Ridge, Lasso (sparse demo)
- Nonlinear data: sin curve → Linear vs Tree/KNN/MLP
- Plots: residuals, predictions vs. truth, Ridge α vs R²