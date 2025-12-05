"""
Unsupervised Learning algorithms.

This subpackage contains implementations of unsupervised learning algorithms
including clustering and dimensionality reduction.

Classes
-------
KMeans
    K-Means clustering.
DBSCAN
    Density-Based Spatial Clustering of Applications with Noise.
PCA
    Principal Component Analysis.
SpectralClustering
    Spectral clustering for community detection.
LabelPropagation
    Label propagation algorithm for community detection.
"""

from rice_ml.unsupervised_learning.k_means_clustering import KMeans
from rice_ml.unsupervised_learning.dbscan import DBSCAN
from rice_ml.unsupervised_learning.pca import PCA
from rice_ml.unsupervised_learning.community_detection import (
    SpectralClustering,
    LabelPropagation,
)

__all__ = [
    'KMeans',
    'DBSCAN',
    'PCA',
    'SpectralClustering',
    'LabelPropagation',
]