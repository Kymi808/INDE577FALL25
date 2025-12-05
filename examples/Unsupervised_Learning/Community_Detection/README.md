# Community Detection

This directory contains example code and notes for the Community Detection algorithm
in unsupervised learning.

## Algorithm

Community detection groups nodes in a graph so edges are dense within groups and sparse across groups.
Typical methods include modularity maximization (e.g., greedy CNM, Louvain/Leiden), spectral clustering (eigenvectors of Laplacian/adjacency), and hierarchical approaches (e.g., Girvan–Newman using edge betweenness).
Objective: maximize intra-community connectivity (e.g., modularity) or minimize cut-based criteria (e.g., conductance).
Key hyperparameters: number of communities k (spectral), resolution/γ (Louvain/Leiden), stopping depth (hierarchical), and random seeds for reproducibility.

## Data

Input is a graph G=(V,E). This example uses Zachary’s Karate Club, loaded via networkx.karate_club_graph().
We evaluate with modularity and mean conductance; where ground truth exists (Karate), we also report ARI and NMI.
