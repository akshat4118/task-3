# Iris Flower Clustering using K-Means and PCA

## Overview
Apply K-Means clustering (k=3) to the Iris dataset, visualize clusters, and compare assignments with known species labels. PCA reduces four feature dimensions to two components for plotting.

## Dataset
The Iris dataset is included with Scikit-learn; no external CSV is required.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook.

## Features
- Standardize numeric features
- K-Means clustering with k=3
- PCA dimensionality reduction
- Cluster and true-species scatter plots
- Crosstab, silhouette score, and Adjusted Rand Index

## Run
Install packages: `pip install numpy pandas matplotlib seaborn scikit-learn`
Open `Iris_Flower_Clustering_KMeans_PCA.ipynb` in Jupyter or VS Code and run all cells, or run `python iris_flower_clustering.py`.

Cluster IDs are arbitrary and do not directly map to species IDs.
