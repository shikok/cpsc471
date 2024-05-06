# Graph Neural Network Explainability

This repository contains the implementation and experiments for our research on enhancing the explainability of Graph Neural Networks (GNNs) through a novel node importance quantification method. Our approach focuses on evaluating the proximity of node embeddings (NE) to graph embeddings (GE) in binary classification tasks, aiming to provide deeper insights into how GNNs make decisions.

## Overview

Graph Neural Networks have shown exceptional performance in graph-level classification tasks by aggregating node representations into a graph representation using various pooling functions. However, understanding individual node contributions to the graph-level prediction remains a challenge. Our research introduces a novel mathematical framework called the Inverse Distance Weighted Score (IDWS) to quantify node importance based on their proximity to the graph embeddings, enhancing model-level explanations.

## Method

## Repository Structure

### Python Scripts

- `train.py`: Contains all the basic logic for training the GNN models.
- `train_with_optuna.py`: Functions for training models with hyperparameter optimization using Optuna.
- `evaluation.py`: Global evaluation metrics used to assess the performance and explainability of the models.
- `node_importance.py`: Implementation of the node importance quantification method (IDWS).
- `model.py`: Implementation of the GNN architecture used in our experiments.
- `datasets.py`: Factory module for handling dataset loading and preprocessing.
- `baselines.py`: Implementation of baseline methods for comparison.

### Jupyter Notebooks

- `BACE_notebook.ipynb`: Notebook containing results and visualizations for the BACE dataset.
- `BA2Motif_notebook.ipynb`: Notebook featuring results and visualizations for the BA2Motif dataset.
