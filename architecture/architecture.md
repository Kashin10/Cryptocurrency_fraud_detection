# System Architecture

## Overview

This project detects fraudulent cryptocurrency transactions using a Graph Neural Network (GraphSAGE) trained on the Elliptic Bitcoin Dataset.

## Pipeline

1. Load transaction features, labels, and transaction graph.
2. Build a directed transaction network.
3. Standardize features using StandardScaler.
4. Reduce dimensionality using PCA.
5. Convert data into a PyTorch Geometric graph.
6. Train a GraphSAGE model for node classification.
7. Generate explanations using GNNExplainer.
8. Visualize results using Streamlit.

## Components

### Preprocessing
- Missing value handling
- Feature scaling
- PCA dimensionality reduction
- Graph construction

### GraphSAGE Model
- Two GraphSAGE convolution layers
- ReLU activations
- Feature normalization
- Fully connected classifier

### Explainability
- GNNExplainer
- Feature importance
- Suspicious transaction link identification

### Dashboard
- Fraud probability visualization
- Interactive graph exploration
- Explainability reports
