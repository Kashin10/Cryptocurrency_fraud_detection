# Explainable GNN-Based Cryptocurrency Fraud Detection

## Overview

Graph Neural Network based cryptocurrency fraud detection system using GraphSAGE, PyTorch Geometric, and transaction graph analysis.

The project uses graph-based deep learning to identify suspicious cryptocurrency transactions and visualize fraudulent subgraphs.

## Features

- Real GraphSAGE node classification
- Elliptic Bitcoin Dataset preprocessing
- Temporal transaction split evaluation
- PCA + StandardScaler preprocessing pipeline
- Weighted CrossEntropy fraud optimization
- GNNExplainer explainability
- Suspicious subgraph visualization
- Real-time fraud probability scoring
- Interactive Streamlit + Plotly dashboard
- Docker containerization
- Kubernetes deployment manifests

## Tech Stack

### AI/ML
- PyTorch
- PyTorch Geometric
- GraphSAGE
- Scikit-learn
- NumPy
- Pandas

### Visualization
- Streamlit
- Plotly
- NetworkX

### Deployment
- Docker
- Kubernetes

## Dataset

Based on the Elliptic Bitcoin Transaction Dataset.

The preprocessing pipeline includes:
- graph construction
- node index mapping
- PCA dimensionality reduction
- feature scaling
- temporal train/validation/test splits
- transaction edge alignment

## Architecture

Transaction Graph
↓
GraphSAGE Node Embeddings
↓
Fraud Classification
↓
Explainability Layer
↓
Streamlit Dashboard

## Project Structure

- app/ → inference logic
- training/ → training and preprocessing
- models/ → GraphSAGE architecture
- dashboard/ → Streamlit visualization
- deployment/ → Docker + Kubernetes

## How To Run

### Install dependencies

```bash
pip install -r requirements.txt
````

### Train model

```bash
python training/train.py
```

### Run inference

```bash
python app/app.py
```

### Launch dashboard

```bash
streamlit run dashboard/streamlit_dashboard.py
```

## Resume Alignment

This repository directly aligns with the resume project description:

* GraphSAGE node classifier
* PyTorch Geometric workflows
* Elliptic Bitcoin Dataset preprocessing
* Weighted CrossEntropy optimization
* PCA + StandardScaler feature engineering
* Temporal split evaluation for unseen transaction prediction
* GNNExplainer-based explainability
* Fraud subgraph extraction
* Streamlit + Plotly dashboard visualization
* Real-time fraud scoring
* Dockerized deployment
* Kubernetes deployment workflows

## Future Improvements

* Real Elliptic dataset integration
* Live blockchain transaction ingestion
* Advanced GNNExplainer integration
* Multi-GPU distributed training
* Neo4j graph database integration

## Author

Kashin Bhardwaj
