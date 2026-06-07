# Results

## Dataset

Elliptic Bitcoin Dataset

## Training Strategy

- Temporal train-validation-test split
- Weighted CrossEntropy Loss
- Adam Optimizer

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Explainability

The trained model is analyzed using GNNExplainer.

Outputs include:

- Top influential node features
- Important transaction relationships
- Fraud probability estimates

## Dashboard

The Streamlit dashboard allows users to:

- Select transactions
- View fraud probabilities
- Inspect important graph connections
- Generate forensic summaries
