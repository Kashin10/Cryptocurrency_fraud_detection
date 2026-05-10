
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import numpy as np

from torch_geometric.data import Data

import os


# ---------- LOAD DATA ----------
features = pd.read_csv(
    "txs_features.txt",
    header=None,
    skiprows=1,
    low_memory=False
)

tx_ids = features.iloc[:, 0]
time_steps = features.iloc[:, 1]

X = features.iloc[:, 2:].astype(float)

edges = pd.read_csv("txs_edgelist.txt")
labels = pd.read_csv("txs_classes.txt")

# ---------- FIX LABELS ----------
labels['class'] = labels['class'].replace({
    3: -1,
    1: 1,
    2: 2
})

# ---------- BUILD GRAPH ----------
G = nx.from_pandas_edgelist(
    edges,
    'txId1',
    'txId2',
    create_using=nx.DiGraph()
)

G.add_nodes_from(tx_ids.tolist())

# ---------- FEATURE PROCESSING ----------
X = X.fillna(0)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=50)

X_reduced = pca.fit_transform(X_scaled)

# ---------- NODE INDEX ----------
node_list = list(G.nodes())

node_to_index = {
    node: i
    for i, node in enumerate(node_list)
}

# ---------- EDGE INDEX ----------
edge_index = [
    [node_to_index[u], node_to_index[v]]
    for u, v in G.edges()
]

edge_index = torch.tensor(
    edge_index,
    dtype=torch.long
).t().contiguous()

# ---------- ALIGN FEATURES ----------
feature_dict = dict(zip(tx_ids, X_reduced))

num_nodes = len(node_list)

feat_dim = X_reduced.shape[1]

X_graph = np.zeros((num_nodes, feat_dim))

for n in node_list:
    if n in feature_dict:
        X_graph[node_to_index[n]] = feature_dict[n]

x = torch.tensor(X_graph, dtype=torch.float)

# ---------- ALIGN LABELS ----------
label_dict = dict(zip(labels['txId'], labels['class']))

y = torch.full((num_nodes,), -1)

for n in node_list:
    if n in label_dict:
        y[node_to_index[n]] = label_dict[n]

y = y.long()

# ---------- TEMPORAL SPLIT ----------
time_dict = dict(zip(tx_ids, time_steps))

time_vec = torch.zeros(num_nodes)

for n in node_list:
    if n in time_dict:
        time_vec[node_to_index[n]] = time_dict[n]

train_mask = (time_vec <= 34) & ((y == 1) | (y == 2))

val_mask = (
    (time_vec > 34)
    & (time_vec <= 39)
    & ((y == 1) | (y == 2))
)

test_mask = (
    (time_vec > 39)
    & ((y == 1) | (y == 2))
)

# ---------- DATA OBJECT ----------
data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    time=time_vec,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)

os.makedirs("saved_models", exist_ok=True)

torch.save(
    data,
    "saved_models/elliptic_processed.pt"
)

torch.save(
    node_list,
    "saved_models/node_list.pt"
)

print("Preprocessing COMPLETE")
