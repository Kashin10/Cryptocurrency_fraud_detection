
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from models.graphsage_model import GraphSAGEAlgo


data = torch.load(
    "saved_models/elliptic_processed.pt",
    weights_only=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = data.to(device)

data.y[data.y == 2] = 0

# ================= MODEL =================
model = GraphSAGEAlgo(
    in_dim=data.num_features,
    hidden_dim=128,
    num_classes=2
).to(device)

# ================= LOSS =================
class_weights = torch.tensor(
    [1.0, 6.0]
).to(device)

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005,
    weight_decay=5e-4
)

# ================= TRAIN =================
def train():

    model.train()

    optimizer.zero_grad()

    out = model(
        data.x,
        data.edge_index
    )

    mask = data.train_mask & (data.y != -1)

    loss = criterion(
        out[mask],
        data.y[mask]
    )

    loss.backward()

    optimizer.step()

    return loss.item()

# ================= EVAL =================
@torch.no_grad()
def evaluate(mask):

    model.eval()

    logits = model(
        data.x,
        data.edge_index
    )

    pred = logits.argmax(dim=1)

    mask = mask & (data.y != -1)

    y_true = data.y[mask].cpu()

    y_pred = pred[mask].cpu()

    acc = (
        (y_pred == y_true)
        .sum()
        .item()
        / mask.sum().item()
    )

    return acc

# ================= LOOP =================
print("Training GraphSAGE Fraud Detector...")

for epoch in range(1, 101):

    loss = train()

    if epoch % 10 == 0:

        train_acc = evaluate(data.train_mask)

        val_acc = evaluate(data.val_mask)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Train {train_acc:.4f} | "
            f"Val {val_acc:.4f}"
        )

# ================= TEST =================
test_acc = evaluate(data.test_mask)

print("\\nFinal Test Accuracy:", test_acc)

torch.save(
    model.state_dict(),
    "saved_models/graphsage_fraud.pt"
)

print("Model saved.")
