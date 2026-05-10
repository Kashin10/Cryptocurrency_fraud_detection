
import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class GraphSAGEAlgo(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()

        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)

        self.classifier = torch.nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x, edge_index):

        h = self.sage1(x, edge_index)
        h = F.relu(h)

        h = F.dropout(
            h,
            p=0.5,
            training=self.training
        )

        h = self.sage2(h, edge_index)
        h = F.relu(h)

        h = F.normalize(h, p=2, dim=1)

        return self.classifier(h)
