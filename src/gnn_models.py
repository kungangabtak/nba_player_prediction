# src/gnn_models.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class PlayerPerformanceGNN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(PlayerPerformanceGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 8)
        self.lin = nn.Linear(8, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x