
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from typing import Dict, List, Any

class TopologicalHotspotGNN(torch.nn.Module):
    """
    Advanced GNN using Graph Attention (GAT) to identify congestion hotspots 
    directly from RTL netlist topology.
    """
    def __init__(self, in_channels: int, hidden_channels: int):
        super(TopologicalHotspotGNN, self).__init__()
        
        # We use Multi-Head Attention to capture complex signal paths
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        
        # Node-level risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_channels * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Probability of being a hotspot
        )

    def forward(self, x, edge_index):
        # 1. Message Passing
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # 2. Per-node risk score
        risk_scores = self.risk_head(x)
        return risk_scores

def predict_node_risks(gnn_model, pyg_data) -> Dict[int, float]:
    """Helper to run inference and return a mapping of node index -> risk score"""
    gnn_model.eval()
    with torch.no_grad():
        risks = gnn_model(pyg_data.x, pyg_data.edge_index)
        return {i: float(r) for i, r in enumerate(risks)}
