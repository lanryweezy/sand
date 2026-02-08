import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class SiliconGNN(torch.nn.Module):
    """
    Graph Neural Network for Silicon PPA Prediction.
    Captures logical and spatial connectivity of standard cells and macros.
    """
    def __init__(self, in_channels, hidden_channels, out_channels=3):
        super(SiliconGNN, self).__init__()
        torch.manual_seed(42)
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Batch Normalization for stability in deep silicon graphs
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Final Prediction Head (MLP)
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = nn.Linear(hidden_channels // 2, out_channels) # Area, Power, Timing
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()

        # 2. Readout layer (Global Pooling)
        # We use both mean and max pooling to capture global design characteristics
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # 3. Final classifier head
        x = self.lin1(x).relu()
        x = self.dropout(x)
        x = self.lin2(x).relu()
        x = self.lin3(x)
        
        return x

from torch_geometric.data import Data

def convert_to_pyg_data(silicon_graph):
    """
    Converts a CanonicalSiliconGraph to a PyTorch Geometric Data object.
    """
    graph = silicon_graph.graph
    
    # Node features: [area, power, delay, capacitance, estimated_congestion, timing_criticality]
    # We'll map node types to integers for the model
    node_type_map = {'cell': 0, 'macro': 1, 'port': 2, 'clock': 3, 'power': 4, 'signal': 5}
    
    x_list = []
    node_to_idx = {}
    for i, (node, attrs) in enumerate(graph.nodes(data=True)):
        node_to_idx[node] = i
        node_type_val = node_type_map.get(attrs.get('node_type').value if hasattr(attrs.get('node_type'), 'value') else attrs.get('node_type'), 5)
        
        # Build node feature vector
        feat = [
            float(node_type_val),
            float(attrs.get('area', 1.0)),
            float(attrs.get('power', 0.0)),
            float(attrs.get('delay', 0.0)),
            float(attrs.get('capacitance', 0.0)),
            float(attrs.get('estimated_congestion', 0.0)),
            float(attrs.get('timing_criticality', 0.0))
        ]
        x_list.append(feat)
    
    x = torch.tensor(x_list, dtype=torch.float)
    
    # Edge index
    edge_index_list = []
    for src, dst, key, attrs in graph.edges(keys=True, data=True):
        edge_index_list.append([node_to_idx[src], node_to_idx[dst]])
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def save_gnn_model(model, path):
    torch.save(model.state_dict(), path)

def load_gnn_model(in_channels, hidden_channels, out_channels, path):
    model = SiliconGNN(in_channels, hidden_channels, out_channels)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
