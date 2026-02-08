
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np

class SiliconGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for loading silicon design graphs from our JSON file.
    """
    def __init__(self, json_path, transform=None, pre_transform=None):
        super().__init__()
        self.json_path = json_path
        self.transform = transform
        self.pre_transform = pre_transform
        self.samples = self._load_samples()

    def _load_samples(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Filter out samples with missing PPA labels
        valid_samples = []
        for sample in data['samples']:
            if sample.get('labels', {}).get('ppa'):
                 # Ensure all PPA values are present and not None
                ppa = sample['labels']['ppa']
                if ppa.get('area_um2') is not None and ppa.get('power_mw') is not None and ppa.get('timing_ns') is not None:
                    valid_samples.append(sample)
        print(f"Loaded {len(valid_samples)} valid samples from {self.json_path}")
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Features: Use graph statistics for now. A real implementation would use node-level features.
        # We'll create a single "super-node" representing the whole graph for simplicity.
        features = sample['features']
        node_features = torch.tensor([[
            features.get('num_nodes', 0),
            features.get('num_edges', 0),
            features.get('density', 0),
            features.get('avg_area', 0),
            features.get('avg_power', 0)
        ]], dtype=torch.float)

        # Edges: Since we have a single super-node, there are no edges.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Labels (PPA)
        ppa = sample['labels']['ppa']
        labels = torch.tensor([
            ppa.get('area_um2', 0),
            ppa.get('power_mw', 0),
            ppa.get('timing_ns', 0)
        ], dtype=torch.float)

        data = Data(x=node_features, edge_index=edge_index, y=labels)
        
        # Apply transformations if any
        if self.transform:
            data = self.transform(data)
            
        return data

import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool

class PpaGNN(torch.nn.Module):
    """
    A simple Graph Neural Network to predict PPA from graph-level features.
    """
    def __init__(self, num_node_features, num_targets=3):
        super(PpaGNN, self).__init__()
        self.fc1 = Linear(num_node_features, 64)
        self.fc2 = Linear(64, 128)
        self.fc3 = Linear(128, 64)
        self.output_layer = Linear(64, num_targets)

    def forward(self, data):
        # Since we are using global graph features, we pass them through fully connected layers.
        # This is not a "true" GNN yet but sets up the architecture.
        # The 'data' object still holds graph info, which we'll use later for node features.
        x = data.x

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # In a real GNN, we would pool node embeddings here.
        # Since we have one "super-node", this is equivalent to just using its features.
        
        output = self.output_layer(x)
        return output

    def __str__(self):
        return self.__class__.__name__
