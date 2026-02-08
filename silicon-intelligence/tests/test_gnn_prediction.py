
import os
import torch
import numpy as np
import pytest
import sys
from torch_geometric.data import Data

# Add the parent directory to the path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gnn_model import PpaGNN

MODEL_PATH = './ppa_gnn_model.pt'
NUM_FEATURES = 5  # Based on the features used during training
NUM_TARGETS = 3

@pytest.fixture
def gnn_model():
    """Pytest fixture to load the GNN model."""
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Trained GNN model not found at {MODEL_PATH}. Please run train_gnn.py first.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PpaGNN(num_node_features=NUM_FEATURES, num_targets=NUM_TARGETS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def test_model_file_exists():
    """Test if the trained model file exists."""
    assert os.path.exists(MODEL_PATH), f"Model file should exist at {MODEL_PATH}"

def test_prediction_output_shape_and_type(gnn_model):
    """Test the shape and type of the model's prediction."""
    # Create a sample feature tensor
    sample_features = torch.tensor([[
        100,    # num_nodes
        200,    # num_edges
        0.2,    # density
        150,    # avg_area
        0.08    # avg_power
    ]], dtype=torch.float)

    device = next(gnn_model.parameters()).device
    data = Data(x=sample_features, edge_index=torch.empty((2, 0), dtype=torch.long)).to(device)

    with torch.no_grad():
        prediction = gnn_model(data)
    
    # Check the output type and shape
    assert isinstance(prediction, torch.Tensor), "Prediction should be a torch.Tensor"
    assert prediction.shape == (1, NUM_TARGETS), f"Prediction shape should be (1, {NUM_TARGETS})"

def test_prediction_output_values(gnn_model):
    """Test the validity of the predicted values."""
    # Create another sample feature tensor
    sample_features = torch.tensor([[50, 60, 0.1, 100, 0.05]], dtype=torch.float)

    device = next(gnn_model.parameters()).device
    data = Data(x=sample_features, edge_index=torch.empty((2, 0), dtype=torch.long)).to(device)

    with torch.no_grad():
        prediction = gnn_model(data).numpy()
        
    # We expect one prediction for our single sample
    assert prediction.shape == (1, NUM_TARGETS)
    ppa_values = prediction[0]

    # Check that values are not NaN or infinity
    assert not np.isnan(ppa_values).any(), "PPA values should not be NaN"
    assert np.isfinite(ppa_values).all(), "PPA values should be finite"

    # Check for non-negative Area and Timing
    # We can't guarantee non-negative power without a more advanced model/loss function
    area, power, timing = ppa_values
    assert area >= 0, "Predicted area should be non-negative"
    assert timing >= 0, "Predicted timing should be non-negative"

