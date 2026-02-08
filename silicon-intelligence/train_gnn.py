
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import numpy as np
import os
import sys

# Ensure the module path is correct
sys.path.append(os.path.dirname(__file__))

from gnn_model import SiliconGraphDataset, PpaGNN

def train(model, device, train_loader, optimizer, epoch):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Reshape the flattened labels to match the output shape
        y = data.y.view(-1, 3)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def validate(model, device, loader):
    """Validation loop"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            # Reshape the flattened labels to match the output shape
            y = data.y.view(-1, 3)
            loss = F.mse_loss(output, y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def main():
    print("="*60)
    print("STARTING GNN MODEL TRAINING")
    print("="*60)
    
    # --- Configuration ---
    json_path = './training_dataset.json'
    model_save_path = './ppa_gnn_model.pt'
    epochs = 50
    batch_size = 4 
    learning_rate = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset and Dataloaders ---
    if not os.path.exists(json_path):
        print(f"ERROR: Dataset not found at {json_path}")
        print("Please run 'prepare_training_data.py' first.")
        return
        
    dataset = SiliconGraphDataset(json_path=json_path)
    
    if len(dataset) == 0:
        print("ERROR: No valid samples found in the dataset. Exiting.")
        return

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Dataset loaded: {len(dataset)} samples ({train_size} training, {val_size} validation)")

    # --- Model, Optimizer, Loss ---
    # The number of features is based on our 'super-node' approach in the dataset class
    num_node_features = dataset[0].num_node_features
    model = PpaGNN(num_node_features=num_node_features, num_targets=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model: {model}")
    print(f"Number of features: {num_node_features}")

    # --- Training Loop ---
    print("\nStarting training...")
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = validate(model, device, val_loader)
        
        print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'  -> Model saved to {model_save_path}')

    print("\n" + "="*60)
    print("GNN MODEL TRAINING COMPLETE")
    print(f"Final model saved to {model_save_path} with best validation loss: {best_val_loss:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
