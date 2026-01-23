import torch
from networks.graph_neural_network import SiliconGNN, convert_to_pyg_data
from core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType
from ml_prediction_models import DesignPPAPredictor

def test_gnn_integration():
    print("Testing GNN Integration in DesignPPAPredictor...")
    
    # 1. Create a dummy silicon graph
    graph_manager = CanonicalSiliconGraph()
    # Add some nodes
    graph_manager.graph.add_node("n1", node_type=NodeType.CELL, area=1.5, power=0.02)
    graph_manager.graph.add_node("n2", node_type=NodeType.CELL, area=1.2, power=0.01)
    graph_manager.graph.add_node("n3", node_type=NodeType.PORT, area=0.0, power=0.0)
    # Add some edges
    graph_manager.graph.add_edge("n1", "n2", edge_type="connection")
    graph_manager.graph.add_edge("n2", "n3", edge_type="connection")
    
    # 2. Setup Predictor with GNN
    predictor = DesignPPAPredictor()
    predictor.enable_gnn(in_channels=7, hidden_channels=32)
    
    # 3. Dummy training data
    design_records = [
        {
            'features': {'node_count': 3, 'edge_count': 2}, # Flat features for legacy fallback
            'labels': {'actual_area': 2.7, 'actual_power': 0.03, 'actual_timing': 0.15, 'drc_violations': 0},
            'silicon_graph': graph_manager,
            'design_name': 'dummy_design'
        }
    ]
    
    # 4. Train
    print("Starting training...")
    predictor.train(design_records)
    
    # 5. Predict
    print("\nStarting prediction...")
    test_record = {'silicon_graph': graph_manager, 'node_count': 3, 'edge_count': 2}
    predictions = predictor.predict(test_record)
    
    print("\nResults:")
    for k, v in predictions.items():
        print(f"  {k}: {v:.4f}")

    if all(k in predictions for k in ['area', 'power', 'timing']):
        print("\n✅ GNN Integration Successful!")
    else:
        print("\n❌ GNN Integration Failed!")

if __name__ == "__main__":
    test_gnn_integration()
