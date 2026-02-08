
import sys
import os
import torch

# Ensure we can import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cognitive.physical_risk_oracle import PhysicalRiskOracle
from core.canonical_silicon_graph import CanonicalSiliconGraph

def test_gnn_oracle_inference():
    print("Testing GNN-Enhanced Physical Risk Oracle (Phase 5)...")
    
    oracle = PhysicalRiskOracle()
    
    # Check if GNN is actually active
    if not hasattr(oracle, 'topological_gnn'):
        print("FAILURE: GNN components not initialized in Oracle.")
        sys.exit(1)
        
    print("SUCCESS: GNN model loaded in Oracle.")
    
    # Create a small dummy graph
    graph = CanonicalSiliconGraph()
    # Adding a simple sequence of nodes
    # n1 -> n2 -> n3
    graph.graph.add_node('in_0', node_type='port', area=1.0)
    graph.graph.add_node('gate_0', node_type='cell', area=5.0)
    graph.graph.add_node('out_0', node_type='port', area=1.0)
    graph.graph.add_edge('in_0', 'gate_0')
    graph.graph.add_edge('gate_0', 'out_0')
    
    # Run congestion prediction
    # This will trigger convert_to_pyg_data and gnn inference
    risks = oracle._predict_congestion(graph)
    
    print("\nTopological Risk Scores:")
    for node, score in risks.items():
        print(f"  Node {node}: {score:.4f}")
        
    if len(risks) == 3:
        print("\nSUCCESS: GNN produced per-node risk signals.")
    else:
        print("\nFAILURE: Risk mapping incomplete.")
        sys.exit(1)

if __name__ == "__main__":
    test_gnn_oracle_inference()
