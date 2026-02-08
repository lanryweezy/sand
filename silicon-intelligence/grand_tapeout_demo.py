"""
Grand Tape-Out Demonstration - Silicon Intelligence (v3.0 ELITE)
Unified Autonomous Design Pipeline: RTL -> AST -> GNN -> RL -> PDK -> TCL -> Telemetry.
Branded by Street Heart Technologies.
"""

import os
import time
import torch
from torch_geometric.data import Data

from core.rtl_transformer import RTLTransformer
from core.pdk_manager import PDKManager
from core.tcl_generator import TCLGeneratorFactory
from core.rl_environment import SiliconRLEnvironment
from networks.rl_trainer import SiliconRLTrainer
from core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType
from utils.authority_dashboard import AuthorityDashboard
from gnn_model import PpaGNN # Import our GNN model

def predict_ppa_with_gnn(model_path, features_tensor):
    """Loads the GNN model and predicts PPA."""
    if not os.path.exists(model_path):
        print("    -> WARNING: GNN model not found. Skipping PPA prediction.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # The feature count must match what the model was trained on
    model = PpaGNN(num_node_features=features_tensor.shape[1], num_targets=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create a Data object, similar to training
    data = Data(x=features_tensor, edge_index=torch.empty((2, 0), dtype=torch.long)).to(device)

    with torch.no_grad():
        prediction = model(data)
    
    return prediction.cpu().numpy()


def run_grand_tapeout_demo():
    print("="*60)
    print("STARTING SILICON INTELLIGENCE GRAND TAPE-OUT DEMO")
    print("="*60)
    
    dashboard = AuthorityDashboard()
    transformer = RTLTransformer()
    pdk_mgr = PDKManager()
    
    # 1. RTL INGESTION & AST REFACTORING
    print("\n[STEP 1] Autonomous RTL Optimization (AST)...")
    src_v = "final_boss.v"
    with open(src_v, "w") as f:
        f.write('''
module final_boss (
    input clk,
    input [63:0] data_in_a,
    input [63:0] data_in_b,
    output [63:0] result
);
    wire [63:0] high_latency_net;
    assign high_latency_net = data_in_a * data_in_b; // Mult-cycle logic
    assign result = high_latency_net;
endmodule
''')
    
    ast = transformer.parse_rtl(src_v)
    ast, reg_name = transformer.add_pipeline_stage(ast, 'final_boss', 'high_latency_net')
    ast = transformer.update_signal_sinks(ast, 'final_boss', 'high_latency_net', reg_name)
    optimized_rtl = transformer.generate_verilog(ast)
    
    with open("final_boss_pipelined.v", "w") as f:
        f.write(optimized_rtl)
    
    dashboard.log_optimization("AST_PIPELINE", "final_boss.high_latency_net", "SUCCESS")
    
    # 2. GNN PPA PREDICTION
    print("\n[STEP 2] GNN Topological PPA Prediction...")
    
    # This is a placeholder. A real implementation would extract these from the design graph.
    # We use values that are plausible for a small design to get a realistic prediction.
    # Features: num_nodes, num_edges, density, avg_area, avg_power
    design_features = torch.tensor([[
        50, # num_nodes
        60, # num_edges
        0.1, # density
        100, # avg_area
        0.05 # avg_power
    ]], dtype=torch.float)

    ppa_prediction = predict_ppa_with_gnn('./ppa_gnn_model.pt', design_features)
    
    if ppa_prediction is not None:
        area, power, timing = ppa_prediction[0]
        prediction_summary = f"Area: {area:.2f}um^2, Power: {power:.3f}mW, Timing: {timing:.3f}ns"
        print(f"    -> GNN Prediction: {prediction_summary}")
        dashboard.log_optimization("GNN_INFERENCE", "final_boss_pipelined", prediction_summary)
    else:
        # Mocking GNN inference results for the dashboard
        dashboard.log_optimization("GNN_INFERENCE", "final_boss_pipelined", "SKIPPED (model not found)")

    
    # 3. RL AGENT STRATEGY TUNING
    print("\n[STEP 3] RL Policy Optimization for Placement...")
    graph_manager = CanonicalSiliconGraph()
    graph_manager.graph.add_node("mult_unit", node_type=NodeType.CELL, area=500.0, timing_criticality=0.95)
    
    env = SiliconRLEnvironment(graph_manager)
    trainer = SiliconRLTrainer(action_targets=["mult_unit"])
    
    # Run a quick training burst to "evolve" the agent for this specific design
    trainer.train_episode(env, max_steps=5)
    
    dashboard.log_optimization("RL_EVOLVE", "PlacementAgent", "POLICY_REFINED")
    
    # 4. PDK-AWARE TCL GENERATION
    print("\n[STEP 4] Hardened Commercial TCL Generation (Innovus)...")
    config = {
        'design_name': 'final_boss',
        'verilog': 'final_boss_pipelined.v',
        'sdc': 'constraints/signoff.sdc',
        'pdk_root': '/usr/local/pdk',
        'pdk_variant': 'hd',
        'node': 130
    }
    innovus_gen = TCLGeneratorFactory.get_generator('innovus')
    tcl = innovus_gen.generate_full_flow(config)
    
    with open("signoff_tapeout.tcl", "w") as f:
        f.write(tcl)
    
    dashboard.log_optimization("TCL_SIGNOFF", "SkyWater_130nm", "READY")
    
    # 5. FINAL AUTHORITY TELEMETRY
    print("\n" + "="*60)
    dashboard.display_full_report()
    print("="*60)
    
    print("\nGRAND DEMO COMPLETE.")
    print("Status: DESIGN READY FOR PHYSICAL VERIFICATION.")

if __name__ == "__main__":
    run_grand_tapeout_demo()
