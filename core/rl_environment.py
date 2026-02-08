"""
Reinforcement Learning Environment for Silicon Design Optimization.
Provides the state, action, and reward interface for Silicon Agents.
"""

from typing import Dict, List, Any, Tuple
import copy
import os
import sys
import torch

# Add project root to path for correct module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.rtl_transformer import RTLTransformer
from rtl_bridge import RTLParsingBridge
from networks.graph_neural_network import SiliconGNN, convert_to_pyg_data
from core.canonical_silicon_graph import NodeType, EdgeType
import pyverilog.vparser.ast as vast

class SiliconRLEnvironment:
    """
    Gym-like environment for Reinforcement Learning in RTL Design.
    """

    def __init__(self, rtl_file: str, module_name: str, gnn_model_path: str = None):
        if not os.path.exists(rtl_file):
            raise FileNotFoundError(f"RTL file not found: {rtl_file}")
        self.rtl_file = rtl_file
        self.module_name = module_name
        self.transformer = RTLTransformer()
        self.bridge = RTLParsingBridge()
        self.initial_ast = self.transformer.parse_rtl(self.rtl_file)
        self.current_ast = copy.deepcopy(self.initial_ast)

        # Create the initial graph
        verilog_code = self.transformer.generate_verilog(self.initial_ast)
        self.initial_graph = self.bridge.build_graph_from_rtl(verilog_code)
        self.current_graph = copy.deepcopy(self.initial_graph)

        self.history = []
        self.action_space = []

        # Initialize the GNN model
        self.gnn_model = SiliconGNN(in_channels=7, hidden_channels=64, out_channels=3)
        if gnn_model_path and os.path.exists(gnn_model_path):
            self.gnn_model.load_state_dict(torch.load(gnn_model_path))
        self.gnn_model.eval()

        self._update_action_space()

    def _update_action_space(self):
        """
        Analyze the AST to find all possible transformation targets.
        For now, let's find all wire/reg signals that can be pipelined.
        """
        self.action_space = []
        target_module = self._get_target_module(self.current_ast)
        if not target_module:
            return

        # Find all assign statements - their outputs are candidates for pipelining
        for item in target_module.items:
            if isinstance(item, vast.Assign):
                if isinstance(item.left, vast.Lvalue):
                    signal_name = self._get_signal_name(item.left.var)
                    if signal_name:
                        # Avoid adding duplicates if a signal is used in multiple assigns (less common for outputs)
                        action = {'type': 'add_pipeline_stage', 'target_signal': signal_name}
                        if action not in self.action_space:
                            self.action_space.append(action)

    def _get_target_module(self, ast):
        for item in ast.description.definitions:
            if isinstance(item, vast.ModuleDef) and item.name == self.module_name:
                return item
        return None

    def _get_signal_name(self, var_node):
        """Helper to get signal name from AST node."""
        if isinstance(var_node, vast.Identifier):
            return var_node.name
        return None

    def reset(self) -> Dict[str, Any]:
        """Reset the environment to the initial state"""
        self.current_ast = copy.deepcopy(self.initial_ast)
        self.current_graph = copy.deepcopy(self.initial_graph)
        self.history = []
        self._update_action_space()
        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        """
        Get the current state representation from the current graph
        by getting a PPA prediction from the GNN.
        """
        # 1. Convert the graph to a PyTorch Geometric Data object
        pyg_data = convert_to_pyg_data(self.current_graph)

        # 2. Get the PPA prediction from the GNN
        with torch.no_grad():
            ppa_prediction = self.gnn_model(pyg_data.x, pyg_data.edge_index, None)

        # Use the PPA prediction as the state
        state = {
            'power': ppa_prediction[0][0].item(),
            'performance': ppa_prediction[0][1].item(),
            'area': ppa_prediction[0][2].item()
        }
        return state

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action and return (next_state, reward, done, info).
        """
        if not self.action_space or action_idx >= len(self.action_space):
            return self._get_state(), -10.0, True, {'error': 'Invalid action index'}

        action = self.action_space[action_idx]
        prev_state = self._get_state()
        
        action_type = action.get('type')
        if action_type == 'add_pipeline_stage':
            target_signal = action.get('target_signal')
            try:
                self.current_ast, new_reg_name = self.transformer.add_pipeline_stage(
                    self.current_ast, self.module_name, target_signal
                )
                self.transformer.update_signal_sinks(
                    self.current_ast, self.module_name, target_signal, new_reg_name
                )

                # Lightweight graph update
                self.current_graph.graph.add_node(new_reg_name, node_type=NodeType.CELL, cell_type='DFF', area=1.2, power=0.08)
                if self.current_graph.graph.has_node(target_signal):
                    self.current_graph.graph.add_edge(target_signal, new_reg_name, edge_type=EdgeType.CONNECTION)

            except Exception as e:
                return self._get_state(), -5.0, True, {'error': str(e)}

        self._update_action_space()
        new_state = self._get_state()
        reward = self._calculate_reward(prev_state, new_state, action)
        
        self.history.append(action)
        done = len(self.history) >= 10
        
        return new_state, reward, done, {}

    def _calculate_reward(self, prev_state: Dict, new_state: Dict, action: Dict) -> float:
        """
        Sparse but honest reward function based on GNN PPA predictions.
        """
        # Define weights for the PPA components
        weights = {'power': -0.3, 'performance': 0.5, 'area': -0.2}

        # Calculate the PPA score for the previous and new states
        prev_ppa_score = (weights['power'] * prev_state['power'] +
                          weights['performance'] * prev_state['performance'] +
                          weights['area'] * prev_state['area'])
        
        new_ppa_score = (weights['power'] * new_state['power'] +
                         weights['performance'] * new_state['performance'] +
                         weights['area'] * new_state['area'])

        # The reward is the difference in the PPA scores
        reward = new_ppa_score - prev_ppa_score
        
        # Add a small penalty for each action to encourage efficiency
        reward -= 0.1

        return float(reward)

    def get_action_space_size(self):
        return len(self.action_space)

def test_rl_env():
    test_rtl_file = "test_design_env.v"
    with open(test_rtl_file, "w") as f:
        f.write('''
module test_engine (
    input clk,
    input [7:0] data_in,
    output [7:0] data_out
);
    wire [7:0] processed_data;
    assign processed_data = data_in ^ 8'hFF;
    assign data_out = processed_data;
endmodule
''')
    
    env = SiliconRLEnvironment(test_rtl_file, "test_engine")
    state = env.reset()
    print(f"Initial State: {state}")
    print(f"Action space size: {env.get_action_space_size()}")
    if env.get_action_space_size() > 0:
        action_idx = 0
        print(f"Performing action: {env.action_space[action_idx]}")
        next_state, reward, done, info = env.step(action_idx)
        print(f"Next State: {next_state}")
        print(f"Reward: {reward}")
    os.remove(test_rtl_file)

if __name__ == "__main__":
    test_rl_env()
