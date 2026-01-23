"""
Reinforcement Learning Environment for Silicon Design Optimization.
Provides the state, action, and reward interface for Silicon Agents.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import copy
from core.canonical_silicon_graph import CanonicalSiliconGraph

class SiliconRLEnvironment:
    """
    Gym-like environment for Reinforcement Learning in Silicon Design.
    """
    
    def __init__(self, initial_graph: CanonicalSiliconGraph):
        self.initial_graph = initial_graph
        self.current_graph = copy.deepcopy(initial_graph)
        self.history = []
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to the initial state"""
        self.current_graph = copy.deepcopy(self.initial_graph)
        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        """Get the current state representation (observations)"""
        # In a real system, this would be a GNN embedding or a feature vector
        graph = self.current_graph.graph
        stats = {
            'node_count': len(graph.nodes),
            'edge_count': len(graph.edges),
            'avg_timing_slack': np.mean([n[1].get('timing_criticality', 0) for n in graph.nodes(data=True)]),
            'total_area': sum([n[1].get('area', 1.0) for n in graph.nodes(data=True) if n[1].get('node_type', '').value == 'cell']),
            'congestion_index': np.mean([n[1].get('estimated_congestion', 0) for n in graph.nodes(data=True)])
        }
        return stats

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action and return (next_state, reward, done, info).
        Action format: {'type': 'pipeline', 'target': 'node_id'} or {'type': 'move', 'cell': 'id', 'delta': (x,y)}
        """
        prev_stats = self._get_state()
        
        # Apply action to current_graph
        action_type = action.get('type')
        if action_type == 'pipeline':
            # Mock pipelining effect: improve timing, increase area
            target = action.get('target')
            if target in self.current_graph.graph.nodes:
                self.current_graph.graph.nodes[target]['timing_criticality'] *= 0.8
                self.current_graph.graph.nodes[target]['area'] *= 1.2
        
        elif action_type == 'relocate':
            # Mock relocation: reduce congestion
            target = action.get('target')
            if target in self.current_graph.graph.nodes:
                self.current_graph.graph.nodes[target]['estimated_congestion'] *= 0.9
        
        new_state = self._get_state()
        reward = self._calculate_reward(prev_stats, new_state)
        
        # Simple termination: after 10 steps or if reward is very high
        self.history.append(action)
        done = len(self.history) >= 10
        
        return new_state, reward, done, {}

    def _calculate_reward(self, prev_stats: Dict, new_stats: Dict) -> float:
        """
        Reward function: (+) for improvements, (-) for penalties.
        """
        reward = 0.0
        
        # Reward timing improvement
        timing_delta = prev_stats['avg_timing_slack'] - new_stats['avg_timing_slack']
        reward += timing_delta * 100.0
        
        # Penalty for area increase
        area_penalty = new_stats['total_area'] - prev_stats['total_area']
        reward -= area_penalty * 2.0
        
        # Reward congestion reduction
        congestion_delta = prev_stats['congestion_index'] - new_stats['congestion_index']
        reward += congestion_delta * 50.0
        
        return float(reward)

def test_rl_env():
    from core.canonical_silicon_graph import NodeType
    graph_manager = CanonicalSiliconGraph()
    graph_manager.graph.add_node("c1", node_type=NodeType.CELL, area=10.0, timing_criticality=0.9)
    
    env = SiliconRLEnvironment(graph_manager)
    state = env.reset()
    print(f"Initial State: {state}")
    
    next_state, reward, done, info = env.step({'type': 'pipeline', 'target': 'c1'})
    print(f"Next State: {next_state}")
    print(f"Reward: {reward}")

if __name__ == "__main__":
    test_rl_env()
