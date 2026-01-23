"""
Professional RL Demonstration - Agent Self-Improvement Loop.
Shows a Silicon Agent learning to optimize timing slack over multiple episodes.
Branded by Street Heart Technologies.
"""

from core.rl_environment import SiliconRLEnvironment
from networks.rl_trainer import SiliconRLTrainer
from core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType
from utils.authority_dashboard import AuthorityDashboard
import time

def run_rl_learning_demo():
    print("Initializing RL Self-Improvement Loop...")
    dashboard = AuthorityDashboard()
    
    # 1. Setup Environment
    graph_manager = CanonicalSiliconGraph()
    targets = ["cell_1", "cell_2", "cell_3"]
    for t in targets:
        graph_manager.graph.add_node(t, node_type=NodeType.CELL, area=10.0, timing_criticality=0.9, estimated_congestion=0.5)
    
    env = SiliconRLEnvironment(graph_manager)
    trainer = SiliconRLTrainer(action_targets=targets)
    
    # 2. Training Loop
    episodes = 20
    print(f"Training for {episodes} episodes...")
    
    rewards_history = []
    
    for ep in range(episodes):
        total_reward = trainer.train_episode(env)
        rewards_history.append(total_reward)
        
        if (ep + 1) % 5 == 0:
            print(f" Episode {ep+1}/{episodes}, Total Reward: {total_reward:.2f}")
            dashboard.log_optimization("RL_TRAIN", f"EPISODE_{ep+1}", f"RWD:{total_reward:.1f}")

    # 3. Final Report
    print("\nLearning Complete. Agent has evolved design policies.")
    
    # Show Dashboard with RL insights
    dashboard.log_optimization("SELF_EVOLVE", "PlacementAgent", "POLICY_UPDATED")
    dashboard.display_full_report()
    
    if rewards_history[-1] > rewards_history[0]:
        print("\n✅ RL Self-Improvement Verified: Agent performance improved over time!")
    else:
        print("\n⚠️ RL convergence not fully observed in short demo.")

if __name__ == "__main__":
    run_rl_learning_demo()
