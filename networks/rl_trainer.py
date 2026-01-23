"""
Reinforcement Learning Trainer - Manages Policy Training for Silicon Agents.
Uses a simple Q-Learning approach for demonstration, extensible to Deep Q-Networks (DQN).
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional

class SiliconRLTrainer:
    """
    Manages the learning loop and policy updates for Silicon Agents.
    """
    
    def __init__(self, action_targets: List[str]):
        self.action_targets = action_targets
        self.q_table = {} # state_key -> {action_idx: value}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2 # Exploration rate
        
    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """Discretize state for Q-Table mapping"""
        # Professional discretization: round metrics to nearest 0.1
        return f"t{round(state['avg_timing_slack'], 1)}_c{round(state['congestion_index'], 1)}"

    def choose_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Epsilon-greedy action selection"""
        state_key = self._get_state_key(state)
        
        # Exploration
        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.action_targets) - 1)
        else:
            # Exploitation
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(len(self.action_targets))
            action_idx = np.argmax(self.q_table[state_key])
            
        return {
            'type': 'pipeline', # Simple demo action type
            'target': self.action_targets[action_idx],
            'action_idx': action_idx
        }

    def update_policy(self, state: Dict[str, Any], action_idx: int, reward: float, next_state: Dict[str, Any]):
        """Q-Learning update rule"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_targets))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.action_targets))
            
        best_next_q = np.max(self.q_table[next_state_key])
        current_q = self.q_table[state_key][action_idx]
        
        # New Q-Value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

    def train_episode(self, env: Any, max_steps: int = 10) -> float:
        """Run a single training episode"""
        state = env.reset()
        total_reward = 0
        
        for _ in range(max_steps):
            action_data = self.choose_action(state)
            next_state, reward, done, _ = env.step(action_data)
            
            self.update_policy(state, action_data['action_idx'], reward, next_state)
            
            state = next_state
            total_reward += reward
            if done:
                break
                
        return total_reward

if __name__ == "__main__":
    print("RL Trainer Service Initialized.")
