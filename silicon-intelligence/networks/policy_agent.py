
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Tuple

class PolicyNetwork(nn.Module):
    """Neural network with Actor and Critic heads for PPO/A2C algorithms"""
    def __init__(self, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Actor: Outputs probability distribution over actions
        self.actor = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: Outputs state value (V)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        features = self.feature_layer(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class PolicyAgent:
    """The RL Agent that learns through interaction inside DesignOptimizationEnv"""
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = [] # Design Memory / Experience Replay

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Decide next optimization step using the internal policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state_tensor)
        
        # Create a categorical distribution and sample an action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)

    def save_to_memory(self, state, log_prob, reward, next_state, done):
        """Record experience for later learning"""
        self.memory.append((state, log_prob, reward, next_state, done))

    def learn(self):
        """Update policy based on collected design episodes"""
        if len(self.memory) < 5: # Minimum batch size
            return

        states, log_probs, rewards, next_states, dones = zip(*self.memory)
        
        # Standard Reinforcement Learning update logic (A2C style)
        # 1. Compute discounted returns
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + 0.99 * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns)
        log_probs = torch.stack(log_probs)
        
        # 2. Update Neural Network
        # Loss = Lead-Policy Loss - lead-Value Loss
        _, values = self.model(torch.FloatTensor(np.array(states)))
        values = values.squeeze()
        
        advantage = returns - values.detach()
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.MSELoss()(values, returns)
        
        total_loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.memory = [] # Clear memory after update
        print(f"Policy Updated: Loss={total_loss.item():.4f}")

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Agent Policy saved to {path}")

    def load_model(self, path: str):
        if torch.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Agent Policy loaded from {path}")
