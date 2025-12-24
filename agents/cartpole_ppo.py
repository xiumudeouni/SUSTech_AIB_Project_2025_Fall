"""
PPO for CartPole (Gymnasium)
------------------------------------
- Proximal Policy Optimization (PPO) algorithm
- Actor-Critic architecture
- Generalized Advantage Estimation (GAE)
- Designed to be imported by train.py
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------------
# Default Hyperparameters
# -----------------------------
LR = 1e-4
GAMMA = 0.99
K_EPOCHS = 4
EPS_CLIP = 0.2
UPDATE_TIMESTEP = 100
ACTION_STD = 0.6  # Not used for discrete CartPole

@dataclass
class PPOConfig:
    lr: float = LR
    gamma: float = GAMMA
    eps_clip: float = EPS_CLIP
    k_epochs: int = K_EPOCHS
    update_timestep: int = UPDATE_TIMESTEP
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPOSolver:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig | None = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)
        
        self.policy = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        
        self.policy_old = ActorCritic(obs_dim, act_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = []
        self.time_step = 0
        
        # Temporary storage for the current step
        self.last_log_prob = None
        self.last_val = None
        
        # For logging
        self.exploration_rate = 0.0 # PPO handles exploration via entropy, but train.py might print it

    def act(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        Select action using the current policy.
        """
        state_t = torch.FloatTensor(state).to(self.device)
        # state is [1, obs_dim]
        
        if evaluation_mode:
            with torch.no_grad():
                action_probs = self.policy.actor(state_t)
                action = torch.argmax(action_probs, dim=1)
            return action.item()
        else:
            action, log_prob, val = self.policy_old.act(state_t)
            self.last_log_prob = log_prob
            self.last_val = val
            return action.item()

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float | None:
        self.buffer.append({
            'state': state, 'action': action, 'log_prob': self.last_log_prob,
            'reward': reward, 'done': done, 'val': self.last_val
        })
        self.time_step += 1
        
        if self.time_step % self.cfg.update_timestep == 0:
            return self.update() # 更新时返回 Loss
        return None

    def update(self) -> float:
        # Convert buffer to lists
        rewards = [t['reward'] for t in self.buffer]
        is_terminals = [t['done'] for t in self.buffer]
        old_states = torch.FloatTensor(np.vstack([t['state'] for t in self.buffer])).to(self.device)
        old_actions = torch.LongTensor([t['action'] for t in self.buffer]).to(self.device)
        old_logprobs = torch.stack([t['log_prob'] for t in self.buffer]).to(self.device).squeeze()
        
        # Monte Carlo estimate of returns
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.cfg.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(self.device)
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-7)
        
        total_loss = 0
        for _ in range(self.cfg.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards_to_go - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages

            # 计算复合 Loss
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards_to_go) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            total_loss += loss.mean().item()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return total_loss / self.cfg.k_epochs #

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())
