"""
RTG (Return-To-Go) Predictor for Decision Transformer
-----------------------------------------------------
This module implements a Value-based neural network that predicts the expected
return-to-go (RTG) for any given state. It can be trained on offline data and
used to provide dynamic RTG targets for Decision Transformer during inference.

Key idea:
- Train a value function V(s) to estimate expected future returns from state s
- During DT inference, use V(s) as the RTG target instead of fixed values
- This allows DT to work better on poor-quality datasets where optimal RTG is unknown

Architecture:
- Simple MLP: state -> hidden layers -> RTG prediction
- Trained with MSE loss on actual trajectory returns

Usage:
    1. Train RTG predictor on offline trajectories
    2. Load trained DT model
    3. During inference, use RTG predictor to get dynamic RTG for current state
    4. Feed predicted RTG to DT to get action
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from typing import Tuple


class RTGPredictor(nn.Module):
    """
    Neural network that predicts return-to-go (RTG) from state.
    Essentially a value function V(s) that estimates expected future returns.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict RTG for given state(s).
        
        Args:
            state: [B, state_dim] or [state_dim]
        
        Returns:
            rtg: [B, 1] or [1] predicted return-to-go
        """
        return self.net(state)


class RTGPredictorSolver:
    """
    Trainer and inference wrapper for RTG Predictor.
    Handles training on offline data and prediction during inference.
    Uses TD Learning (Fitted Value Iteration) to estimate V(s).
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            lr: Learning rate
            gamma: Discount factor (usually 1.0 for RTG in DT context)
            tau: Soft update rate for target network
            device: Device for training/inference
        """
        self.device = device
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = 100.0
        
        # Initialize model (Value Network)
        self.model = RTGPredictor(state_dim, hidden_dim, n_layers).to(device)
        
        # Target Network
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.loss_fn = nn.HuberLoss()
    
    def train_step(
        self, 
        states: np.ndarray, 
        rewards: np.ndarray, 
        next_states: np.ndarray, 
        dones: np.ndarray
    ) -> float:
        """
        Single training step using TD Learning.
        Target: r + gamma * V_target(s') * (1-done)
        
        Args:
            states: [B, state_dim]
            rewards: [B, 1] or [B]
            next_states: [B, state_dim]
            dones: [B, 1] or [B]
        
        Returns:
            loss: Training loss value
        """
        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        # Ensure shapes are [B, 1]
        if rewards.dim() == 1: rewards = rewards.unsqueeze(1)
        if dones.dim() == 1: dones = dones.unsqueeze(1)

        scaled_rewards = rewards / self.reward_scale
        
        # Compute Target
        with torch.no_grad():
            next_values = self.target_model(next_states)
            # TD Target 使用缩放后的奖励
            target_values = scaled_rewards + self.gamma * next_values * (1 - dones)
        
        pred_values = self.model(states)
        loss = self.loss_fn(pred_values, target_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # <--- 建议：增加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self._soft_update_target_network()
        return loss.item()
    
    def _soft_update_target_network(self):
        """Soft update target network parameters: theta_target = tau*theta + (1-tau)*theta_target"""
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def predict(self, state: np.ndarray) -> float:
        """
        Predict RTG for a single state.
        
        Args:
            state: [state_dim] or [1, state_dim] - single state
        
        Returns:
            rtg: Predicted return-to-go value (scalar)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensor
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(self.device)
            
            # Ensure correct shape [1, state_dim]
            if state.dim() == 1:
                state = state.unsqueeze(0)
            elif state.dim() == 3:  # [1, 1, state_dim]
                state = state.squeeze(1)
            
            # Predict
            pred = self.model(state)
        
        self.model.train()
        return pred.item() * self.reward_scale
    
    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Predict RTG for a batch of states.
        
        Args:
            states: [B, state_dim] - batch of states
        
        Returns:
            rtgs: [B] predicted return-to-go values
        """
        self.model.eval()
        
        with torch.no_grad():
            states = torch.from_numpy(states).float().to(self.device)
            preds = self.model(states)
        
        self.model.train()
        return preds.cpu().numpy().flatten()
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
        }, path)
        print(f"[RTG Predictor] Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f"[RTG Predictor] Model loaded from {path}")
