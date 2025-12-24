"""
Decision Transformer for CartPole (Offline RL)
----------------------------------------------
- Treats RL as a sequence modeling problem using Transformer architecture
- Input: sequences of (return-to-go, state, action) tuples
- Output: predicted actions
- Trained on offline trajectories via supervised learning

Reading guide:
1. TransformerBlock: standard multi-head self-attention + FFN
2. DecisionTransformer: embeds (R, s, a) and predicts actions autoregressively
3. DTSolver: agent that loads offline data and trains the transformer
4. During inference: condition on desired return to generate high-reward behavior

Reference: Chen et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling" (NeurIPS 2021)
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# -----------------------------
# Hyperparameters
# -----------------------------
EMBED_DIM = 128          # Embedding dimension
N_HEADS = 4              # Number of attention heads
N_LAYERS = 3             # Number of transformer blocks
DROPOUT = 0.1            # Dropout rate
MAX_EP_LEN = 1000        # Maximum episode length
CONTEXT_LEN = 20         # Context length (K in paper)
BATCH_SIZE = 64          # Training batch size
LR = 1e-4                # Learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization


@dataclass
class DTConfig:
    """Configuration for Decision Transformer."""
    embed_dim: int = EMBED_DIM
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    dropout: float = DROPOUT
    max_ep_len: int = MAX_EP_LEN
    context_len: int = CONTEXT_LEN
    batch_size: int = BATCH_SIZE
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with masking.
    Ensures tokens can only attend to previous tokens (autoregressive).
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % n_heads == 0
        
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        
        # Q, K, V projections for all heads (batched)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask (upper triangular = -inf)
        # Will be registered as buffer during forward
        self.register_buffer("mask", None, persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # [batch, seq_len, embed_dim]
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        # Reshape to [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B, H, T, T]
        
        # Apply causal mask
        if self.mask is None or self.mask.size(0) < T:
            # Create causal mask: lower triangular matrix
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            self.mask = mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted sum of values
        y = att @ v  # [B, H, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        
        # Output projection
        y = self.resid_dropout(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    """
    Standard Transformer block: LayerNorm -> Attention -> LayerNorm -> FFN.
    Uses pre-normalization (GPT-2 style).
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm style
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer model.
    
    Architecture:
    1. Embed returns, states, actions separately
    2. Add timestep embeddings
    3. Interleave into sequence: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
    4. Pass through Transformer blocks
    5. Predict actions from state tokens
    """
    def __init__(self, obs_dim: int, act_dim: int, cfg: DTConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        
        # Token embeddings
        self.embed_return = nn.Linear(1, cfg.embed_dim)
        self.embed_state = nn.Linear(obs_dim, cfg.embed_dim)
        self.embed_action = nn.Linear(act_dim, cfg.embed_dim)
        
        # Timestep (position) embedding
        self.embed_timestep = nn.Embedding(cfg.max_ep_len, cfg.embed_dim)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(cfg.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        
        # Output heads
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.predict_action = nn.Linear(cfg.embed_dim, act_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(
        self,
        returns_to_go: torch.Tensor,  # [B, T, 1]
        states: torch.Tensor,          # [B, T, obs_dim]
        actions: torch.Tensor,         # [B, T, act_dim]
        timesteps: torch.Tensor,       # [B, T]
    ) -> torch.Tensor:
        """
        Forward pass through Decision Transformer.
        
        Args:
            returns_to_go: Return-to-go at each timestep
            states: State observations
            actions: Actions taken (one-hot encoded for discrete)
            timesteps: Timestep indices (for positional encoding)
        
        Returns:
            action_preds: Predicted action logits [B, T, act_dim]
        """
        B, T = states.shape[0], states.shape[1]
        
        # Embed each modality
        return_embeddings = self.embed_return(returns_to_go)     # [B, T, D]
        state_embeddings = self.embed_state(states)              # [B, T, D]
        action_embeddings = self.embed_action(actions)           # [B, T, D]
        
        # Add timestep embeddings
        time_embeddings = self.embed_timestep(timesteps)         # [B, T, D]
        return_embeddings = return_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        
        # Stack and interleave: (R, s, a, R, s, a, ...)
        # Shape: [B, 3*T, D]
        stacked = torch.stack(
            [return_embeddings, state_embeddings, action_embeddings], dim=2
        ).reshape(B, 3 * T, self.embed_dim)
        
        stacked = self.embed_dropout(stacked)
        
        # Pass through transformer blocks
        x = stacked
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # Extract state tokens (indices 1, 4, 7, ... = 3*i+1)
        # Predict actions from state representations
        state_indices = torch.arange(1, 3 * T, 3, device=x.device)
        state_reprs = x[:, state_indices, :]  # [B, T, D]
        
        action_preds = self.predict_action(state_reprs)  # [B, T, act_dim]
        
        return action_preds
    
    def get_action(
        self,
        returns_to_go: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> int:
        """
        Get action for the last timestep (inference mode).
        Used during evaluation.
        """
        # Forward pass
        action_preds = self.forward(returns_to_go, states, actions, timesteps)
        
        # Take the last timestep's prediction
        last_action_logits = action_preds[:, -1, :]  # [B, act_dim]
        
        # For CartPole (discrete), take argmax
        action = torch.argmax(last_action_logits, dim=-1).item()
        return action


class DTSolver:
    """
    Decision Transformer agent for offline RL on CartPole.
    
    Workflow:
    1. Load offline trajectory data
    2. Train transformer to predict actions given (R, s, a) context
    3. At test time, condition on desired return and generate actions
    """
    def __init__(self, obs_dim: int, act_dim: int, cfg: DTConfig | None = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg or DTConfig()
        
        self.device = torch.device(self.cfg.device)
        
        # Build model
        self.model = DecisionTransformer(obs_dim, act_dim, self.cfg).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        
        # Track training stats
        self.train_losses = []
    
    def train_step(
        self,
        returns_to_go: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        timesteps: np.ndarray,
    ) -> float:
        """
        Single training step on a batch of trajectory segments.
        
        Args:
            returns_to_go: [B, T, 1]
            states: [B, T, obs_dim]
            actions: [B, T] (action indices)
            timesteps: [B, T]
        
        Returns:
            loss: Training loss (cross-entropy for discrete actions)
        """
        # Convert to tensors
        rtg = torch.from_numpy(returns_to_go).float().to(self.device)
        s = torch.from_numpy(states).float().to(self.device)
        a = torch.from_numpy(actions).long().to(self.device)
        t = torch.from_numpy(timesteps).long().to(self.device)
        
        # One-hot encode actions for embedding
        a_onehot = F.one_hot(a, num_classes=self.act_dim).float()  # [B, T, act_dim]
        
        # Forward pass
        action_preds = self.model(rtg, s, a_onehot, t)  # [B, T, act_dim]
        
        # Compute loss (cross-entropy)
        # Flatten for loss computation
        loss = F.cross_entropy(
            action_preds.reshape(-1, self.act_dim),
            a.reshape(-1),
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def act(
        self,
        returns_to_go: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        timesteps: np.ndarray,
    ) -> int:
        """
        Select action given context (for evaluation).
        
        Args:
            returns_to_go: [1, T, 1] - desired return trajectory
            states: [1, T, obs_dim] - state history
            actions: [1, T-1] - action history (one less than states)
            timesteps: [1, T] - timestep indices
        
        Returns:
            action: Predicted action (0 or 1)
        """
        self.model.eval()
        
        with torch.no_grad():
            rtg = torch.from_numpy(returns_to_go).float().to(self.device)
            s = torch.from_numpy(states).float().to(self.device)
            
            # Pad actions with dummy (will only use up to T-1)
            T = s.shape[1]
            a_padded = np.zeros((1, T), dtype=np.int64)
            a_padded[0, :len(actions[0])] = actions[0]
            a = torch.from_numpy(a_padded).long().to(self.device)
            a_onehot = F.one_hot(a, num_classes=self.act_dim).float()
            
            t = torch.from_numpy(timesteps).long().to(self.device)
            
            # Get action prediction
            action = self.model.get_action(rtg, s, a_onehot, t)
        
        self.model.train()
        return action
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'cfg': self.cfg.__dict__,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])


class OfflineDataset:
    """
    Dataset for offline trajectory data.
    Handles segmenting trajectories into context windows and computing return-to-go.
    """
    def __init__(self, trajectories: list, context_len: int = 20):
        """
        Args:
            trajectories: List of dicts with keys:
                - 'states': [T, obs_dim]
                - 'actions': [T]
                - 'rewards': [T]
                - 'dones': [T]
            context_len: Maximum context length (K in paper)
        """
        self.trajectories = trajectories
        self.context_len = context_len
        
        # Preprocess: compute returns-to-go
        self._compute_returns_to_go()
    
    def _compute_returns_to_go(self):
        """Compute discounted return-to-go for each trajectory."""
        for traj in self.trajectories:
            rewards = traj['rewards']
            T = len(rewards)
            rtg = np.zeros(T)
            
            # Compute return-to-go (no discounting for simplicity)
            # rtg[t] = sum of rewards from t to end
            rtg[-1] = rewards[-1]
            for t in range(T - 2, -1, -1):
                rtg[t] = rewards[t] + rtg[t + 1]
            
            traj['returns_to_go'] = rtg
    
    def sample_batch(self, batch_size: int) -> Tuple:
        """
        Sample a batch of trajectory segments.
        
        Returns:
            returns_to_go: [B, T, 1]
            states: [B, T, obs_dim]
            actions: [B, T]
            timesteps: [B, T]
        """
        batch_rtg, batch_states, batch_actions, batch_timesteps = [], [], [], []
        
        for _ in range(batch_size):
            # Sample a random trajectory
            traj = self.trajectories[np.random.randint(len(self.trajectories))]
            
            T = len(traj['states'])
            
            # Sample a random starting point
            if T <= self.context_len:
                start_idx = 0
                end_idx = T
            else:
                start_idx = np.random.randint(0, T - self.context_len + 1)
                end_idx = start_idx + self.context_len
            
            # Extract segment
            rtg = traj['returns_to_go'][start_idx:end_idx]
            states = traj['states'][start_idx:end_idx]
            actions = traj['actions'][start_idx:end_idx]
            timesteps = np.arange(start_idx, end_idx)
            
            # Pad if necessary
            segment_len = end_idx - start_idx
            if segment_len < self.context_len:
                pad_len = self.context_len - segment_len
                rtg = np.concatenate([np.zeros(pad_len), rtg])
                states = np.concatenate([np.zeros((pad_len, states.shape[1])), states])
                actions = np.concatenate([np.zeros(pad_len, dtype=np.int64), actions])
                timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps])
            
            batch_rtg.append(rtg)
            batch_states.append(states)
            batch_actions.append(actions)
            batch_timesteps.append(timesteps)
        
        return (
            np.array(batch_rtg)[:, :, None],  # [B, T, 1]
            np.array(batch_states),           # [B, T, obs_dim]
            np.array(batch_actions),          # [B, T]
            np.array(batch_timesteps),        # [B, T]
        )
    
    def __len__(self):
        return len(self.trajectories)
