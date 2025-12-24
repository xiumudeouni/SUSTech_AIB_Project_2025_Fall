"""
PyTorch Double DQN (DDQN) for CartPole
--------------------------------------
Inherits from the standard DQNSolver.
The ONLY difference is in the Target calculation (step 3 of experience_replay).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

from agents.cartpole_dqn import DQNSolver, DQNConfig

class DoubleDQNSolver(DQNSolver):
    """
    Double DQN Agent.
    It uses the Online Network to SELECT the best action for the next state,
    but uses the Target Network to EVALUATE the Q-value of that action.
    """

    def experience_replay(self):
        """
        Overridden method to implement Double DQN logic.
        """
        # 1) Warmup check
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
            self._decay_eps()
            return

        # 2) Sample
        s, a, r, s2, m = self.memory.sample(self.cfg.batch_size)

        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a_t  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r_t  = torch.as_tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        m_t  = torch.as_tensor(m,  dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3) Compute Targets (THIS IS THE KEY DIFFERENCE)
        with torch.no_grad():
            # Standard DQN was:
            # q_next = self.target(s2_t).max(dim=1, keepdim=True)[0]
            
            # --- Double DQN Logic Start ---
            # A. Use ONLINE network to select the best action argmax Q(s', a)
            best_actions = self.online(s2_t).argmax(dim=1, keepdim=True)
            
            # B. Use TARGET network to evaluate that specific action
            # gather(1, best_actions) picks the value at the index of best_actions
            q_next = self.target(s2_t).gather(1, best_actions)
            # --- Double DQN Logic End ---

            target = r_t + m_t * self.cfg.gamma * q_next

        # 4) Compute Loss
        q_sa = self.online(s_t).gather(1, a_t)
        loss = nn.functional.mse_loss(q_sa, target)

        # 5) Optimize
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # 6) Decay & Update
        self._decay_eps()
        if self.steps % self.cfg.target_update == 0:
            self.update_target(hard=True)