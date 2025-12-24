"""
Comparison Script: Standard DQN vs. Double DQN
----------------------------------------------
1. Trains Standard DQN.
2. Trains Double DQN with the EXACT same hyperparameters.
3. Plots the learning curves to visualize stability and performance.
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

# Import both agents
from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_double_dqn import DoubleDQNSolver

# Configuration
ENV_NAME = "CartPole-v1"
VIS_DIR = "visualizations_ddqn"
MODEL_DIR = "models"
NUM_EPISODES = 1000  # Enough to see convergence for CartPole

# Shared Hyperparameters for fair comparison
COMMON_CONFIG = DQNConfig(
    lr=1e-3,
    batch_size=64,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.9995,
    target_update=200
)

def run_training(agent_class, run_name: str) -> list[int]:
    """
    Generic training loop that accepts an agent class (DQN or DoubleDQN).
    """
    print(f"\n--- Starting Training: {run_name} ---")
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Instantiate the specific agent class
    # We pass a copy of config to ensure no cross-contamination
    agent = agent_class(obs_dim, act_dim, cfg=COMMON_CONFIG)
    
    scores = []
    
    for episode in range(1, NUM_EPISODES + 1):
        # Seed logic ensures both agents face similar initial conditions roughly
        state, _ = env.reset(seed=episode)
        state = np.reshape(state, (1, obs_dim))
        steps = 0
        done = False

        while not done:
            steps += 1
            action = agent.act(state)
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Standard penalty (optional, keeping consistent)
            if done and steps < 500: reward = -1.0
            
            next_state = np.reshape(next_state_raw, (1, obs_dim))
            agent.step(state, action, reward, next_state, done)
            state = next_state

        scores.append(steps)
        if episode % 10 == 0:
            print(f"[{run_name}] Ep {episode}: {steps} steps (Eps: {agent.exploration_rate:.3f})")

    # Save model
    save_path = os.path.join(MODEL_DIR, f"{run_name.lower().replace(' ', '_')}.torch")
    agent.save(save_path)
    env.close()
    return scores

def plot_results(results: dict):
    """
    Draws the comparison plot.
    """
    plt.figure(figsize=(10, 6))
    
    for name, scores in results.items():
        # Compute moving average for cleaner plots
        window = 10
        ma = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(ma, label=name)

    plt.title("DQN vs Double DQN (CartPole-v1)")
    plt.xlabel(f"Episodes (Moving Avg Window: 10)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(VIS_DIR, "dqn_vs_double_dqn.png")
    plt.savefig(save_path)
    print(f"\n[Info] Visualization saved to {save_path}")
    plt.close()

def main():
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Dictionary to store results
    all_results = {}

    # 1. Train Standard DQN
    all_results["Standard DQN"] = run_training(DQNSolver, "Standard DQN")

    # 2. Train Double DQN
    all_results["Double DQN"] = run_training(DoubleDQNSolver, "Double DQN")

    # 3. Plot
    plot_results(all_results)

if __name__ == "__main__":
    main()