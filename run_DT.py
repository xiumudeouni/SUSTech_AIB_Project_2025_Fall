"""
Run Decision Transformer Evaluation Script
------------------------------------------
This script loads a trained Decision Transformer model and evaluates it 
in the CartPole environment with a specified Target Return-to-Go (RTG).

Usage:
    python run_DT.py --model_path models/cartpole_dt.torch --rtg 500 --episodes 100
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import torch
import sys

# Ensure we can import from agents/
sys.path.append(os.getcwd())

from agents.cartpole_dt import DTSolver, DTConfig

ENV_NAME = "CartPole-v1"

def evaluate_dt(agent, target_return=500.0, episodes=100, max_steps=1000):
    """
    Evaluate Decision Transformer model.
    """
    env = gym.make(ENV_NAME)
    episode_returns = []
    
    print(f"Starting evaluation over {episodes} episodes...")
    print(f"Target Return (RTG): {target_return}")
    
    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(range(1, episodes + 1), desc="Evaluating")
    except ImportError:
        print("tqdm not found, showing simple progress...")
        iterator = range(1, episodes + 1)

    for ep in iterator:
        # Use random seed for robust evaluation
        state, _ = env.reset() 
        
        done = False
        steps = 0
        episode_return = 0
        
        returns_to_go = [target_return]
        states = [state]
        actions = []
        timesteps = [0]
        
        while not done and steps < max_steps:
            context_len = agent.cfg.context_len
            start_idx = max(0, len(states) - context_len)
            
            # Prepare inputs
            # Reshape to [1, seq_len, dim]
            rtg = np.array(returns_to_go[start_idx:]).reshape(1, -1, 1)
            s = np.array(states[start_idx:]).reshape(1, -1, agent.obs_dim)
            # Pad actions with 0 for the first step if empty, otherwise use history
            a = np.array(actions[start_idx:] if actions else [0]).reshape(1, -1)
            t = np.array(timesteps[start_idx:]).reshape(1, -1)
            
            # Get action from model
            action = agent.act(rtg, s, a, t)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            episode_return += reward
            
            # Update history
            states.append(next_state)
            actions.append(action)
            timesteps.append(steps)
            returns_to_go.append(returns_to_go[-1] - reward)
        
        episode_returns.append(episode_return)
        
        # Simple progress print if tqdm is missing
        if isinstance(iterator, range) and ep % 10 == 0:
            print(f"Episode {ep}/{episodes}: {episode_return:.1f}")
    
    env.close()
    return episode_returns

def main():
    parser = argparse.ArgumentParser(description="Run Decision Transformer Evaluation")
    parser.add_argument("--model_path", type=str, default="models/cartpole_dt_random.torch", help="Path to the trained model")
    parser.add_argument("--rtg", type=float, default=200, help="Target Return-to-Go (e.g., 500, 200)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please check the path or train a model first using train_offline_advanced.py")
        return

    # Initialize Environment to get dimensions
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    # Initialize Agent
    config = DTConfig()
    # Ensure device is set correctly
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agent = DTSolver(obs_dim, act_dim, config)
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    print(f"Device: {config.device}")
    try:
        agent.load(args.model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run Evaluation
    scores = evaluate_dt(agent, target_return=args.rtg, episodes=args.episodes)
    
    # Calculate Statistics
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    print("\n" + "="*40)
    print(f"Evaluation Results (RTG={args.rtg})")
    print("="*40)
    print(f"Episodes:      {args.episodes}")
    print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
    print(f"Max Score:     {max_score:.2f}")
    print(f"Min Score:     {min_score:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
