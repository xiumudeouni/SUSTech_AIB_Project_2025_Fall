"""
Training Script for RTG Predictor + Decision Transformer
--------------------------------------------------------
This script trains an RTG predictor on offline data and evaluates how it
improves DT performance on poor-quality datasets (especially random data).

Workflow:
1. Load offline dataset (e.g., random_data.pkl)
2. Train RTG predictor to estimate value function V(s)
3. Load pre-trained DT model
4. Evaluate DT with two modes:
   a) Fixed RTG: Use fixed target RTG (e.g., 500)
   b) Predicted RTG: Use RTG predictor to dynamically predict RTG
5. Compare performance to show improvement on poor datasets

Usage:
    # Train RTG predictor
    python train_with_rtg_predictor.py --mode train_rtg --data offline_data/random_data.pkl --epochs 100
    
    # Evaluate DT with fixed RTG
    python train_with_rtg_predictor.py --mode eval_fixed --dt_model models/cartpole_dt_random.torch --target_rtg 500
    
    # Evaluate DT with predicted RTG
    python train_with_rtg_predictor.py --mode eval_predicted --dt_model models/cartpole_dt_random.torch --rtg_model models/rtg_predictor.torch
    
    # Compare both methods
    python train_with_rtg_predictor.py --mode compare --dt_model models/cartpole_dt_random.torch --rtg_model models/rtg_predictor.torch
    
    # Full pipeline: train RTG predictor then compare
    python train_with_rtg_predictor.py --mode full_pipeline --data offline_data/random_data.pkl --dt_model models/cartpole_dt_random.torch
"""

import os
import argparse
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from agents.cartpole_dt import DTSolver, DTConfig, OfflineDataset
from agents.rtg_predictor import RTGPredictorSolver

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
DATA_DIR = "offline_data"
VIZ_DIR = "visualizations"


class TransitionDataset:
    """
    Dataset for TD Learning: samples (s, r, s', done) transitions.
    """
    def __init__(self, trajectories: list):
        self.states = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        for traj in trajectories:
            states = traj['states']
            rewards = traj['rewards']
            dones = traj['dones']
            
            # Create transitions
            # s_t, r_t, s_{t+1}, done_t
            # Note: trajectories usually have T states and T rewards
            # We need s_{t+1}, so we iterate up to T-1
            T = len(states)
            for t in range(T - 1):
                self.states.append(states[t])
                self.rewards.append(rewards[t])
                self.next_states.append(states[t+1])
                self.dones.append(dones[t])
            
            # Handle terminal state if needed (often s_T is terminal)
            # For CartPole, done=True means episode ended.
            # If done[t] is True, next_state might be irrelevant but we store it anyway.
            
        self.states = np.array(self.states)
        self.rewards = np.array(self.rewards)
        self.next_states = np.array(self.next_states)
        self.dones = np.array(self.dones)
        
        print(f"TransitionDataset: {len(self.states)} transitions loaded.")
        
    def sample_batch(self, batch_size: int):
        indices = np.random.randint(0, len(self.states), size=batch_size)
        return (
            self.states[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


def train_rtg_predictor(data_path: str, epochs: int = 50, batch_size: int = 64):
    """
    Train RTG predictor on offline dataset using TD Learning.
    
    Args:
        data_path: Path to offline data pickle file
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        solver: Trained RTG predictor solver
        losses: Training loss history
    """
    print("="*70)
    print(f"Training RTG Predictor (TD Learning) on {data_path}")
    print("="*70)
    
    # Load offline data
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Compute statistics
    total_returns = [traj['rewards'].sum() for traj in trajectories]
    print(f"Dataset statistics:")
    print(f"  Mean return: {np.mean(total_returns):.2f}")
    print(f"  Std return: {np.std(total_returns):.2f}")
    print(f"  Min return: {np.min(total_returns):.2f}")
    print(f"  Max return: {np.max(total_returns):.2f}")
    
    # Create dataset
    dataset = TransitionDataset(trajectories)
    
    # Initialize RTG predictor
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    env.close()
    
    solver = RTGPredictorSolver(
        state_dim=state_dim,
        hidden_dim=128,
        n_layers=3,
        lr=1e-4,
        gamma=0.99  # Undiscounted return for RTG
    )
    
    # Training loop
    losses = []
    num_batches_per_epoch = 1000  # More updates for TD
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        
        for _ in range(num_batches_per_epoch):
            # Sample batch
            states, rewards, next_states, dones = dataset.sample_batch(batch_size)
            
            # Training step
            loss = solver.train_step(states, rewards, next_states, dones)
            epoch_loss += loss
        
        avg_loss = epoch_loss / num_batches_per_epoch
        losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4e}")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "rtg_predictor.torch")
    solver.save(save_path)
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('TD Loss', fontsize=12)
    plt.title('RTG Predictor Training Loss (TD Learning)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(VIZ_DIR, exist_ok=True)
    plot_path = os.path.join(VIZ_DIR, "rtg_predictor_training.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining complete! Loss curve saved to {plot_path}")
    
    return solver, losses


def evaluate_dt_fixed_rtg(dt_model_path: str, target_rtg: float = 500.0, episodes: int = 20):
    """
    Evaluate DT with fixed target RTG.
    
    Args:
        dt_model_path: Path to trained DT model
        target_rtg: Fixed target RTG value
        episodes: Number of evaluation episodes
    
    Returns:
        avg_return: Average episode return
        all_returns: List of all episode returns
    """
    print("="*70)
    print(f"Evaluating DT with Fixed RTG = {target_rtg}")
    print("="*70)
    
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Load DT model
    cfg = DTConfig()
    dt_agent = DTSolver(state_dim, act_dim, cfg)
    dt_agent.load(dt_model_path)
    
    all_returns = []
    
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        
        # Initialize history with single timestep
        states = np.reshape(state, (1, 1, state_dim))  # [1, 1, state_dim]
        actions = np.zeros((1, 0), dtype=np.int64)  # [1, 0] - no actions yet
        returns_to_go = np.reshape(target_rtg, (1, 1, 1))  # [1, 1, 1]
        timesteps = np.zeros((1, 1), dtype=np.int64)  # [1, 1]
        
        done = False
        ep_return = 0
        t = 0
        
        while not done:
            # Prepare input for DT (keep only context_len)
            if states.shape[1] > cfg.context_len:
                context_start = states.shape[1] - cfg.context_len
                states_in = states[:, context_start:, :]
                actions_in = actions[:, context_start:]
                rtg_in = returns_to_go[:, context_start:, :]
                timesteps_in = timesteps[:, context_start:]
            else:
                states_in = states
                actions_in = actions
                rtg_in = returns_to_go
                timesteps_in = timesteps
            
            action = dt_agent.act(rtg_in, states_in, actions_in, timesteps_in)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ep_return += reward
            
            # Update history (only if not done, to keep proper sequence)
            if not done:
                t += 1
                next_state = np.reshape(next_state, (1, 1, state_dim))
                states = np.concatenate([states, next_state], axis=1)
                actions = np.concatenate([actions, np.reshape(action, (1, 1))], axis=1)
                timesteps = np.concatenate([timesteps, np.reshape(t, (1, 1))], axis=1)
                
                # Update RTG (decrement by reward)
                new_rtg = returns_to_go[0, -1, 0] - reward
                returns_to_go = np.concatenate([
                    returns_to_go,
                    np.reshape(new_rtg, (1, 1, 1))
                ], axis=1)
            else:
                # Just record the action for the final step
                actions = np.concatenate([actions, np.reshape(action, (1, 1))], axis=1)
        
        all_returns.append(ep_return)
        if ep % 5 == 0:
            print(f"Episode {ep}/{episodes}: Return = {ep_return}")
    
    avg_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    
    print(f"\n{'='*70}")
    print(f"Fixed RTG Results:")
    print(f"  Average Return: {avg_return:.2f} ± {std_return:.2f}")
    print(f"  Min Return: {np.min(all_returns):.2f}")
    print(f"  Max Return: {np.max(all_returns):.2f}")
    print(f"{'='*70}\n")
    
    env.close()
    return avg_return, all_returns


def evaluate_dt_predicted_rtg(dt_model_path: str, rtg_model_path: str, episodes: int = 20):
    """
    Evaluate DT with dynamically predicted RTG from RTG predictor.
    
    Args:
        dt_model_path: Path to trained DT model
        rtg_model_path: Path to trained RTG predictor model
        episodes: Number of evaluation episodes
    
    Returns:
        avg_return: Average episode return
        all_returns: List of all episode returns
    """
    print("="*70)
    print("Evaluating DT with Predicted RTG")
    print("="*70)
    
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Load DT model
    cfg = DTConfig()
    dt_agent = DTSolver(state_dim, act_dim, cfg)
    dt_agent.load(dt_model_path)
    
    # Load RTG predictor
    rtg_solver = RTGPredictorSolver(state_dim)
    rtg_solver.load(rtg_model_path)
    
    all_returns = []
    
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        
        # Initialize history with single timestep
        states = np.reshape(state, (1, 1, state_dim))  # [1, 1, state_dim]
        actions = np.zeros((1, 0), dtype=np.int64)  # [1, 0] - no actions yet
        timesteps = np.zeros((1, 1), dtype=np.int64)  # [1, 1]
        
        # Predict initial RTG
        pred_rtg = rtg_solver.predict(state.reshape(state_dim))
        returns_to_go = np.reshape(pred_rtg, (1, 1, 1))  # [1, 1, 1]
        
        done = False
        ep_return = 0
        t = 0
        
        while not done:
            # Prepare input for DT (keep only context_len)
            if states.shape[1] > cfg.context_len:
                context_start = states.shape[1] - cfg.context_len
                states_in = states[:, context_start:, :]
                actions_in = actions[:, context_start:]
                rtg_in = returns_to_go[:, context_start:, :]
                timesteps_in = timesteps[:, context_start:]
            else:
                states_in = states
                actions_in = actions
                rtg_in = returns_to_go
                timesteps_in = timesteps
            
            action = dt_agent.act(rtg_in, states_in, actions_in, timesteps_in)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ep_return += reward
            
            # Update history (only if not done, to keep proper sequence)
            if not done:
                t += 1
                next_state_reshaped = np.reshape(next_state, (1, 1, state_dim))
                states = np.concatenate([states, next_state_reshaped], axis=1)
                actions = np.concatenate([actions, np.reshape(action, (1, 1))], axis=1)
                timesteps = np.concatenate([timesteps, np.reshape(t, (1, 1))], axis=1)
                
                # Predict RTG for next state
                pred_rtg = rtg_solver.predict(next_state)
                returns_to_go = np.concatenate([
                    returns_to_go,
                    np.reshape(pred_rtg, (1, 1, 1))
                ], axis=1)
            else:
                # Just record the action for the final step
                actions = np.concatenate([actions, np.reshape(action, (1, 1))], axis=1)
        
        all_returns.append(ep_return)
        if ep % 5 == 0:
            print(f"Episode {ep}/{episodes}: Return = {ep_return}")
    
    avg_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    
    print(f"\n{'='*70}")
    print(f"Predicted RTG Results:")
    print(f"  Average Return: {avg_return:.2f} ± {std_return:.2f}")
    print(f"  Min Return: {np.min(all_returns):.2f}")
    print(f"  Max Return: {np.max(all_returns):.2f}")
    print(f"{'='*70}\n")
    
    env.close()
    return avg_return, all_returns


def compare_methods(dt_model_path: str, rtg_model_path: str, 
                   target_rtgs: list = [40 ,60, 80, 100, 200, 500],
                   episodes: int = 20):
    """
    Compare DT performance with fixed RTG vs predicted RTG.
    
    Args:
        dt_model_path: Path to trained DT model
        rtg_model_path: Path to trained RTG predictor model
        target_rtgs: List of fixed RTG values to test
        episodes: Number of evaluation episodes per configuration
    """
    print("="*70)
    print("Comparing Fixed RTG vs Predicted RTG")
    print("="*70)
    
    # Evaluate with predicted RTG
    pred_avg, pred_returns = evaluate_dt_predicted_rtg(dt_model_path, rtg_model_path, episodes)
    
    # Evaluate with different fixed RTGs
    fixed_results = {}
    for target_rtg in target_rtgs:
        fixed_avg, fixed_returns = evaluate_dt_fixed_rtg(dt_model_path, target_rtg, episodes)
        fixed_results[target_rtg] = {
            'avg': fixed_avg,
            'returns': fixed_returns
        }
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Average returns comparison
    ax1 = axes[0]
    fixed_avgs = [fixed_results[rtg]['avg'] for rtg in target_rtgs]
    
    x_pos = np.arange(len(target_rtgs))
    ax1.bar(x_pos, fixed_avgs, width=0.6, alpha=0.7, label='Fixed RTG', color='steelblue')
    ax1.axhline(y=pred_avg, color='red', linestyle='--', linewidth=2, label=f'Predicted RTG (avg={pred_avg:.1f})')
    
    ax1.set_xlabel('Target RTG Value', fontsize=12)
    ax1.set_ylabel('Average Return', fontsize=12)
    ax1.set_title('Fixed RTG vs Predicted RTG Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(target_rtgs)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Distribution comparison (box plot)
    ax2 = axes[1]
    all_data = [fixed_results[rtg]['returns'] for rtg in target_rtgs] + [pred_returns]
    positions = list(range(len(target_rtgs))) + [len(target_rtgs) + 0.5]
    labels = [str(rtg) for rtg in target_rtgs] + ['Predicted']
    
    bp = ax2.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                     labels=labels, showmeans=True)
    
    # Color boxes
    for i, patch in enumerate(bp['boxes']):
        if i < len(target_rtgs):
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('red')
            patch.set_alpha(0.7)
    
    ax2.set_xlabel('RTG Configuration', fontsize=12)
    ax2.set_ylabel('Episode Returns', fontsize=12)
    ax2.set_title('Return Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(VIZ_DIR, exist_ok=True)
    save_path = os.path.join(VIZ_DIR, "rtg_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print("Comparison Results Summary:")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Avg Return':<15} {'Std Return':<15}")
    print(f"{'-'*50}")
    
    for rtg in target_rtgs:
        avg = fixed_results[rtg]['avg']
        std = np.std(fixed_results[rtg]['returns'])
        print(f"Fixed RTG={rtg:<13} {avg:<15.2f} {std:<15.2f}")
    
    pred_std = np.std(pred_returns)
    print(f"{'Predicted RTG':<20} {pred_avg:<15.2f} {pred_std:<15.2f}")
    print(f"{'='*70}")
    
    # Calculate improvement
    best_fixed_avg = max(fixed_avgs)
    improvement = ((pred_avg - best_fixed_avg) / best_fixed_avg) * 100 if best_fixed_avg > 0 else 0
    
    print(f"\nImprovement over best fixed RTG: {improvement:+.2f}%")
    print(f"Comparison plot saved to {save_path}\n")


def full_pipeline(data_path: str, dt_model_path: str, 
                 rtg_epochs: int = 50, eval_episodes: int = 20):
    """
    Run the full pipeline: train RTG predictor and compare methods.
    
    Args:
        data_path: Path to offline dataset
        dt_model_path: Path to trained DT model
        rtg_epochs: Number of epochs for RTG predictor training
        eval_episodes: Number of evaluation episodes
    """
    print("\n" + "="*70)
    print("FULL PIPELINE: RTG Predictor Training + Comparison")
    print("="*70 + "\n")
    
    # Step 1: Train RTG predictor
    rtg_solver, losses = train_rtg_predictor(data_path, epochs=rtg_epochs)
    
    # Step 2: Compare methods
    rtg_model_path = os.path.join(MODEL_DIR, "rtg_predictor.torch")
    compare_methods(dt_model_path, rtg_model_path, episodes=eval_episodes)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train RTG Predictor and evaluate with Decision Transformer"
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train_rtg', 'eval_fixed', 'eval_predicted', 'compare', 'full_pipeline'],
                       help='Execution mode')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to offline dataset (for train_rtg and full_pipeline)')
    parser.add_argument('--dt_model', type=str, default=None,
                       help='Path to trained DT model')
    parser.add_argument('--rtg_model', type=str, default=None,
                       help='Path to trained RTG predictor model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs for RTG predictor')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Evaluation episodes')
    parser.add_argument('--target_rtg', type=float, default=500.0,
                       help='Target RTG for fixed evaluation')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.data is None:
        args.data = os.path.join(DATA_DIR, "random_data.pkl")
    if args.dt_model is None:
        args.dt_model = os.path.join(MODEL_DIR, "cartpole_dt_random.torch")
    if args.rtg_model is None:
        args.rtg_model = os.path.join(MODEL_DIR, "rtg_predictor.torch")
    
    # Execute based on mode
    if args.mode == 'train_rtg':
        if not os.path.exists(args.data):
            print(f"[Error] Data file not found: {args.data}")
            print("Please run data collection first or provide valid data path.")
            return
        train_rtg_predictor(args.data, epochs=args.epochs)
    
    elif args.mode == 'eval_fixed':
        if not os.path.exists(args.dt_model):
            print(f"[Error] DT model not found: {args.dt_model}")
            return
        evaluate_dt_fixed_rtg(args.dt_model, args.target_rtg, args.episodes)
    
    elif args.mode == 'eval_predicted':
        if not os.path.exists(args.dt_model):
            print(f"[Error] DT model not found: {args.dt_model}")
            return
        if not os.path.exists(args.rtg_model):
            print(f"[Error] RTG model not found: {args.rtg_model}")
            return
        evaluate_dt_predicted_rtg(args.dt_model, args.rtg_model, args.episodes)
    
    elif args.mode == 'compare':
        if not os.path.exists(args.dt_model):
            print(f"[Error] DT model not found: {args.dt_model}")
            return
        if not os.path.exists(args.rtg_model):
            print(f"[Error] RTG model not found: {args.rtg_model}")
            return
        compare_methods(args.dt_model, args.rtg_model, episodes=args.episodes)
    
    elif args.mode == 'full_pipeline':
        if not os.path.exists(args.data):
            print(f"[Error] Data file not found: {args.data}")
            return
        if not os.path.exists(args.dt_model):
            print(f"[Error] DT model not found: {args.dt_model}")
            return
        full_pipeline(args.data, args.dt_model, args.epochs, args.episodes)


if __name__ == "__main__":
    main()
