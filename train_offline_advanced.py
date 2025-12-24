"""
Advanced Offline Training Script for Decision Transformer
---------------------------------------------------------
- Visualize DT training process with loss curves and eval metrics
- Compare DT performance on different quality datasets
- Generate comprehensive comparison plots

Usage:
    # Train with visualization
    python train_offline_advanced.py --mode train_viz --data offline_data/expert_data.pkl --epochs 100
    
    # Collect different quality datasets
    python train_offline_advanced.py --mode collect_all
    
    # Compare DT on different datasets
    python train_offline_advanced.py --mode compare
"""

from __future__ import annotations
import os
import argparse
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-interactive plotting

from agents.cartpole_dt import DTSolver, DTConfig, OfflineDataset
from agents.cartpole_dqn import DQNSolver, DQNConfig

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
DATA_DIR = "offline_data"
VIZ_DIR = "visualizations"


def collect_quality_datasets(dqn_model_path: str = None):
    """
    Collect datasets of different qualities:
    1. Expert data (epsilon=0.0, greedy)
    2. Good data (epsilon=0.3, mostly greedy)
    3. Medium data (epsilon=0.5, balanced)
    4. Poor data (epsilon=0.7, more random)
    5. Random data (epsilon=1.0, completely random)
    """
    if dqn_model_path is None:
        dqn_model_path = os.path.join(MODEL_DIR, "cartpole_dqn_solved.torch")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load DQN agent
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    agent.load(dqn_model_path)
    
    # Different quality levels
    quality_configs = {
        'expert': {'epsilon': 0.0, 'episodes': 200, 'desc': 'Expert (greedy)'},
        'good': {'epsilon': 0.3, 'episodes': 200, 'desc': 'Good (ε=0.3)'},
        'medium': {'epsilon': 0.5, 'episodes': 200, 'desc': 'Medium (ε=0.5)'},
        'poor': {'epsilon': 0.7, 'episodes': 200, 'desc': 'Poor (ε=0.7)'},
        'random': {'epsilon': 1.0, 'episodes': 200, 'desc': 'Random (ε=1.0)'},
    }
    
    for quality_name, config in quality_configs.items():
        output_path = os.path.join(DATA_DIR, f"{quality_name}_data.pkl")
        
        # Skip if already exists
        if os.path.exists(output_path):
            print(f"[Collect] {output_path} already exists, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Collecting {config['desc']} data...")
        print(f"{'='*60}")
        
        agent.exploration_rate = config['epsilon']
        trajectories = []
        
        for ep in range(1, config['episodes'] + 1):
            state, _ = env.reset(seed=ep + 1000)  # Different seed range
            state = np.reshape(state, (1, obs_dim))
            done = False
            
            states_buffer = []
            actions_buffer = []
            rewards_buffer = []
            dones_buffer = []
            
            while not done:
                # Act with specified epsilon
                action = agent.act(state, evaluation_mode=False)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                states_buffer.append(state.squeeze(0))
                actions_buffer.append(action)
                rewards_buffer.append(reward)
                dones_buffer.append(done)
                
                state = np.reshape(next_state, (1, obs_dim))
            
            trajectory = {
                'states': np.array(states_buffer),
                'actions': np.array(actions_buffer, dtype=np.int64),
                'rewards': np.array(rewards_buffer, dtype=np.float32),
                'dones': np.array(dones_buffer, dtype=bool),
            }
            trajectories.append(trajectory)
            
            if ep % 50 == 0:
                avg_return = np.mean([t['rewards'].sum() for t in trajectories[-50:]])
                print(f"  Episode {ep}/{config['episodes']}, Avg return: {avg_return:.1f}")
        
        # Save
        with open(output_path, 'wb') as f:
            pickle.dump(trajectories, f)
        
        avg_return = np.mean([t['rewards'].sum() for t in trajectories])
        avg_length = np.mean([len(t['states']) for t in trajectories])
        print(f"[Saved] {output_path}")
        print(f"  Total trajectories: {len(trajectories)}")
        print(f"  Average return: {avg_return:.1f}")
        print(f"  Average length: {avg_length:.1f}")
    
    env.close()
    print(f"\n{'='*60}")
    print("All datasets collected!")
    print(f"{'='*60}")


def train_dt_with_visualization(
    data_path: str,
    epochs: int = 100,
    target_return: float = 500.0,
    eval_interval: int = 5,
    save_name: str = None,
):
    """
    Train Decision Transformer with detailed visualization.
    
    Returns:
        dict with training history
    """
    print(f"[Train DT] Loading offline data from {data_path}")
    
    # Load data
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    data_stats = {
        'num_trajectories': len(trajectories),
        'avg_length': np.mean([len(t['states']) for t in trajectories]),
        'avg_return': np.mean([t['rewards'].sum() for t in trajectories]),
        'min_return': np.min([t['rewards'].sum() for t in trajectories]),
        'max_return': np.max([t['rewards'].sum() for t in trajectories]),
    }
    
    print(f"[Data Stats]")
    print(f"  Trajectories: {data_stats['num_trajectories']}")
    print(f"  Avg length: {data_stats['avg_length']:.1f}")
    print(f"  Avg return: {data_stats['avg_return']:.1f}")
    print(f"  Return range: [{data_stats['min_return']:.1f}, {data_stats['max_return']:.1f}]")
    
    # Create dataset and agent
    cfg = DTConfig()
    dataset = OfflineDataset(trajectories, context_len=cfg.context_len)
    
    obs_dim = trajectories[0]['states'].shape[1]
    act_dim = 2
    agent = DTSolver(obs_dim, act_dim, cfg=cfg)
    
    print(f"[Model] Parameters: {sum(p.numel() for p in agent.model.parameters())}")
    print(f"[Model] Device: {agent.device}")
    
    # Training history
    history = {
        'epochs': [],
        'train_losses': [],
        'eval_returns': [],
        'eval_returns_200': [],  # Conditioned on RTG=200
        'eval_returns_300': [],  # Conditioned on RTG=300
        'eval_returns_500': [],  # Conditioned on RTG=500
        'data_stats': data_stats,
    }
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    best_eval_return = -float('inf')
    
    for epoch in range(1, epochs + 1):
        # Training
        epoch_losses = []
        steps_per_epoch = 100
        
        for _ in range(steps_per_epoch):
            rtg, states, actions, timesteps = dataset.sample_batch(cfg.batch_size)
            loss = agent.train_step(rtg, states, actions, timesteps)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        history['epochs'].append(epoch)
        history['train_losses'].append(avg_loss)
        
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}", end='')
        
        # Evaluation
        if epoch % eval_interval == 0:
            # Evaluate with different target returns
            eval_200 = evaluate_dt_silent(agent, target_return=200.0, episodes=10)
            eval_300 = evaluate_dt_silent(agent, target_return=300.0, episodes=10)
            eval_500 = evaluate_dt_silent(agent, target_return=500.0, episodes=10)
            
            history['eval_returns_200'].append(eval_200)
            history['eval_returns_300'].append(eval_300)
            history['eval_returns_500'].append(eval_500)
            
            print(f" | Eval (RTG=200/300/500): {eval_200:.0f}/{eval_300:.0f}/{eval_500:.0f}")
            
            # Save best model (using RTG=500 as criterion)
            if eval_500 > best_eval_return:
                best_eval_return = eval_500
                model_name = save_name if save_name else "cartpole_dt.torch"
                model_path = os.path.join(MODEL_DIR, model_name)
                agent.save(model_path)
                print(f"    [Saved] New best model: {eval_500:.0f}")
        else:
            print()
    
    print(f"[Training Complete] Best eval return: {best_eval_return:.1f}")
    
    # Plot training curves
    plot_name = save_name.replace('.torch', '') if save_name else 'dt_training'
    plot_training_history(history, save_path=os.path.join(VIZ_DIR, f"{plot_name}_curves.png"))
    
    return history


def evaluate_dt_silent(agent, target_return=500.0, episodes=5, max_steps=1000):
    """Silent evaluation (no prints)."""
    env = gym.make(ENV_NAME)
    episode_returns = []
    
    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10000 + ep)
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
            
            rtg = np.array(returns_to_go[start_idx:]).reshape(1, -1, 1)
            s = np.array(states[start_idx:]).reshape(1, -1, agent.obs_dim)
            a = np.array(actions[start_idx:] if actions else [0]).reshape(1, -1)
            t = np.array(timesteps[start_idx:]).reshape(1, -1)
            
            action = agent.act(rtg, s, a, t)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            episode_return += reward
            
            states.append(next_state)
            actions.append(action)
            timesteps.append(steps)
            returns_to_go.append(returns_to_go[-1] - reward)
        
        episode_returns.append(episode_return)
    
    env.close()
    return np.mean(episode_returns)


def plot_training_history(history, save_path):
    """Plot training loss and evaluation returns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1 = axes[0]
    ax1.plot(history['epochs'], history['train_losses'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Evaluation returns
    ax2 = axes[1]
    eval_epochs = history['epochs'][::len(history['epochs'])//len(history['eval_returns_500'])][:len(history['eval_returns_500'])]
    
    ax2.plot(eval_epochs, history['eval_returns_200'], 'g-', marker='o', label='RTG=200', linewidth=2)
    ax2.plot(eval_epochs, history['eval_returns_300'], 'orange', marker='s', label='RTG=300', linewidth=2)
    ax2.plot(eval_epochs, history['eval_returns_500'], 'r-', marker='^', label='RTG=500', linewidth=2)
    
    # Add horizontal line for data quality
    data_return = history['data_stats']['avg_return']
    ax2.axhline(y=data_return, color='purple', linestyle='--', linewidth=2, label=f'Data Avg ({data_return:.0f})')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Average Return', fontsize=12)
    ax2.set_title('Evaluation Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Visualization] Saved to {save_path}")


def compare_datasets():
    """
    Train DT on all available datasets and compare performance.
    """
    print("\n" + "="*60)
    print("COMPARING DT PERFORMANCE ON DIFFERENT QUALITY DATASETS")
    print("="*60 + "\n")
    
    # Find all datasets
    datasets = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('_data.pkl'):
            quality = filename.replace('_data.pkl', '')
            datasets[quality] = os.path.join(DATA_DIR, filename)
    
    if not datasets:
        print("[Error] No datasets found! Run --mode collect_all first.")
        return
    
    print(f"Found {len(datasets)} datasets: {list(datasets.keys())}\n")
    
    # Train on each dataset
    results = {}
    
    for quality, data_path in sorted(datasets.items()):
        print(f"\n{'='*60}")
        print(f"Training on {quality.upper()} dataset")
        print(f"{'='*60}")
        
        model_name = f"cartpole_dt_{quality}.torch"
        history = train_dt_with_visualization(
            data_path=data_path,
            epochs=100,
            target_return=500.0,
            eval_interval=5,
            save_name=model_name,
        )
        
        results[quality] = history
    
    # Create comparison plots
    plot_dataset_comparison(results, save_path=os.path.join(VIZ_DIR, "dt_dataset_comparison.png"))
    
    # Print summary table
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Dataset':<12} {'Data Avg':<12} {'Final (200)':<12} {'Final (300)':<12} {'Final (500)':<12}")
    print("-" * 60)
    
    for quality in sorted(results.keys()):
        h = results[quality]
        data_avg = h['data_stats']['avg_return']
        final_200 = h['eval_returns_200'][-1] if h['eval_returns_200'] else 0
        final_300 = h['eval_returns_300'][-1] if h['eval_returns_300'] else 0
        final_500 = h['eval_returns_500'][-1] if h['eval_returns_500'] else 0
        
        print(f"{quality:<12} {data_avg:<12.1f} {final_200:<12.1f} {final_300:<12.1f} {final_500:<12.1f}")
    
    print("="*60)


def plot_dataset_comparison(results, save_path):
    """Plot comparison of DT performance across different datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {
        'expert': 'red',
        'good': 'orange',
        'medium': 'green',
        'poor': 'blue',
        'random': 'purple',
    }
    
    # Plot 1: Training loss
    ax1 = axes[0, 0]
    for quality, history in sorted(results.items()):
        color = colors.get(quality, 'black')
        ax1.plot(history['epochs'], history['train_losses'], 
                color=color, label=quality.capitalize(), linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss by Dataset Quality', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Eval performance (RTG=500)
    ax2 = axes[0, 1]
    for quality, history in sorted(results.items()):
        color = colors.get(quality, 'black')
        eval_epochs = history['epochs'][::len(history['epochs'])//len(history['eval_returns_500'])][:len(history['eval_returns_500'])]
        ax2.plot(eval_epochs, history['eval_returns_500'], 
                color=color, marker='o', label=quality.capitalize(), linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Average Return', fontsize=12)
    ax2.set_title('Evaluation Performance (RTG=500)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Data quality vs final performance
    ax3 = axes[1, 0]
    qualities = sorted(results.keys())
    data_avgs = [results[q]['data_stats']['avg_return'] for q in qualities]
    final_returns = [results[q]['eval_returns_500'][-1] if results[q]['eval_returns_500'] else 0 for q in qualities]
    
    x_pos = np.arange(len(qualities))
    width = 0.35
    ax3.bar(x_pos - width/2, data_avgs, width, label='Data Avg Return', alpha=0.8, color='skyblue')
    ax3.bar(x_pos + width/2, final_returns, width, label='DT Final Return (RTG=500)', alpha=0.8, color='salmon')
    ax3.set_xlabel('Dataset Quality', fontsize=12)
    ax3.set_ylabel('Average Return', fontsize=12)
    ax3.set_title('Data Quality vs DT Performance', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([q.capitalize() for q in qualities])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Return conditioned on different RTGs (final epoch)
    ax4 = axes[1, 1]
    rtg_values = [200, 300, 500]
    x_pos = np.arange(len(qualities))
    width = 0.25
    
    for i, rtg in enumerate(rtg_values):
        key = f'eval_returns_{rtg}'
        returns = [results[q][key][-1] if results[q][key] else 0 for q in qualities]
        ax4.bar(x_pos + i*width - width, returns, width, 
               label=f'RTG={rtg}', alpha=0.8)
    
    ax4.set_xlabel('Dataset Quality', fontsize=12)
    ax4.set_ylabel('Average Return', fontsize=12)
    ax4.set_title('Effect of Target RTG on Performance', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos + width/2)
    ax4.set_xticklabels([q.capitalize() for q in qualities])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Comparison] Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Advanced DT training with visualization")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['collect_all', 'train_viz', 'compare'],
                        help='Mode: collect all quality datasets, train with viz, or compare')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to offline data (for train_viz mode)')
    parser.add_argument('--dqn_model', type=str, default=None,
                        help='Path to DQN model for data collection')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Evaluation interval')
    
    args = parser.parse_args()
    
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    if args.mode == 'collect_all':
        # Collect datasets of all qualities
        collect_quality_datasets(args.dqn_model)
    
    elif args.mode == 'train_viz':
        # Train with visualization
        if args.data is None:
            args.data = os.path.join(DATA_DIR, "expert_data.pkl")
        
        if not os.path.exists(args.data):
            print(f"[Error] Data file not found: {args.data}")
            return
        
        train_dt_with_visualization(
            data_path=args.data,
            epochs=args.epochs,
            eval_interval=args.eval_interval,
        )
    
    elif args.mode == 'compare':
        # Compare DT on all datasets
        compare_datasets()


if __name__ == "__main__":
    main()
