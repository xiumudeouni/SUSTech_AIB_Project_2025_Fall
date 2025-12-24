""""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
- Trains a DQN agent and logs scores via ScoreLogger (PNG + CSV)
- Saves model to ./models/cartpole_dqn.torch
- Evaluates from a saved model (render optional)

Student reading map:
  1) train(): env loop → agent.act() → env.step() → agent.step() [Encapsulated]
  2) evaluate(): loads saved model and runs agent.act(evaluation_mode=True)
"""

from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt # 新增绘图库

from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_ppo import PPOSolver, PPOConfig
#from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
MODEL_PATH_DQN = os.path.join(MODEL_DIR, "cartpole_dqn.torch")
MODEL_PATH_PPO = os.path.join(MODEL_DIR, "cartpole_ppo.torch")

def plot_learning_curves(scores: list[int], losses: list[float], algorithm: str):
    """绘制双轴学习曲线：左轴得分，右轴收敛 Loss"""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 绘制得分 (左轴)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color='tab:blue')
    ax1.plot(scores, color='tab:blue', alpha=0.3)
    if len(scores) >= 20:
        mv_score = np.convolve(scores, np.ones(20)/20, mode='valid')
        ax1.plot(range(19, len(scores)), mv_score, color='navy', lw=2, label='Score (MA 20)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 绘制 Loss (右轴)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Loss', color='tab:red')
    ax2.plot(losses, color='tab:red', linestyle='--', alpha=0.6, label='Loss')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'Training Convergence - {algorithm.upper()}')
    fig.tight_layout()
    plt.savefig(f"{algorithm}_results.png")
    print(f"[Info] Plot saved as {algorithm}_results.png")
    plt.show()

def train(num_episodes: int = 500, terminal_penalty: bool = True, algorithm: str = "dqn"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = gym.make(ENV_NAME)
    #logger = ScoreLogger(ENV_NAME)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n

    # 建议的 PPO 参数优化
    if algorithm.lower() == "ppo":
        cfg = PPOConfig(
            lr=3e-4,           # 初始学习率
            eps_clip=0.5,      # 缩紧剪切范围增加稳定性
            update_timestep=500, # 样本量
            k_epochs=15,        # 多次迭代提升效果
            gamma=0.99          # 折扣因子
        )
        agent = PPOSolver(obs_dim, act_dim, cfg=cfg)
        model_path = MODEL_PATH_PPO
    else:
        # 在 train_plot.py 中显式指定 cartpole_dqn.py 中调优好的参数
        cfg = DQNConfig(
            lr=1e-3, 
            gamma=0.99, 
            batch_size=64, 
            eps_decay=0.99995
        )
        agent = DQNSolver(obs_dim, act_dim, cfg=cfg)
        model_path = MODEL_PATH_DQN
        
    all_scores = []
    all_losses = []

    for run in range(1, num_episodes + 1):
        # --- 新增：学习率线性衰减逻辑 ---
        # 只有在 200 回合后才开始衰减
        """if algorithm.lower() == "ppo" and run > 200:
            # 这里的 300 是剩余的回合数 (500 - 200)
            decay_ratio = max(0.1, 1.0 - (run - 200) / 300)
            lr_now = cfg.lr * decay_ratio
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = lr_now"""
        # -----------------------------

        # --- 动态收敛逻辑(850次) ---
        if algorithm.lower() == "ppo":
            # 1. 学习率线性衰减（后期降得更低）
            lr_factor = max(0.01, 1.0 - (run / num_episodes))
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = cfg.lr * lr_factor
            
            # 2. 动态收紧 clip 范围（关键：防止 500 分后震荡）
            if run > 500:
                agent.cfg.eps_clip = 0.05  # 500次后强行稳定策略
                agent.cfg.update_timestep = 1000 # 增加样本量以平滑 Loss
        # ------------------

        state, _ = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0
        episode_losses = []

        while True:
            steps += 1
            action = agent.act(state)
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminal_penalty and done:
                reward = -1.0
            
            next_state = np.reshape(next_state_raw, (1, obs_dim))
            
            # 记录每步产生的 Loss
            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None and loss > 0:
                episode_losses.append(loss)

            state = next_state
            if done:
                all_scores.append(steps)
                # 计算本回合平均 Loss 用于绘图
                avg_loss = np.mean(episode_losses) if episode_losses else (all_losses[-1] if all_losses else 0)
                all_losses.append(avg_loss)
                
                print(f"Run: {run}, Score: {steps}, Avg Loss: {avg_loss:.4f}")
                #logger.add_score(steps, run)
                break

    env.close()
    agent.save(model_path)
    
    # 训练结束，生成图表
    plot_learning_curves(all_scores, all_losses, algorithm)
    return agent

def evaluate(model_path: str | None = None,
             algorithm: str = "dqn",
             episodes: int = 5,
             render: bool = True,
             fps: int = 60):
    """
    Evaluate a trained agent in the environment using greedy policy (no ε).
    - Loads weights from disk
    - Optionally renders (pygame window)
    - Reports per-episode steps and average

    Args:
        model_path: If None, auto-pick the first .torch file under ./models
        algorithm: Reserved hook if you later support PPO/A2C agents
        episodes: Number of evaluation episodes
        render: Whether to show a window; set False for headless CI
        fps: Target frame-rate during render (sleep-based pacing)
    """
    # Resolve model path
    model_dir = MODEL_DIR
    if model_path is None:
        candidates = [f for f in os.listdir(model_dir) if f.endswith(".torch")]
        if not candidates:
            raise FileNotFoundError(f"No saved model found in '{model_dir}/'. Please train first.")
        model_path = os.path.join(model_dir, candidates[0])
        print(f"[Eval] Using detected model: {model_path}")
    else:
        print(f"[Eval] Using provided model: {model_path}")

    # Create env for evaluation; 'human' enables pygame-based rendering
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # (If you add PPO/A2C later, pick their agent classes by 'algorithm' here.)
    if algorithm.lower() == "dqn":
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif algorithm.lower() == "ppo":
        agent = PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Load trained weights
    agent.load(model_path)
    print(f"[Eval] Loaded {algorithm.upper()} model from: {model_path}")

    scores = []
    # Sleep interval to approximate fps; set 0 for fastest evaluation
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            # Greedy action (no exploration) by calling act() in evaluation mode
            action = agent.act(state, evaluation_mode=True)

            # Step env forward
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            # Slow down rendering to be watchable
            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"[Eval] Average over {episodes} episodes: {avg:.2f}")
    return scores


if __name__ == "__main__":
    # 示例：训练 PPO 并绘图
    #train(num_episodes=850, algorithm="ppo")
    evaluate(model_path="models/cartpole_ppo_solved_5.torch", algorithm="ppo", episodes=10, render=True, fps=60)
    # 示例：训练 DQN 并绘图
    #train(num_episodes=1000, algorithm="dqn")
    #evaluate(model_path="models/cartpole_dqn.torch", algorithm="dqn", episodes=10, render=True, fps=60)