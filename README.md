# 仓库地址：https://github.com/xiumudeouni/SUSTech_AIB_Project_2025_Fall
# Aritificial Intelligence B Project
Performance Study of Double DQN, PPO, and Decision Transformer with
RTG Predictor on the OpenAI Gym CartPole Environment
---
*Ling Huang* and *Sijia Chen*
---

# 目录
* [1. 框架介绍](#1-框架介绍)
* [2. 模型Benchmark](#2-模型benchmark)
* [3. 最速模型训练与复现流程](#3-模型训练与复现流程)
* [4. 可视化](#4-可视化)
* [5. 离线数据](#5-离线数据)
* [6. DT脚本详解](#6-dt脚本详解)
* [7. RTGPredictor脚本详解](#7-rtgpredictor脚本详解)

## 1. 框架介绍
### 1.1 模型
在 `agents/` 文件夹中，有不同模型的实现：
- `cartpole_dqn.py` 提供的 **DQN** 实现。
- `cartpole_double_dqn.py` 在 `cartpole_dqn.py` 基础之上的 **DDQN** 实现。
- `cartpole_ppo.py` **PPO**的实现
- `cartpole_dt.py` **Desicion Transformer** 的实现。
- `rtg_predictor.py` **RTG Predictor** 的实现。

在 `models/` 文件夹中，训练的模型分别被命名为：
- `cartpole_dqn_sovled.torch` 使用 **DQN** 训练的，平均得分500分的模型。
- `double_dqn.torch` 使用 **DDQN** 训练的，平均得分500分的模型。
- `cartpole_ppo_solved_5.torch` 使用**PPO** 训练的，平均得分500分的模型。
- `cartpole_dt_*.torch` 使用 **Desicion Transformer** 在不同数据集上训练的 RTG=500 的最优模型。
- `rtg_predictor.torch` **MLP** 加 **Bellman** 迭代，训练的用于预测Value的模型。必须搭配 `cartpole_dt_random.torch` 使用（你也可以用 `rtg_predictor_poor.torch` 搭配 `cartpole_dt_poor.torch` 使用）。

### 1.2 脚本
- `train.py` 这是提供的初始脚本。提交的脚本进行了一些修改，包括
1. 注释掉所有的 `score.logger` 
2. 注释了模型训练部分
3. 接入了PPO
主要用来评估训练好的**DQN**和**PPO**模型。
- `train_ddqn_comparison.py` 这个脚本集成了 
1. **DDQN** 的训练 
2. 与 **DQN** 的对比可视化
- `train_ppo.py` 这个脚本集成了 
1. **PPO** 的训练
2. 与 **DQN** 的可视化对比
- `train_offline_advanced.py` 这个脚本集成了
1. 不同质量的离线数据集收集
2. **DT**模型训练
3. 训练可视化
- `train_with_rtg_predictor.py` 这个脚本集成了
1. **RTG Predictor** 的训练
2. 与 Fixed RTG DT 的可视化比较。
- `run_DT.py` 这个脚本可以评价训练好的 **DT** 模型在环境中的表现。
- `visualize_datasets.py` 这个脚本用来可视化训练 **DT** 前生成的数据集。

## 2. 模型Benchmark
Benchmark采用 **线上评估**
### 2.1 DQN Benchmark
修改 `train.py` 的运行
```python
if __name__ == "__main__":
    # Example: quick training then a short evaluation
    #agent = train(num_episodes=1000, terminal_penalty=True)
    evaluate(model_path="models/cartpole_dqn_solved.torch", algorithm="dqn", episodes=100, render=False, fps=60)
``` 

### 2.2. DDQN Benchmark
修改 `train.py` 的运行
```python
if __name__ == "__main__":
    # Example: quick training then a short evaluation
    #agent = train(num_episodes=1000, terminal_penalty=True)
    evaluate(model_path="models/double_dqn.torch", algorithm="dqn", episodes=100, render=False, fps=60)
``` 

### 2.3 PPO Benchmark
修改 `train.py` 的运行
```python
if __name__ == "__main__":
    # Example: quick training then a short evaluation
    #agent = train(num_episodes=1000, terminal_penalty=True)
    evaluate(model_path="models/cartpole_ppo_solved_5.torch", algorithm="ppo", episodes=100, render=False, fps=60)
```

### 2.4 DT Benchmark
加载某一个训练好的 **DT** 模型，并且在线评估。使用 `run_DT` 脚本。对于在 *expert*，*good*，*medium* 和 *poor* 数据集上训练的模型，建议使用 `--rtg 500` 以达到接近500分的结果。对于 *random* 数据集训练的模型，推荐的值是 `--rtg 200` 以达到最优表现。
```bash
# --model_path：输入DT的模型路径
# --rtg：你需要的RTG（例如100，500）
# --episodes：评估多少轮
python run_DT.py --model_path models/cartpole_dt_good.torch --rtg 500 --episodes 100
```

### 2.5 DT with RTG Predictor Benchmark
加载某一个训练好的 **DT** 模型和 **RTG_Predictor** 模型，并且在线评估。请注意，**你选择的两个模型务必是在同一个数据集上训练的**。
使用 `train_with_rtg_predictor.py` 脚本进行评估。
```bash
# 评估 DT 模型 (使用预测 RTG)
# --mode: 运行模式 (eval_predicted)
# --dt_model: DT 模型的路径，默认 `models/cartpole_random.torch`
# --rtg_model: RTG Predictor路径，默认 `models/rtg_predictor.torch`. Predictor不带后缀的命名指的是在 random 数据集上训练的结果
# --episodes: 评估次数
python train_with_rtg_predictor.py --mode eval_predicted --episodes 1000
```

## 3. 模型训练与复现流程
### 3.1 DDQN
直接运行 `train_ddqn_comparison.py`。会在 `models/` 路径下生成 `double_dqn.torch` 和 `standard_dqn.torch`。并在 `visualization_ddqn/` 文件夹下生成可视化训练图像。
### 3.2 PPO
修改 `train_ppo.py` 的运行。分别训练PPO和DQN并且分别可视化。
```python
if __name__ == "__main__":
    # 示例：训练 PPO 并绘图
    train(num_episodes=850, algorithm="ppo")
    #evaluate(model_path="models/cartpole_ppo_solved_5.torch", algorithm="ppo", episodes=10, render=True, fps=60)
    # 示例：训练 DQN 并绘图
    #train(num_episodes=1000, algorithm="dqn")
    #evaluate(model_path="models/cartpole_dqn.torch", algorithm="dqn", episodes=10, render=True, fps=60)
```
### 3.3 DT
使用 `train_offline_advanced.py` 脚本。
```bash
# 首先收集不同质量的数据集，会自动保存在 `offline_data`文件夹下。如果这个文件夹中已经有离线数据文件，会跳过收集阶段。数据收集是通过将训练好的DQN模型`models/cartpole_dqn_solved.torch`，通过调整epsilon_greedy进行的。
python train_offline_advanced.py --mode collect_all
```
```bash
# 直接运行下面的命令进行训练和比较。默认参数训练100轮。每5轮对模型进行一次在线评估（每次评估10次）。储存RTG=500得分最高的模型（所以训练结束时候的模型并不是存储的模型）。训练好的模型会存储在`models/`文件夹下。图片会生成在`visualizations/`文件夹下。
python train_offline_advanced.py --mode compare
```

### 3.4 DT with RTG Predictor
使用 `train_with_rtg_predictor.py` 脚本。默认在 random 数据集上训练。
```bash
# 自动开始训练500轮。loss会先显著下降，然后上升，然后稳定在4e-9左右。训练过程较慢，可以将epoch改为200。训练结束后，脚本自动评估DT+rtg_Predictor和固定RTG的DT的表现。1000轮评估使得对模型的评估随机性更小。
# eopch: 训练轮数（建议大于200，我训练时使用了500）
# episodes: 评估轮数（100轮随机性很强，1000轮较为合适）
python train_with_rtg_predictor.py --mode full_pipeline --epoch 500 --episodes 1000
```

## 4. 可视化
一共有三个可视化图像的文件夹，分别是
### 4.1 `visualizations/` 
用于存储
1. `cartpole_dt_*_curves.png` **DT**在不同数据集上的训练曲线。 
2. `dt_dataset_comparison.png` **DT**训练曲线对比图。
3. `final_score_comparison.png` 数据集分布对比图。
4. `rtg_predictor_training.png` **RTG Predictor** 训练曲线图。
5. `rtg_comparison.png` 使用 **RTG Predictor** 和使用 fixed RTG 在线评估的对比图。

### 4.2 `visualizations_ddqn/`
`dqn_vs_double_dqn.png` 用于存储 **DDQN** 和 **DQN** 的训练对比图。

### 4.3 `visualization_ppo/`
用于存储 **PPO** 不同参数训练对比和与 **DQN** 的训练对比图。
1. `ppo_dqn_comparison/` **PPO** 和 **DQN** 训练曲线。
2. `ppo_parameter_comparison/` 不同参数 **PPO** 训练曲线对比。


## 5. 离线数据
`offline_data/` 这个文件夹存储用于训练 Decision Transformer 的离线轨迹数据。

### 5.1 数据格式
每个 `.pkl` 文件包含一个轨迹列表。有200条轨迹。每条轨迹是一个字典：
```python
{
    'states': np.ndarray,   # shape: [T, obs_dim]
    'actions': np.ndarray,  # shape: [T], dtype: int64
    'rewards': np.ndarray,  # shape: [T], dtype: float32
    'dones': np.ndarray,    # shape: [T], dtype: bool
}
```

### 5.2 数据收集
使用已训练的 DQN 模型收集专家数据：
```bash
python train_offline_advanced.py --mode collect_all
```
这将生成五个 `.pkl` 文件。轨迹数量是硬编码的，但是可以在脚本中修改。

### 5.3 数据统计
- **专家数据**: 使用贪婪策略的 DQN agent 收集，平均 return 应在 400-500
- **混合数据**: 可以混合不同质量的数据
- **随机数据**: 完全随机策略，用于对比实验

## 6. DT脚本详解
### 6.1 作用
Decision Transformer (DT) 是一种将强化学习问题转化为**序列建模（Sequence Modeling）**问题的算法。
在本项目中，DT 的主要作用是作为**离线强化学习（Offline RL）智能体**。它不通过传统的贝尔曼方程（如 DQN）迭代价值，而是直接根据过去的**状态（State）**、**动作（Action）**和**预期回报（Return-to-Go, RTG）**序列，来预测下一个最优动作。
这使得它能够从固定的离线数据集中学习策略，并在推理时通过指定**目标回报 (Target Return)** 来控制智能体的表现。

### 6.2 定义
DT 是一个基于 **Transformer (GPT 架构)** 的神经网络模型，定义在 `agents/cartpole_dt.py` 的 `DecisionTransformer` 类中。

*   **核心架构**：
    *   **Embedding 层**：
        *   `embed_return`: 将标量 RTG 映射到 `embed_dim`。
        *   `embed_state`: 将状态向量映射到 `embed_dim`。
        *   `embed_action`: 将动作（One-hot 编码）映射到 `embed_dim`。
        *   `embed_timestep`: 位置编码，将时间步 `t` 映射到 `embed_dim`。
    *   **序列构建**：将 Embedding 后的 `(R, s, a)` 三元组交错堆叠，形成序列 `[R_0, s_0, a_0, R_1, s_1, a_1, ...]`。
    *   **Transformer Block**：使用 GPT-2 风格的 Pre-normalization 结构（LayerNorm -> Self-Attention -> LayerNorm -> FFN）。
        *   层数 (`n_layers`): 3
        *   头数 (`n_heads`): 4
        *   嵌入维度 (`embed_dim`): 128
        *   上下文长度 (`context_len`): 20
    *   **输出头**：仅从**状态 Token** 的输出预测下一个动作的 Logits。

### 6.3 训练过程
DT 的训练采用**监督学习（Supervised Learning）**的方式，由 `DTSolver` 类管理。

1.  **数据预处理** (`OfflineDataset`)：
    *   加载离线轨迹数据。
    *   计算每个时间步的 **Return-to-Go (RTG)**：$R_t = \sum_{k=t}^{T} r_k$。
2.  **数据采样**：
    *   从轨迹中随机采样长度为 `context_len` (20) 的片段。
    *   如果片段长度不足，进行零填充 (Padding)。
3.  **前向传播**：
    *   输入：`returns_to_go`, `states`, `actions`, `timesteps`。
    *   模型预测每个时间步的动作 Logits。
4.  **损失计算**：
    *   使用 **交叉熵损失 (Cross-Entropy Loss)**。
    *   比较模型预测的动作与数据集中实际执行的动作。
5.  **优化**：
    *   优化器：**AdamW** (`lr=1e-4`, `weight_decay=1e-4`)。
    *   梯度裁剪：Norm 阈值为 1.0。

### 6.4 训练结果
*   **训练指标**：主要监控 **Training Loss** (Cross-Entropy)。
*   **评估指标**：在实际环境 (CartPole-v1) 中运行 Episode，给定初始 `target_return` (如 500)。
*   **预期表现**：
    *   在专家数据上训练的 DT 应能稳定获得 500 分。
    *   模型应具备 **条件执行能力**：给定低 RTG (如 200) 时表现较差，给定高 RTG (如 500) 时表现较好。

### 6.5 脚本模式
DT 的脚本通常包含以下模式：
*   **Train Mode (训练模式)**：加载离线数据，运行 Epoch 循环，保存模型权重。
*   **Evaluate Mode (评估模式)**：加载训练好的模型，在 Gym 环境中运行，设置初始 `target_return`（例如 500），观察实际得分。

### 6.6 脚本命令行代码
训练脚本名为 `train_offline_advanced.py`。这个脚本集成了数据集收集，模型训练，训练可视化。以下是典型的运行命令：

```bash
# 训练 DT 模型 (带可视化)
# --mode: 运行模式 (train_viz)
# --data: 离线数据路径
# --epochs: 训练轮数

# 不要用这个命令，这只是我自己测试使用的
python train_offline_advanced.py --mode train_viz --data offline_data/expert_data.pkl --epochs 100

# 收集不同质量的数据集 (用于训练 DT)
python train_offline_advanced.py --mode collect_all

# 比较 DT 在不同数据集上的表现
python train_offline_advanced.py --mode compare
```

### 6.7 复现流程

```bash
# 首先收集不同质量的数据集，会自动保存在 `offline_data`文件夹下。如果这个文件夹中已经有离线数据文件，会跳过收集阶段。数据收集是通过将训练好的DQN模型`models/cartpole_dqn_solved.torch`，通过调整epsilon_greedy进行的。
python train_offline_advanced.py --mode collect_all

# 直接运行下面的命令进行训练和比较。默认参数训练100轮。每5轮对模型进行一次在线评估。储存RTG=500，得分最高的模型（所以训练结束时候的模型并不是存储的模型）。训练好的模型会存储在`models`文件夹下。图片会生成在`visualizations`文件夹下。
python train_offline_advanced.py --mode compare

```

### 6.8 `run_DT.py`脚本的使用
这个脚本只是用来评价使用DT训练好的模型在环境中的表现。可以在控制台中看到模型具体的表现结果。复现不需要，只是供我测试的脚本。
命令行示例如下
```bash
# --model_path：输入DT的模型路径
# --rtg：你需要的RTG（例如100，500）
# --episodes：评估多少轮
python run_DT.py --model_path models/cartpole_dt_good.torch --rtg 200 --episodes 100
```

## 7. RTGPredictor脚本详解
这是本项目的创新点，是为了提升DT在差数据集上的表现。我们主要使用了`carpole_dt_random.torch`模型，并且只主要关注`random_data.pkl`数据集。

### 7.1 作用
RTG Predictor（回报预测器）在本项目中充当 **价值函数 (Value Function, $V(s)$)** 的角色。
它的主要作用是在 Decision Transformer (DT) 的推理阶段，**动态预测当前状态下的预期回报**，并将其作为 DT 的输入 `target_return`。
这解决了标准 DT 需要手动指定固定 `target_return` 的局限性，特别是在以下情况中：
1.  **状态较差时**：如果智能体处于即将失败的状态，强制要求高回报（如 500）可能导致 DT 产生不可预测的行为。RTG Predictor 能给出更现实的预期回报。
2.  **数据质量差时**：在随机或低质量数据集上训练时，RTG Predictor 能帮助 DT 更好地利用学到的策略。

### 7.2 定义
RTG Predictor 是一个基于 **多层感知机 (MLP)** 的神经网络，定义在 `agents/rtg_predictor.py` 的 `RTGPredictor` 类中。

*   **网络架构**：
    *   输入层：`Linear(state_dim, hidden_dim)` + `ReLU` + `Dropout`
    *   隐藏层：多层 `Linear` + `ReLU` + `Dropout`
    *   输出层：`Linear(hidden_dim, 1)`
*   **输入**：当前状态向量 `s_t` (CartPole 中维度为 4)。
*   **输出**：一个标量值，代表**缩放后**的预期回报。
*   **数学表达**：$V(s_t) \approx \mathbb{E}[\sum_{k=t}^{T} r_k]$

### 7.3 训练过程
与 DT 的监督学习不同，RTG Predictor 使用 **时序差分学习 (TD Learning)** 进行训练，类似于 DQN 或 Actor-Critic 中的 Critic 网络。实现位于 `RTGPredictorSolver` 类中。

1.  **数据构建**：
    *   从离线轨迹中构建转换元组 `(state, reward, next_state, done)`。
    *   使用 `TransitionDataset` 类进行批次采样。

2.  **核心机制**：
    *   **奖励缩放 (Reward Scaling)**：为了训练稳定性，奖励值被除以 `100.0`。
    *   **目标网络 (Target Network)**：使用一个通过软更新 (Soft Update, $\tau$) 滞后更新的目标网络来计算 TD Target。
    *   **TD Target 计算**：
        $$y = \frac{r}{\text{scale}} + \gamma \cdot V_{\text{target}}(s') \cdot (1 - \text{done})$$
    *   **损失函数**：使用 **Huber Loss** (比 MSE 对异常值更鲁棒) 计算预测值 $V(s)$ 与目标值 $y$ 之间的误差。
    *   **优化器**：Adam 优化器，配合 **梯度裁剪 (Gradient Clipping)** 防止梯度爆炸。

3.  **训练循环**：
    *   采样 Batch -> 计算 Target -> 计算 Loss -> 反向传播 -> 软更新 Target Network。

### 7.4 训练结果
*   **指标**：主要监控 **TD Loss (Huber Loss)**。随着训练进行，Loss 应逐渐下降并趋于稳定。
*   **产出**：
    *   模型文件：`models/rtg_predictor.torch`
    *   损失曲线：`visualizations/rtg_predictor_training.png`
*   **实际效果**：训练好的模型能够根据当前杆子的角度和位置，准确预测出“还能坚持多久”（即预期得分）。

### 7.5 脚本模式
脚本 `train_with_rtg_predictor.py` 提供了多种运行模式：

*   `train_rtg`: **训练模式**。加载离线数据，训练 RTG Predictor。
*   `eval_fixed`: **基准评估**。使用固定的 `target_return` (如 500) 运行 DT。
*   `eval_predicted`: **动态评估**。使用 RTG Predictor 实时预测的值作为 DT 的 `target_return`。
*   `compare`: **对比模式**。同时运行上述两种评估并绘制对比图。
*   `full_pipeline`: **全流程**。先训练 RTG Predictor，然后进行对比评估。

### 7.6 脚本命令行代码
相关的实现包含在 `train_with_rtg_predictor.py` 中：


```bash
# 训练 RTG 预测器
# --mode: 运行模式 (train_rtg)
# --data: 离线数据路径
# --epochs: 训练轮数
python train_with_rtg_predictor.py --mode train_rtg --data offline_data/random_data.pkl --epochs 200

# 评估 DT 模型 (使用固定 RTG)
# --mode: 运行模式 (eval_fixed)
# --dt_model: DT 模型路径
# --target_rtg: 目标回报
python train_with_rtg_predictor.py --mode eval_fixed --dt_model models/cartpole_dt_random.torch --target_rtg 500

# 评估 DT 模型 (使用预测 RTG)
# --mode: 运行模式 (eval_predicted)
# --episodes: 评估次数
python train_with_rtg_predictor.py --mode eval_predicted --episodes 1000

# 完整流程：训练 RTG 预测器并比较
python train_with_rtg_predictor.py --mode full_pipeline --epoch 500 --episodes 1000
```

### 7.7 复现流程
直接运行命令行
```bash
# 自动开始训练500轮。loss会先显著下降，然后上升，然后稳定在4e-9左右。训练过程较慢，可以将epoch改为200。训练结束后，脚本自动评估DT+rtg_Predictor和固定RTG的DT的表现。1000轮评估使得对模型的评估随机性更小。
# eopch: 训练轮数（建议大于200，我训练时使用了500）
# episodes: 评估轮数（100轮随机性很强，1000轮较为合适）
python train_with_rtg_predictor.py --mode full_pipeline --epoch 500 --episodes 1000
```