# Decision Transformer (DT) for CartPole - Offline RL

这个文件夹存储用于训练 Decision Transformer 的离线轨迹数据。

## 数据格式

每个 `.pkl` 文件包含一个轨迹列表，每条轨迹是一个字典：

```python
{
    'states': np.ndarray,   # shape: [T, obs_dim]
    'actions': np.ndarray,  # shape: [T], dtype: int64
    'rewards': np.ndarray,  # shape: [T], dtype: float32
    'dones': np.ndarray,    # shape: [T], dtype: bool
}
```

## 数据收集

使用已训练的 DQN 模型收集专家数据：

```bash
python train_offline.py --mode collect --episodes 200
```

这将生成 `expert_data.pkl` 文件。

## 数据统计

- **专家数据**: 使用贪婪策略的 DQN agent 收集，平均 return 应在 400-500
- **混合数据**: 可以混合不同质量的数据（expert + random）
- **随机数据**: 完全随机策略，用于对比实验
