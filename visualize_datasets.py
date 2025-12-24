import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

DATA_DIR = "offline_data"
VIZ_DIR = "visualizations"
DATASETS = {
    "Expert": "expert_data.pkl",
    "Good": "good_data.pkl",
    "Medium": "medium_data.pkl",
    "Poor": "poor_data.pkl",
    "Random": "random_data.pkl"
}

def load_dataset(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_final_scores(data):
    """
    获取每个 episode 的最终得分（即每个 episode 的 reward 总和）
    """
    return [np.sum(traj['rewards']) for traj in data]

def main():
    os.makedirs(VIZ_DIR, exist_ok=True)
    scores = {}
    for name, filename in DATASETS.items():
        data = load_dataset(filename)
        if data:
            scores[name] = get_final_scores(data)
            print(f"{name}: 平均得分 = {np.mean(scores[name]):.2f}")
    # 可视化
    plt.figure(figsize=(8, 5))
    order = ["Expert", "Good", "Medium", "Poor", "Random"]
    data_to_plot = [scores[n] for n in order if n in scores]
    labels = [n for n in order if n in scores]
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel("Final Return")
    plt.title("Distribution of Final Returns Across Datasets")
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(VIZ_DIR, "final_scores_comparison.png")
    plt.savefig(save_path)
    print(f"已保存图片: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
