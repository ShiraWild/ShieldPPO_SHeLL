import os
import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seeds = ["1", "3", "5"]

results = {}

env_reward_bound = {}

for seed in seeds:
    results[seed] = {}
    filename = f"models/Test_add_frames/from_GPU/stats_k={seed}.log"
    res = torch.load(filename)
    time_steps, rewards, costs, tasks = res
    time_steps, rewards, costs, tasks = map(np.array, [time_steps, rewards, costs, tasks])

    unique = np.unique(tasks)
    for task in unique:
        idx = (tasks == task)
        t = time_steps[idx]
        r = rewards[idx]
        c = costs[idx]
        if task not in env_reward_bound:
            env_reward_bound[task] = (r.min(), r.max())
        else:
            past = env_reward_bound[task]
            env_reward_bound[task] = (min(r.min(), past[0]), max(r.max(), past[1]))
        results[seed][task] = (t, r, c)

# Create a single plot with six figures
fig, axs = plt.subplots(2, 3, figsize=(18, 8))

colors = [f"C{i}" for i in range(10)]

def smooth(x):
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i-50):min(n, i+50)].mean()
    return b

for j, seed in enumerate(seeds):
    for i, task in enumerate(env_reward_bound.keys()):
        min_l = np.min([len(results[seed][task][0]) for seed in seeds])
        t = np.stack([results[seed][task][0][:min_l] for seed in seeds])
        r = np.stack([results[seed][task][1][:min_l] for seed in seeds])
        c = np.stack([results[seed][task][2][:min_l] for seed in seeds])
        t = t.mean(0)
        low, up = env_reward_bound[task]

        r_mean = r.mean(axis=0)  # Mean across all seeds
        r_std = r.std(axis=0)    # Standard deviation across all seeds

        r_normalized = (r_mean - low) / (up - low + 1e-8)

        r_std_normalized = r_std / (up - low + 1e-8)

        r_smoothed = smooth(r_normalized)

        # Plot reward
        axs[0, j].plot(t, r_smoothed, label=f"Seed {seed}", color=colors[i], linewidth=2.0)
        axs[0, j].fill_between(t, r_smoothed - r_std_normalized, r_smoothed + r_std_normalized, color=colors[i], alpha=0.1)
        axs[0, j].set_title(f"Reward (Seed {seed})", fontsize=10)

        # Plot cost
        axs[1, j].plot(t, c.mean(0), label=f"Seed {seed}", color=colors[i], linewidth=2.0)
        axs[1, j].fill_between(t, c.mean(0)-c.std(0), c.mean(0)+c.std(0), color=colors[i], alpha=0.1)
        axs[1, j].set_title(f"Cost (Seed {seed})", fontsize=10)

# Set common labels and legends
axs[0, 0].set_ylabel("Normalized Reward", fontsize=10)
axs[0, 0].legend(fontsize=8)
axs[1, 0].set_ylabel("Cost", fontsize=10)
fig.text(0.5, 0.0, 'Training steps', ha='center', fontsize=10)
fig.tight_layout()

# Save the single image with all six figures
plt.savefig("shik.png", dpi=500)
