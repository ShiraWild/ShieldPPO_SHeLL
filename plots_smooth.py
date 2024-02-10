# this file is for one plot only (one file no_lstm_stats.log)

import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the file name of the log file to plot
#file_names = ["models/Test_add_frames/from_GPU/stats_k=1.log", "models/Test_add_frames/from_GPU/stats_k=3.log", "models/Test_add_frames/from_GPU/stats_k=5.log"]
file_names = ["models/Test_MaskingTres/from_GPU/stats_tresh=0K.log", "models/no_lstm_stats.log"]
files = {}

# Smooth the rewards and costs for visualization
def smooth(x):
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i - 50):min(n, i + 50)].mean()
    return b

fig, axs = plt.subplots(len(file_names), 2, figsize=(7.5, 4*len(file_names)))
plt.subplots_adjust(hspace=50)

#labels = ['last_states = 1 ', 'last_states = 3 ', 'last_states = 5 ']
#labels = ['no masking', 'masking = 50 K', 'masking = 100K', "masking = 200K"]
labels = ['LSTM Architecture','Paper Architecture']

for i, file_name in enumerate(file_names):
    log = torch.load(file_name, encoding='utf-8')
    time_steps, rewards, costs, tasks = log
    time_steps, rewards, costs, tasks = map(np.array, [time_steps, rewards, costs, tasks])

    rewards = smooth(rewards)
    costs = smooth(costs)
    # Plot rewards in the first column
    axs[i, 0].plot(time_steps, rewards, label="Average Reward", color='blue', linewidth=2.0)
    axs[i, 0].fill_between(time_steps, rewards - rewards.std(), rewards + rewards.std(), color='blue', alpha=0.1)
    axs[i, 0].set_xlabel("Training Steps")
    axs[i, 0].set_ylabel("Smoothed Reward")
    axs[i, 0].set_title(f"Training Progress (Rewards) " + labels[i])
    axs[i, 0].legend()

    # Plot costs in the second column
    axs[i, 1].plot(time_steps, costs, label="Average Cost", color='red', linewidth=2.0)
    axs[i, 1].fill_between(time_steps, costs - costs.std(), costs + costs.std(), color='red', alpha=0.1)
    axs[i, 1].set_xlabel("Training Steps")
    axs[i, 1].set_ylabel("Smoothed Cost")
    axs[i, 1].set_title(f"Training Progress (Costs) " + labels[i])
    axs[i, 1].legend()

# Adjust subplot layout
plt.tight_layout()

#path_to_save = "models/Test_add_frames/from_GPU/plots_smooth.png"
path_to_save = "models/Architecture_comparison.png"
# Save the figure
plt.savefig(path_to_save, dpi=100)





