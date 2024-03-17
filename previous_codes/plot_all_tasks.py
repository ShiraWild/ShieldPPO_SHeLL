import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
seeds = [
    10719,
    39112,
    69235,
    93759,
    49230,
]
"""


files = ['models/NO_piapua/500k_steps/stats.log', 'models/PIAPUA_2609/stats.log']
"""
tasks_names = {'EgoMergeEnvNoNormalization': 'Ego Vehicle Merging',
               'HighwayEnvFastNoNormalization': 'Driving on a Highway',
               'IntersectionEnvNoNormalization': 'Turning at an Intersection',
               'MergeEnvNoNormalization': 'Other Vehicle Merging',
               'TwoWayEnvNoNormalization': 'Two-way Road',
               'UTurnEnvNoNormalization': 'U-turn'}
"""
tasks_names = {'HighwayEnvFastNoNormalization': 'Driving on a Highway'}

#methods = ["PPO", "ShieldPPO", 'PPOCaR',"PPO-Lagrangian","CPO"]
methods = ["ShieldPPO"]
results = {}

env_reward_bound = {}


def transform(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


for m in methods:
    results[m] = {}
    for file in files:
        results[m][file] = {}
        if m == 'PPO-RS':
            with open(f"models2/{m}_{seed}/test.txt", 'r') as f:
                time_steps = []
                rewards = []
                costs = []
                tasks = []
                lines = f.readlines()
                for line in lines:
                    res = [v for v in line.strip().split("\t")]
                    time_steps.append(int(res[0]))
                    rewards.append(float(res[1]))
                    costs.append(float(res[2]))
                    tasks.append(res[3])

        else:
            res = torch.load(file)
            time_steps = res[0]
            rewards = res[1]
            costs = res[2]
            tasks = res[3]
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
            results[m][file][task] = (t, r, c)

# plot
fig, axs = plt.subplots(2, 2, figsize=(15, 5), sharex=True)

colors = [f"C{i}" for i in range(10)]


def smooth(x):
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i - 50):min(n, i + 50)].mean()
    return b


for j, m in enumerate(methods):
    for i, task in enumerate(env_reward_bound.keys()):
        min_l = np.min([len(x) for x in [results[m][file][task][0] for file in files]])
        t = np.stack([results[m][file][task][0][:min_l] for file in files])
        r = np.stack([results[m][file][task][1][:min_l] for file in files])
        c = np.stack([results[m][file][task][2][:min_l] for file in files])
        t = t.mean(0)
        for file in range(len(files)):
            r[file] = smooth(r[file])
            c[file] = smooth(c[file])
        reward_std = r.std(0)
        axs[i, 0].plot(t, r.mean(0), label=m, color=colors[j], linewidth=2.0)
        axs[i, 0].fill_between(t, r.mean(0) - reward_std, r.mean(0) + reward_std, color=colors[j], alpha=0.1)
        axs[i, 1].plot(t, c.mean(0), label=m, color=colors[j], linewidth=2.0)
        axs[i, 1].fill_between(t, c.mean(0) - c.std(0), c.mean(0) + c.std(0), color=colors[j], alpha=0.1)
        xticks = ticker.MaxNLocator(6)
        axs[i, 0].xaxis.set_major_locator(xticks)
        axs[i, 1].xaxis.set_major_locator(xticks)
        axs[i, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        axs[i, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

axs[0, 0].set_ylabel("Mean Episodic Return", fontsize=12)
axs[0,1].set_ylabel("Mistake rate", fontsize=12)
for i, task in enumerate(env_reward_bound.keys()):
    axs[0, i].set_title(tasks_names[task])


fig.text(0.5, 0.0, 'Training steps', ha='center', fontsize=12)

plt.tight_layout()

plt.savefig("results_single_agent.png", dpi=1000)
