import matplotlib

matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import torch
import matplotlib.cm as cm  # Import the color map module

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

# each input for 'point in time' is state-action pair, so each sequence len is k_last_states length and each batch_size is 1.
# we have batch_size = 1 which is a disavantages.


################################## set device ##################################

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


# Relevant functions

def smooth(x, window_size=1000):
    x = np.array(x)
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i - window_size):i].mean()
    return b


def create_loss_stats_df(data_path, network):
    log_stats = torch.load(data_path)
    log_stats_df = pd.DataFrame([(key, *value) for key, value in log_stats.items()],
                                columns=["Time Step", "Episode", "Step in Episode", network + " Loss"])
    return log_stats_df



def plot_loss(loss_stats_path, legends, save_path, network, log_scale = False):
    # Create a list of colors for each dataframe
    colors = cm.rainbow(np.linspace(0, 1, len(loss_stats_path)))
  # Iterate over each dataframe, legend, and color
    for i, path in enumerate(loss_stats_path):
        df = create_loss_stats_df(path + ".log", network)
        df.to_csv(path + ".csv")
        plt.plot(df['Time Step'], smooth(df[network + ' Loss']), label= legends[i], color= colors[i], alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel(network + ' Loss')
    plt.legend()
    if log_scale:
        plt.yscale('log')
    plt.savefig(save_path + "/" + network + "Loss over Time")
    plt.title(network+ " Loss over Time")
    plt.clf()


def plot_reward_collisions(agent_stats_paths_lst, agent_stats_evaluation_paths_lst, save_path, labels_lst):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = cm.rainbow(np.linspace(0, 1, len(agent_stats_paths_lst)))
    # Plot Training Rewards and Collisions
    for i, path in enumerate(agent_stats_paths_lst):
        df = torch.load(path)
        # Plot Rewards
        axes[0, 0].plot(df[0], smooth(df[1]), label=labels_lst[i], color=colors[i], alpha=0.7)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards over Time')
        # Plot Collisions (Costs)
        axes[0, 1].plot(df[0], smooth(df[2]), color=colors[i])
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Collisions')
        axes[0, 1].set_title("Training Collisions over Time")

    # Plot Evaluation Rewards and Collisions
    for i, eval_path in enumerate(agent_stats_evaluation_paths_lst):
        eval_df = torch.load(eval_path)
        # Plot Rewards
        axes[1, 0].plot(eval_df[0], smooth(eval_df[1]), label= labels_lst[i], color=colors[i], alpha=0.7)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Evaluation Rewards over Time')
        # Plot Collisions (Costs)
        axes[1, 1].plot(eval_df[0], smooth(eval_df[2]), color= colors[i])
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Collisions')
        axes[1, 1].set_title("Evaluation Collisions over Time")

    # Adjust layout and save the figure
    plt.legend(labels=labels_lst, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(labels=labels_lst, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.suptitle("Training and Evaluation Costs and Rewards over Time", y=1.02)
    plt.savefig(save_path + "/combined-cost-rewards-over-time.png")
    plt.show()
    plt.clf()


# Create plots for Shield / Generator networks
def create_plots(loss_stat_paths_lst, save_path, labels, type_of_samples,  network):
    # predictions & loss statistics
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    colors = cm.rainbow(np.linspace(0, 1, len(loss_stat_paths_lst)))
    for i, path in enumerate(loss_stat_paths_lst):
        first_line, sec_line = ['Len of ' + type_of_samples + ' Samples', 'Avg Prediction for ' + type_of_samples + ' Samples']
        df = create_loss_stats_df(path + ".log", network)
        df.to_csv(path + ".csv")
        axes[0].plot(df['Time Step'], df[first_line], label=first_line, color = colors[i])
        axes[0].set_ylabel(first_line)
        axes[1].plot(df['Time Step'], df[sec_line], label=sec_line, color = colors[i])
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('sec_line')
    plt.title(network + " " + type_of_samples + "samples")
    plt.legend(labels=legend_lst)
    plt.tight_layout()
    plt.savefig(save_path + "/" + network + " " + type_of_samples + " samples.png")
    plt.clf()

       # plot_loss(loss_stats_df, save_path[i] + network + "_loss_over_time.png", network)



# 1. TODO - CHANGE base_path according to instructions. for example: "models/13_01_ShieldPPO_NO_GAN/safety_treshold_0.05_compare_safety_scores/"

base_path = ["models/03.02/GAN/agent_ShieldPPO_check_gan_performance(GAN)/", "models/03.02/NO-GAN/agent_ShieldPPO_check_gan_performance(GAN)/"]

# 2. TODO

legend_lst = ["GAN", "NO GAN"]

# 3.TODO - complete save dir name - the name of directory to save the plots, for example "NO-GAN_compare-safety-scores"
dir_name = "plots_0302_changes_in_update"

"----------------------------------------------------------------------------"


save_path = [p + "plots/" for p in base_path]  # ends with /plots/


shield_loss_stat_paths_lst = [p + "shield_loss_stats" for p in base_path]  # add more (without ".log")


gen_loss_stat_paths_lst = [p + "gen_loss_stats" for p in base_path]   # add more (without ".log")


agent_stats_paths_lst = [p + "stats.log" for p in base_path]  # add more, just an example (ends with "stats.log")

agent_stats_evaluation_paths_lst = [p + "stats_renv.log" for p in base_path]  # add more, just an example (ends with "stats.log")



save_dir = "models/" + dir_name

os.makedirs(save_dir, exist_ok=True)


for new_path in save_path:
    os.makedirs(new_path, exist_ok=True)





plot_reward_collisions(agent_stats_paths_lst, agent_stats_evaluation_paths_lst, save_dir, legend_lst)


#create_plots(shield_loss_stat_paths_lst, save_dir , legend_lst, "Pos", "Shield")
#create_plots(shield_loss_stat_paths_lst, save_dir, legend_lst, "Neg", "Shield")


#create_plots(gen_loss_stat_paths_lst, save_dir , legend_lst, "Pos", "Gen")
#create_plots(gen_loss_stat_paths_lst, save_dir, legend_lst, "Neg", "Gen")
plot_loss(shield_loss_stat_paths_lst, legend_lst, save_dir, "Shield")
plot_loss(gen_loss_stat_paths_lst, legend_lst, save_dir, "Gen", True)

