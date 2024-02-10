import matplotlib

matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import torch
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

def smooth(x, window_size=50):
    x = np.array(x)
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i - window_size):min(n, i + window_size)].mean()
    return b


def create_loss_stats_df(data_path):
    log_stats = torch.load(data_path, map_location=torch.device('cpu'))
    log_stats_df = pd.DataFrame([(key, *value) for key, value in log_stats.items()],
                                columns=["Time Step", "Episode", "Step in Episode", "Shield Loss", "Len of Pos Samples",
                                         "Avg Prediction for Pos Samples", "Len of Neg Samples",
                                         "Avg Prediction for Neg Samples"])
    return log_stats_df


def plot_predictions(df, lines_str, plot_name):  # Create separate subplots for each line
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    # Plot 'Len of Pos Samples'
    first_line, sec_line = lines_str
    axes[0].plot(df['Time Step'], df[first_line], label=first_line, color='blue')
    axes[0].set_ylabel(first_line)
    axes[0].legend()

    # Plot 'Avg Prediction for Pos Samples'
    axes[1].plot(df['Time Step'], df[sec_line], label=sec_line, color='orange')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('sec_line')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.clf()


def plot_shield_loss(df, save_path):
    # save_path - path to be saved
    plt.plot(df['Time Step'], smooth(df['Shield Loss']), label='Shield Loss', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Shield Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.clf()


def plot_rewards(df, save_path):
    # save_path - path to be saved
    plt.plot(df[0], smooth(df[1]))
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.savefig(save_path + "rewards_over_time.png")
    plt.clf()


def plot_collisions(df, save_path):
    # save_path - path to be saved
    plt.plot(df[0], smooth(df[2]))
    plt.xlabel('Time Step')
    plt.ylabel('Collisions')
    plt.savefig(save_path + "collisions_over_time.png")
    plt.clf()


# Apply functions

# TODO - Change these lines while running
shield_loss_stat_paths_lst = [
    "models/20.12 carpole (no gan)/maxEpLen=500_20231220-142933/shield_loss_stats"]  # add more
save_path = [
    "models/20.12 carpole (no gan)/maxEpLen=500_20231220-142933/plot/"]  # add the place to save, for example models/long run/masking=0/ add more

new_folder_path = "models/20.12 carpole (no gan)/maxEpLen=500_20231220-142933/plot"

os.makedirs(new_folder_path, exist_ok=True)

for i, path in enumerate(shield_loss_stat_paths_lst):
    shield_loss_stats_df = create_loss_stats_df(path + ".log")
    shield_loss_stats_df.to_csv(path + ".csv")
    plot_predictions(shield_loss_stats_df, ['Len of Pos Samples', 'Avg Prediction for Pos Samples'],
                     save_path[i] + "pos_predictions.png")
    plot_predictions(shield_loss_stats_df, ['Len of Neg Samples', 'Avg Prediction for Neg Samples'],
                     save_path[i] + "neg_predictions.png")
    plot_shield_loss(shield_loss_stats_df, save_path[i] + "loss_over_time.png")

# TODO - Change these lines while running
agent_stats_paths_lst = [
    "models/20.12 carpole (no gan)/maxEpLen=500_20231220-142933/stats.log"]  # add more, just an example

for i, path in enumerate(agent_stats_paths_lst):
    # remove the map_location while running on GPU cluster
    agent_stats = torch.load(path, map_location=torch.device('cpu'))
    plot_rewards(agent_stats, save_path[i])
    plot_collisions(agent_stats, save_path[i])

"""


#plot_predictions("",['Len of Pos Samples','Avg Prediction for Pos Samples'], "models/GAN/basic_initial_arch_100kmasking/pos_samples.png")
#plot_predictions("",['Len of Neg Samples','Avg Prediction for Neg Samples'], "models/GAN/basic_initial_arch_100kmasking/neg_samples.png")


# Plot 'Shield Loss'


#stats_no_shield = torch.load(r"models/14.12/wilds['HighwayEnvSimple-v0']_timesteps=100_SimplerEnv_checkall__20231214-154550/stats.log", map_location=torch.device('cpu'))

stats = torch.load("models/14.12/shira_20231214-155102/stats.log", map_location=torch.device('cpu'))


    # check_fupdate - masking = 0
shield_loss_stats_df = create_loss_stats_df("models/long run/masking=0/shield_loss_stats.log")
shield_loss_stats_df.to_csv("models/long run/masking=0/shield_loss_stats.csv")
plot_predictions(shield_loss_stats_df,['Len of Pos Samples','Avg Prediction for Pos Samples'], "models/long run/masking=0/pos_samples.png")
plot_predictions(shield_loss_stats_df,['Len of Neg Samples','Avg Prediction for Neg Samples'], "models/long run/masking=0/neg_samples.png")
plot_shield_loss(shield_loss_stats_df, "models/long run/masking=0/loss_over_time.png")




# check_update - masking = 100K

shield_loss_stats_df = create_loss_stats_df("models/5.12/check_fupdate_masking=100K/shield_loss_stats.log")
shield_loss_stats_df.to_csv("models/5.12/check_fupdate_masking=100K/shield_loss_stats.csv")
plot_predictions(shield_loss_stats_df,['Len of Pos Samples','Avg Prediction for Pos Samples'], "models/5.12/check_fupdate_masking=100K/pos_samples.png")
plot_predictions(shield_loss_stats_df,['Len of Neg Samples','Avg Prediction for Neg Samples'], "models/5.12/check_fupdate_masking=100K/neg_samples.png")
plot_shield_loss(shield_loss_stats_df, "models/5.12/check_fupdate_masking=100K/loss_over_time.png")


# check_no shield

shield_loss_stats_df = create_loss_stats_df("models/5.12/check_no_shield/shield_loss_stats.log")
shield_loss_stats_df.to_csv("models/5.12/check_no_shield/shield_loss_stats.csv")
plot_predictions(shield_loss_stats_df,['Len of Pos Samples','Avg Prediction for Pos Samples'], "models/5.12/check_no_shield/pos_samples.png")
plot_predictions(shield_loss_stats_df,['Len of Neg Samples','Avg Prediction for Neg Samples'], "models/5.12/check_no_shield/neg_samples.png")
plot_shield_loss(shield_loss_stats_df, "models/5.12/check_no_shield/loss_over_time.png")

# Plot 'Shield Loss'





plot_rewards(stats)


"""


