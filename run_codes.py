import pandas as pd

# PPO - SHIELD LSTM - 12.09 (WITHOUT ACTOR CRITIC)

import numpy as np
import random
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


#    torch.save((time_steps, rewards, costs, tasks, total_training_time, episodes_len), save_stats_path)




collision_info_log = torch.load("models/collision_info_log.log")

collision_info_log.to_csv("will be deleted.csv", index=False)  # Set index=False to exclude row numbers
