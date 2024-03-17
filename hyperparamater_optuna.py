# import
import optuna
import pandas as pd
import argparse
import collections
import json
import gym
import os
import glob
import numpy as np
import time
from datetime import datetime
import torch
import sys
import torch.nn as nn
from gym.utils import seeding
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym import spaces, register

# internal
from highway_env.envs import HighwayEnvFast, MergeEnv
from envs.NoNormalizationEnvs import CartPoleWithCost
from torch.distributions import MultivariateNormal, Categorical
from updated_ppo_GAN import PPO, ShieldPPO, RuleBasedShieldPPO, PPOCostAsReward, Generator, GeneratorBuffer

# set device to cpu or cuda
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        dim_state = np.prod(self.env.observation_space.shape)
        self.observation_space = spaces.Box(low=low.reshape(-1),
                                            high=high.reshape(-1),
                                            shape=(dim_state,),
                                            dtype=np.float32)

    def observation(self, obs):
        # the function returns an observation
        return obs.reshape(-1)

    def reset(self):
        # returns the first observation of the env
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        # returns the next observation from the current state by taking input action, and additional info
        next_obs, reward, done, placeholder1, info = self.env.step(action)
        cost = self.env._cost(next_obs)
        info['cost'] = cost
        #  observation, reward, terminated (True/False if arrived to terminated state), False. {} -> the last 2 are placeholders for general gym env purposes
        return self.observation(next_obs), reward, done, info


class EnvsWrapper(gym.Wrapper):
    def __init__(self, envs, has_continuous_action_space=False, seed=46456, no_render=False):
        self.envs = envs  # the first environment is assumed to have the full set of actions
        self.np_random = self.np_random, _ = seeding.np_random(seed)
        self.env_index = self.np_random.randint(0, len(envs))
        super().__init__(self.envs[self.env_index])
        self.state_dim = np.prod(self.env.observation_space.shape) + len(envs)  # low = self.env.observation_space.low
        self.no_render = no_render
        if self.no_render:
            self.env.render_mode = 'no_render'
        """
         high = self.env.observation_space.high
         dim_state = np.prod(self.env.observation_space.shape)
         self.observation_space = spaces.Box(low=low.reshape(-1),
                                             high=high.reshape(-1),
                                             shape=(dim_state,),
                                             dtype=np.float32)
         dim_state =no_render
        """
        # TODO- it's manually for cartpole because there's no mapping field like this. if we work with more than 1 env it needs to be fixed.
        self.actions = {0: "Move Left", 1: "Move Right"}
        self.has_continuous_action_space = has_continuous_action_space

    def observation(self, obs):
        one_hot_task = np.zeros(len(self.envs))
        one_hot_task[self.env_index] = 1
        return np.append(obs, one_hot_task)

    def reset(self, gan_output=None):
        self.env_index = self.np_random.randint(0, len(self.envs))
        self.env = self.envs[self.env_index]
        """
         low = self.env.observation_space.low
         high = self.env.observation_space.high
         dim_state = np.prod(self.env.observation_space.shape)
         self.observation_space = spaces.Box(low=low.reshape(-1),
                                             high=high.reshape(-1),
                                             shape=(dim_state,),
                                             dtype=np.float32)
        """
        obs = self.env.reset(gan_output)
        # return obs also because its shape is (number_of_vehicles, number_of_features) -> good for the log
        return self.observation(obs), obs

    def step(self, action):
        try:
            action = action[0]
        except:
            pass
        # TODO - why do we need this mapping? removed it for now.
        """
        mapped_action = list(self.env.action_type.actions.keys())[
            list(self.env.action_type.actions.values()).index(self.actions[action])]
        """
        next_obs, reward, done, info = self.env.step(action)
        cost = self.env._cost(next_obs)
        info['cost'] = cost
        # ext_obs, reward, done, placeholder1, info  = self.env.step(action)
        return self.observation(next_obs), reward, done, info

    def action_space_size(self):
        if self.has_continuous_action_space:
            action_dim = self.envs[0].action_space.shape[0]
        else:
            action_dim = self.envs[0].action_space.n
        return action_dim

    def active_action_space_size(self):
        if self.has_continuous_action_space:
            action_dim = self.env.action_space.shape[0]
        else:
            action_dim = self.env.action_space.n
        return action_dim

    def get_valid_actions(self):
        return list(range(self.action_space.n))  # returns a vector of values 0/1 which indicate which actions are valid

    def get_current_env_name(self):
        return type(self.env).__name__


def register_envs(envs):
    # create end points for the environments
    suffix = '-v0'
    for env in envs:
        entry = env[:-len(suffix)]
        register(
            id=env,
            entry_point='envs:' + entry,
        )


envs = ["CartPoleWithCost-v0"]
register_envs(envs)
env_list = [gym.make(x) for x in envs]
multi_task_env = EnvsWrapper(env_list, has_continuous_action_space=False, no_render=True)

parser = argparse.ArgumentParser(description='Arguments for hyperparameters, using optuna module')

# Add arguments
parser.add_argument('--save_trials', type=int, default=1, help='Save CSV every x trials')
parser.add_argument('--save_path', type=str,
                    default="optuna_trials/maximize-r/trials_stats.csv",
                    help='Path to save CSV file')
parser.add_argument('--n_trials', type=int, default=100, help='Number of trials')
parser.add_argument('--direction', type=str, default='maximize', choices=['minimize', 'maximize'],
                    help='Direction for optimization')
parser.add_argument('--set_objective', type=str, default='r', choices=['r', 'c', 'sl', 'gl'], help='Objective function')
parser.add_argument("--generator_latent_dim", type=float, default=32,
                    help="The dimension of latent space (Generator)")
parser.add_argument('--no_gan', type=bool, default=False, help='if no_gan = True - not using generator')
parser.add_argument('--max_training_timesteps_per_trial', type=int, default=50000,
                    help='max_training_timesteps_per_trial')

args = parser.parse_args()

SAVE_TRIALS = args.save_trials

SAVE_STATS_PATH = args.save_path

N_TRIALS = args.n_trials

DIRECTION = args.direction

generator_latent_dim = args.generator_latent_dim

SET_OBJECTIVE = args.set_objective

no_gan = args.no_gan

max_training_timesteps = args.max_training_timesteps_per_trial


def save_trials_stats_as_csv(trial_number):
    trials_data_df = pd.DataFrame(trials_data)
    params_df = pd.DataFrame(trials_data_df['Params Values'].tolist())
    final_trials_df = pd.concat(
        [trials_data_df[['Trial Number', 'Rewards', 'Costs', 'Shield Loss', 'Gen Loss']], params_df], axis=1)
    final_trials_df.to_csv(SAVE_STATS_PATH, index=False)
    print(f"trials_stats.csv was last updated for trial number {trial_number}")


"---------------------------------------------------------------------------------------------------------------------------------------------------------------"

shield_losses_per_trial = []
gen_losses_per_trial = []


def objective(trial, set_objective):
    """
    trial - the trial number
    set_objective - 'r' for average rewards per trial (maximize) , 'c' for average costs per trial (minimize), 'sl' shield loss per trial, 'gl' - for average gl per trial

    """
    print(f"Starting a trial number {trial.number}")
    # Hyperparameters to optimize
    render = False
    # ppo parameters
    k_epochs_ppo = trial.suggest_int('k_epochs_ppo', 10, 30)
    # update ppo after (_) "max ep len"
    update_ppo = trial.suggest_int('k_epochs_ppo', 0, 10)
    # shield parameters
    k_epochs_shield = trial.suggest_int('k_epochs_shield', 10, 30)
    update_shield_timestep = trial.suggest_categorical('update_shield_timestep', [10000, 25000, 50000, 100000])
    shield_gamma = trial.suggest_float('shield_gamma', 0.1, 0.99)
    lr_shield = trial.suggest_loguniform('lr_shield', 3e-5, 7e-5)
    unsafe_thresh = trial.suggest_float('unsafe_thresh', 0.1, 0.9)
    shield_episodes_bath_size = trial.suggest_int('shield_episodes_batch_size', 1, 6)
    masking_threshold = trial.suggest_int('masking_threshold', 1, 500000)
    # generator paramaters
    # latent_dim = trial.suggest_categorical('latent_dim', [16.0, 32.0, 64.0, 128.0])
    gen_episodes_bath_size = trial.suggest_int('shield_episodes_batch_size', 1, 10)

    update_gen_timestep = trial.suggest_int('update_gen_timestep', 0, 100000)
    # gen masking tresh is according to epochs
    gen_masking_tresh = trial.suggest_int('gen_masking_tresh', 0, 50)

    if no_gan:
        gen_masking_tresh = 1000000000000000000000
        update_gen_timestep = 1000000000000000000000

    action_dim = multi_task_env.action_space_size()
    # optimal params according to ray ppo
    lr_actor = 5e-5
    lr_critic = 5e-5
    gamma = 0.99
    eps_clip = 0.3
    lr_gen = 5e-5
    action_std = None
    k_epochs_gen = 30
    param_ranges = {
        'gravity': {'range': (9.0, 10.0), 'type': float},
        'masscart': {'range': (0.5, 2), 'type': float},
        'masspole': {'range': (0.05, 0.5), 'type': float},
        'length': {'range': (0.4, 0.6), 'type': float}
    }
    # define agent
    ppo_agent = ShieldPPO(state_dim=multi_task_env.state_dim, action_dim=action_dim, latent_dim=generator_latent_dim,
                          lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, eps_clip=eps_clip,
                          k_epochs_ppo=k_epochs_ppo, k_epochs_shield=k_epochs_shield, k_epochs_gen=k_epochs_gen,
                          has_continuous_action_space=False, lr_shield=lr_shield, lr_gen=lr_gen,
                          shield_gamma=shield_gamma, action_std_init=action_std, masking_threshold=masking_threshold,
                          unsafe_tresh=unsafe_thresh, param_ranges=param_ranges)

    masking_threshold = trial.suggest_int('masking_threshold', 1, 500000)
    time_step = 0
    max_ep_len = 200

    update_ppo_timestep = max_ep_len * update_ppo
    average_update_loss = []
    i_episode = 0
    # each entry is the sum of rewards in the episode
    all_episodes_rewards_sum = []
    # each entry is the sum of costs in the episode
    all_episodes_costs_sum = []
    # each entry is the average loss per episode (over all updates)
    all_episodes_shield_loss = []
    # same for gen
    all_episodes_gen_loss = []
    while time_step <= max_training_timesteps:
        print(f"time_step is {time_step}")
        if i_episode >= gen_masking_tresh:
            steps_before_collision = 0
            param_dict, unsafe_scores = ppo_agent.get_generated_env_config()
            state, state_vf = multi_task_env.reset(param_dict)
            gen_chosen_state = state
            gen_chosen_action = unsafe_scores.index(max(unsafe_scores))
        else:
            state, state_vf = multi_task_env.reset()
        valid_actions = multi_task_env.get_valid_actions()
        shield_epoch_trajectory = []
        sum_rewards_ep = 0
        sum_costs_ep = 0
        # the amount of entries = the amount of updates in the current episodes. each entry is a loss from shield.update()
        current_episode_shield_loss = []
        current_episode_gen_loss = []
        for t in range(1, max_ep_len):
            if i_episode >= gen_masking_tresh:
                # An episode generated by GAN (in case of i_episode >= gen_masking_tresh)
                steps_before_collision += 1
            if render:
                multi_task_env.env.render()
            prev_state = state.copy()
            action, unsafe_scores = ppo_agent.select_action(state, valid_actions, time_step)
            if (i_episode >= gen_masking_tresh) and (t == 1):
                # For the first time, the agent will choose the action chosen by generator in any case.
                action = gen_chosen_action
            state, reward, done, info = multi_task_env.step(action)
            # it's like adding 1 - because every reward is 1 (for each time step)
            sum_rewards_ep += reward
            cost = info['cost']
            sum_costs_ep += cost
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.costs.append(cost)
            ppo_agent.buffer.is_terminals.append(done)
            shield_epoch_trajectory.append((torch.tensor(prev_state), torch.tensor([action]), cost, done))
            time_step += 1
            if time_step % update_ppo_timestep == 0:
                ppo_agent.update()
            if time_step % update_shield_timestep and type(ppo_agent) == ShieldPPO:
                shield_update_loss = ppo_agent.update_shield(shield_episodes_bath_size)
                current_episode_shield_loss.append(shield_update_loss)
            if time_step % update_gen_timestep == 0 and ppo_agent == "ShieldPPO" and i_episode >= gen_masking_tresh:
                gen_update_loss = ppo_agent.update_gen(gen_episodes_bath_size)
                current_episode_gen_loss.append(gen_update_loss)
            if done:
                break;

        # add the average shield loss for this episode to episodes_shield_loss
        all_episodes_shield_loss.append(sum(current_episode_shield_loss) / len(current_episode_shield_loss))

        if i_episode >= masking_threshold:
            all_episodes_gen_loss.append(sum(current_episode_gen_loss) / len(current_episode_gen_loss))
        if i_episode >= gen_masking_tresh:
            # In case of using generator - saving gen_chosen_state in list (same structure as k_last_states, as it is sent to the shield in the Gen.loss())
            ppo_agent.add_to_gen(gen_chosen_state, gen_chosen_action, steps_before_collision)
        ppo_agent.add_to_shield(shield_epoch_trajectory)
        all_episodes_rewards_sum.append(sum_rewards_ep)
        all_episodes_costs_sum.append(sum_costs_ep)
        i_episode += 1

    # initaize shield loss and gen loss to Null to deal with the problem of maskng
    shield_loss, gen_loss = None, None
    # Calculate stats for current trial
    shield_loss = sum(all_episodes_shield_loss) / len(all_episodes_shield_loss)
    shield_losses_per_trial.append(shield_loss)

    if i_episode >= gen_masking_tresh and len(all_episodes_gen_loss):
        gen_loss = sum(all_episodes_gen_loss) / len(all_episodes_gen_loss)
        gen_losses_per_trial.append(gen_loss)

    average_reward_for_trial = sum(all_episodes_rewards_sum) / len(all_episodes_rewards_sum)
    print(f"Trial #{trial.number} average episodes rewards is {average_reward_for_trial} ")
    average_cost_for_trial = sum(all_episodes_costs_sum) / len(all_episodes_costs_sum)

    # store trial information in a dict
    trial_data = {
        'Trial Number': trial.number,
        'Params Values': trial.params,
        'Rewards': average_reward_for_trial,
        'Costs': average_cost_for_trial,
        'Shield Loss': shield_loss if shield_loss is not None else "no shield yet",
        'Gen Loss': gen_loss if gen_loss is not None else "no gen yet"
    }
    trials_data.append(trial_data)
    # save csv every save_trials trials
    if (trial.number % SAVE_TRIALS) == 0:
        save_trials_stats_as_csv(trial.number)
    if set_objective == 'r':
        return average_reward_for_trial
    else:  # set_objective == 'c'
        return average_cost_for_trial


trials_data = []

study = optuna.create_study(direction=DIRECTION)
study.optimize(lambda trial: objective(trial, SET_OBJECTIVE), n_trials=N_TRIALS)
