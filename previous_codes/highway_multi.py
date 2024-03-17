import argparse
import collections
import json

import gym
import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from gym.utils import seeding
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from highway_env.envs import HighwayEnvFast, MergeEnv
from safe_rl.utils.mpi_tools import mpi_fork
from safe_rl.utils.run_utils import setup_logger_kwargs
import safe_rl
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

import gym
import highway_env

from SafetyRulesParser import SafetyRulesParser
from ppo import PPO, ShieldPPO, RuleBasedShieldPPO, PPOCostAsReward
from gym import spaces, register
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
        return obs.reshape(-1)

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        cost = self.env._cost(0)
        info['cost'] = cost
        return self.observation(next_obs), reward, done, info


class EnvsWrapper(gym.Wrapper):
    def __init__(self, envs, has_continuous_action_space = False, seed = 46456):
        self.envs = envs # the first environment is assumed to have the full set of actions
        self.np_random = self.np_random, _ = seeding.np_random(seed)
        self.env_index = self.np_random.randint(0, len(envs))
        super().__init__(self.envs[self.env_index])
        self.state_dim =  np.prod(self.env.observation_space.spaces[0].shape) + self.env.observation_space.spaces[1].n + len(envs)
        self.actions = envs[0].action_type.actions
        self.has_continuous_action_space = has_continuous_action_space


    def observation(self, obs):
        one_hot_task = np.zeros(len(self.envs))
        one_hot_task[self.env_index] = 1
        return np.append(obs,one_hot_task)

    def reset(self):
        self.env_index = self.np_random.randint(0, len(self.envs))
        self.env = self.envs[self.env_index]
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        try:
            action = action[0]
        except:
            pass
        mapped_action = list(self.env.action_type.actions.keys())[list(self.env.action_type.actions.values()).index(self.actions[action])]
        next_obs, reward, done, info = self.env.step(mapped_action)
        cost = self.env._cost(mapped_action)
        info['cost'] = cost
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
        values = self.env.action_type.actions.values()
        valid_actions = []
        for item in self.actions.items():
            if item[1] in values:
                valid_actions.append(item[0])
        return valid_actions

    def get_current_env_name(self):
        return type(self.env).__name__



def register_envs(envs):
    suffix = '-v0'
    for env in envs:
        entry = env[:-len(suffix)]
        register(
            id=env,
            entry_point='envs:'+entry,
        )


def train(arguments = None):
    global env, rewards
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                        help="algorithm to use:  ShieldPPO | RuleBasedShieldPPO (REQUIRED)")
    parser.add_argument("--envs", default=["HighwayEnvFastNoNormalization-v0"], nargs="+",
                        help="names of the environment to train on")
    parser.add_argument("--print_freq", default=1000,
                        help="print avg reward in the interval (in num timesteps)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="log avg reward in the interval (in num timesteps)")
    parser.add_argument("--save_model_freq", type=int, default=int(5e4),
                        help="save model frequency (in num timesteps)")
    parser.add_argument("--K_epochs", type=int, default=10,
                        help="update policy for K epochs")
    parser.add_argument("--eps_clip", type=float, default=0.1,
                        help="clip parameter for PPO")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="discount factor")
    parser.add_argument("--lr_actor", type=float, default=0.0005,
                        help="learning rate for actor network")
    parser.add_argument("--lr_critic", type=float, default=0.0001,
                        help="learning rate for critic network")
    parser.add_argument("--max_ep_len", type=int, default=500,
                        help="max timesteps in one episode")
    parser.add_argument("--max_training_timesteps", type=int, default=int(1e6),
                        help="break training loop if timeteps > max_training_timesteps")
    parser.add_argument("--record_mistakes", type=bool, default=False,
                        help="record episodes with mistakes")
    parser.add_argument("--render", type=bool, default=True,
                        help="render environment")
    parser.add_argument("--record_trajectory_length", type=int, default=20,
                        help="Record trajectory length")
    parser.add_argument("--cpu", type=int, default=4,
                        help="Number of cpus")
    parser.add_argument("--agents", type=int, default=10,
                        help="Number of Agents")
    args = parser.parse_args(arguments)

    has_continuous_action_space = False
    max_ep_len = args.max_ep_len  # max timesteps in one episode
    max_training_timesteps = args.max_training_timesteps  # break training loop if timeteps > max_training_timesteps
    # print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
    # log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
    print_freq = args.print_freq  # print avg reward in the interval (in num timesteps)
    log_freq = args.log_freq  # log avg reward in the interval (in num timesteps)
    save_model_freq = args.save_model_freq  # save model frequency (in num timesteps)
    action_std = None
    action_std_decay_freq = None
    action_std_decay_rate = None
    min_action_std = None
    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    update_shield_timestep = max_ep_len
    K_epochs = args.K_epochs  # update policy for K epochs [10,50,100]
    eps_clip = args.eps_clip  # clip parameter for PPO  [0.1,0.2]
    gamma = args.gamma  # discount factor [0.9,0.95  0.99]
    lr_actor = args.lr_actor  # learning rate for actor network [1e-4, 5e-4, 1e-3]
    lr_critic = args.lr_critic  # learning rate for critic network [1e-4, 5e-4, 1e-3]
    #####################################################
    register_envs(args.envs)
    agent_type = args.algo
    np.random.seed(args.seed)

    multi_task_envs  = [[] for x in range(args.agents)]
    agents_time_steps = [[] for x in range(args.agents)]
    agents_time_step = [0]*args.agents
    agents_i_episode = [0]*args.agents
    agents_rewards = [[] for x in range(args.agents)]
    agents_tasks = [[] for x in range(args.agents)]
    agents_costs = [[] for x in range(args.agents)]
    agents_shield_losses = [[] for x in range(args.agents)]
    ppo_agents = [[] for x in range(args.agents)]
    agents_save_model_paths = [[] for x in range(args.agents)]
    agents_save_stats_paths = [[] for x in range(args.agents)]

    for agent in range(args.agents):

        env_list = [gym.make(x) for x in args.envs]
        random_seed = np.random.randint(1000,10000)
        multi_task_envs[agent] = EnvsWrapper(env_list, has_continuous_action_space, seed= random_seed)
        if args.record_mistakes:
            for env in multi_task_envs[agent].envs:
                env.metadata["video.frames_per_second"] = 30
                env.metadata["video.output_frames_per_second"] = 30
        action_dim = multi_task_envs[agent].action_space_size()

        #### create new log file for each run
        curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_path = f"./models/{agent_type}_{agent}_{random_seed}_{curr_time}"
        agents_save_model_paths[agent] = f"./{base_path}/model.pth"
        agents_save_stats_paths[agent] = f"./{base_path}/stats.log"
        save_args_path = f"./{base_path}/commandline_args.txt"
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(base_path + "/Videos", exist_ok=True)

        safe_rl_baselines = ['ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

        with open(save_args_path, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        if agent_type == "ShieldPPO":
            ppo_agents[agent] = ShieldPPO(multi_task_envs[agent].state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                                  has_continuous_action_space, action_std)

        elif agent_type == "RuleBasedShieldPPO":
            rule_parser = SafetyRulesParser("../rules.txt", multi_task_envs[agent].env.action_type.actions)
            ppo_agents[agent] = RuleBasedShieldPPO(multi_task_envs[agent].state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                                           has_continuous_action_space, rule_parser, action_std)
        else:
            raise NotImplementedError


    start_time = time.time()
    # training loop

    to_continue = True
    while (to_continue):
        to_continue = False
        for agent in range(args.agents):
            if agents_time_step[agent] <= max_training_timesteps:
                to_continue = True
                state = multi_task_envs[agent].reset()
                task_name = multi_task_envs[agent].get_current_env_name()
                valid_actions = multi_task_envs[agent].get_valid_actions()
                trajectory = collections.deque([state])
                recorder_closed = False
                current_ep_reward = 0
                current_ep_cost = 0
                is_mistake = False
                if args.record_mistakes:
                    base_video_path = f"./{base_path}/Videos"
                    os.makedirs(base_path, exist_ok=True)
                    video_path = f"{base_video_path}/episode_" + str(agents_i_episode[agent])
                    trajectory_path = f"{base_video_path}/episode_" + str(agents_i_episode[agent]) +"_trajectory.txt"
                    video_recorder = VideoRecorder(multi_task_envs[agent].env, base_path=video_path, enabled=video_path is not None)
                for t in range(1, max_ep_len + 1):

                    # select action with policy
                    if args.render:
                        multi_task_envs[agent].env.render()

                    action = ppo_agents[agent].select_action(state, valid_actions)
                    prev_state = state
                    state, reward, done, info = multi_task_envs[agent].step(action)
                    trajectory.appendleft((action,state))
                    if len(trajectory) > args.record_trajectory_length:
                        trajectory.pop()
                    if args.record_mistakes:
                        video_recorder.capture_frame()
                    # saving reward and is_terminals
                    ppo_agents[agent].buffer.rewards.append(reward)
                    ppo_agents[agent].buffer.costs.append(info['cost'])
                    ppo_agents[agent].buffer.is_terminals.append(done)

                    agents_time_step[agent] += 1
                    current_ep_reward += reward
                    current_ep_cost += info['cost']

                    # update PPO agent
                    if agents_time_step[agent] % update_timestep == 0:
                        ppo_agents[agent].update()

                    if agents_time_step[agent] % update_shield_timestep == 0 and agent == "ShieldPPO":
                        shield_loss = ppo_agents[agent].update_shield(1024)
                        agents_shield_losses[agent].append(shield_loss)

                    if info["cost"] > 0:
                        is_mistake = True

                    if agent == "ShieldPPO" or agent == "RuleBasedShieldPPO":
                        if info["cost"] > 0:
                            if args.record_mistakes:
                                video_recorder.close()
                                recorder_closed = True
                                with open(trajectory_path, 'w') as f:
                                    for item in reversed(trajectory):
                                        f.write(f"{item}\n")
                            for agent_shield in range(args.agents):
                                ppo_agents[agent_shield].add_to_shield(prev_state, action, 0)
                        else:
                            for agent_shield in range(args.agents):
                                ppo_agents[agent_shield].add_to_shield(prev_state, action, 1)

                    # if continuous action space; then decay action std of ouput action distribution
                    if has_continuous_action_space and agents_time_step[agent] % action_std_decay_freq == 0:
                        ppo_agents[agent].decay_action_std(action_std_decay_rate, min_action_std)

                    # log in logging file
                    if agents_time_step[agent] % log_freq == 0:
                        torch.save((agents_time_steps[agent], agents_rewards[agent], agents_costs[agent], agents_tasks[agent]), agents_save_stats_paths[agent])

                    # printing average reward
                    if agents_time_step[agent] % print_freq == 0:
                        recent_reward = np.array(agents_rewards[agent][max(0, len(agents_rewards[agent]) - 10):]).mean()
                        recent_cost = np.array(agents_costs[agent][max(0, len(agents_costs[agent]) - 10):]).mean()
                        if agent == "ShieldPPO":
                            recent_shield_loss = np.array(agents_shield_losses[agent][max(0, len(agents_shield_losses[agent]) - 10):]).mean()
                            print(
                                f"Episode : {agents_i_episode[agent]:4d} Reward {recent_reward:6.2f} Cost {recent_cost:6.2f} Shield Loss {recent_shield_loss:6.2f} Time {(time.time() - start_time) / 60:.2f} min")
                        else:
                            print(
                                f"Episode : {agents_i_episode[agent]:4d} Reward {recent_reward:6.2f} Cost {recent_cost:6.2f} Time {(time.time() - start_time) / 60:.2f} min")

                    # save model weights
                    if agents_time_step[agent] % save_model_freq == 0:
                        ppo_agents[agent].save(agents_save_model_paths[agent])

                    if done:
                        break

                agents_rewards.append(current_ep_reward)
                agents_costs[agent].append(current_ep_cost)
                agents_tasks[agent].append(task_name)
                agents_time_steps[agent].append(agents_time_step[agent])
                if args.record_mistakes and not recorder_closed:
                    video_recorder.close()
                    if not is_mistake:
                        os.remove(video_recorder.path)
                        os.remove(video_recorder.metadata_path)
                    else:
                        with open(trajectory_path, 'w') as f:
                            for item in reversed(trajectory):
                                f.write("%s\n" % item)
                agents_i_episode[agent] += 1
            else:
                multi_task_envs[agent].close()


if __name__ == '__main__':
    train()
