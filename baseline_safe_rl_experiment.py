#!/usr/bin/env python
import gym
import numpy
import numpy as np
import safety_gym
import safe_rl
#import tensorflow as tf

from gym import spaces
from gym.envs.registration import register as register
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
import sys
sys.path.insert(0, '..')
from envs import *


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        o = self.env.reset()
        dim_state = np.prod(o['image'].shape) + 4 + 26 + 81 + 1393 # for direction, events, and goal positions
        self.direction_embed = np.eye(4)
        self.observation_space = spaces.Box(low=0.,
                                            high=255.,
                                            shape=(dim_state,),
                                            dtype=np.uint8)
        self.action_space = self.env.action_space

    def observation(self, obs):
        im = obs['image']
        direction = obs['direction']
        goal_position = obs['goal_position']
        event = obs['event']
        goal_multi_hot = numpy.zeros(26)
        agent_position = obs['agent_position']
        agent_position_one_hot = numpy.zeros(81)
        agent_position_one_hot[agent_position[0] * 9 + agent_position[1]] = 1
        for d in enumerate(goal_position):
            goal_multi_hot[d[0] + 6] = 1
            goal_multi_hot[d[1] + 19] = 1
        direction_embed = self.direction_embed[direction]
        im = im.reshape(-1)
        x = np.concatenate([im, direction_embed, goal_multi_hot, agent_position_one_hot, event], -1)
        return x

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs = self.observation(next_obs)
        if reward < -99:
            cost = 1.
        else:
            cost = 0.
        return next_obs, reward, done, {'cost': cost,
                                        'lava_steps': self.lava_steps}


def make_env(env_key, seed=None):
    env = Wrapper(RandomLavaEnvReward_1000())
    env.seed(seed)
    return env


def main(algo, seed, cpu):

    # Verify experiment
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    assert algo in algo_list, "Invalid algo"

    # Hyperparameters
    exp_name = algo

    steps_per_epoch = 10000
    num_steps = 10000000 # subject to change

    epochs = int(num_steps / steps_per_epoch)
    save_freq = 10
    target_kl = 0.01
    cost_lim = 1e-4

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir="./data")

    # Algo and Env
    algo_name = algo
    algo = eval('safe_rl.'+algo)


    algo(env_fn=lambda: make_env("RandomLavaEnv", seed),
         ac_kwargs=dict(
             hidden_sizes=(64, 64),
             activation=tf.nn.relu,
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs,
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--cl', type=float, default=0.5)
    args = parser.parse_args()
    main(args.algo, args.seed, args.cpu)
    #env = make_env("RandomLavaEnv", 0)
    #o = env.reset()
    #no, r, d, i = env.step(env.action_space.sample())
    #import pdb; pdb.set_trace()
