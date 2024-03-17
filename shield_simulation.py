import collections
import gym
import os
import time
from gym.utils import seeding
# import tensorflow as tf
# import safe_rl
import numpy as np
# from SafetyRulesParser import SafetyRulesParser
from previous_codes.ppo_shieldLSTM import ShieldPPO
from gym import spaces, register

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
        return obs.reshape(-1)

    def reset(self):
        # returns the first observation
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        # returns the next observation from the current state + given action, and reward& info
        next_obs, reward, done, info = self.env.step(action)
        cost = self.env._cost(0)
        info['cost'] = cost
        return self.observation(next_obs), reward, done, info


class EnvsWrapper(gym.Wrapper):
    def __init__(self, envs, has_continuous_action_space=False, seed=46456):
        self.envs = envs  # the first environment is assumed to have the full set of actions
        self.np_random = self.np_random, _ = seeding.np_random(seed)
        self.env_index = self.np_random.randint(0, len(envs))
        super().__init__(self.envs[self.env_index])
        self.state_dim = np.prod(self.env.observation_space.shape) + len(envs)  # low = self.env.observation_space.low
        # high = self.env.observation_space.high
        # dim_state = np.prod(self.env.observation_space.shape)
        # self.observation_space = spaces.Box(low=low.reshape(-1),
        #                                     high=high.reshape(-1),
        #                                     shape=(dim_state,),
        #                                     dtype=np.float32)
        # dim_state =
        self.actions = envs[0].action_type.actions
        self.has_continuous_action_space = has_continuous_action_space

    def observation(self, obs):
        one_hot_task = np.zeros(len(self.envs))
        one_hot_task[self.env_index] = 1
        return np.append(obs, one_hot_task)

    def reset(self):
        self.env_index = self.np_random.randint(0, len(self.envs))
        self.env = self.envs[self.env_index]
        # low = self.env.observation_space.low
        # high = self.env.observation_space.high
        # dim_state = np.prod(self.env.observation_space.shape)
        # self.observation_space = spaces.Box(low=low.reshape(-1),
        #                                     high=high.reshape(-1),
        #                                     shape=(dim_state,),
        #                                     dtype=np.float32)
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        try:
            action = action[0]
        except:
            pass
        mapped_action = list(self.env.action_type.actions.keys())[
            list(self.env.action_type.actions.values()).index(self.actions[action])]
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
    # create end points for the environments
    suffix = '-v0'
    for env in envs:
        entry = env[:-len(suffix)]
        register(
            id=env,
            entry_point='envs:' + entry,
        )


def simulate(ac_path, shield_path , masking_threshold = 0, k_last_states = 1, max_simulation_timesteps = 1e6, max_ep_len = 25, print_freq = 20):

    """
    time step = interaction with the environment (outer loop) = one action
    episodes = sequence of interactions
    each iteration in the inner loop is a new episodes (sequence of actions) and the number of actions is limited due to max_timesteps
    trajectory - sequence of (state,action) pairs that the agent encounters during its interaction with the environment within a single episode.
    """
    global env, rewards

    #        ppo_agent = ShieldPPO(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    # has_continuous_action_space, action_std)

    envs = ["HighwayEnvFastNoNormalization-v0"]
    register_envs(envs)
    env_list = [gym.make(x) for x in envs]
    random_seed = 1
    multi_task_env = EnvsWrapper(env_list, has_continuous_action_space = False)
    action_dim = multi_task_env.action_space_size()

    ppo_agent = ShieldPPO(multi_task_env.state_dim, action_dim, lr_actor = 0.0005, lr_critic = 0.0001, gamma = 0.9, K_epochs = 10 , eps_clip = 0.1,
                          has_continuous_action_space = False , action_std_init = None , masking_threshold= masking_threshold, k_last_states = k_last_states)
    ppo_agent.load(ac_path, shield_path)
    time_step = 0
    i_episode = 0
    time_steps = []
    rewards = []
    tasks = []
    costs = []
    shield_losses = []
    stats = []
    start_time = time.time()
    # training loop
    while time_step <= max_simulation_timesteps:
        # NEW EPOCH / EPISODE (defined by i_episode) - EACH EPISODE STARTS WITH A NEW STATE
        #     print("starting time stamp", time_step)
        state = multi_task_env.reset()
        last_states = [state]
        task_name = multi_task_env.get_current_env_name()
        valid_actions = multi_task_env.get_valid_actions()
        trajectory = collections.deque([state])
        # INITALIZE THE REWARDS & COSTS (CUMULATIVE)
        current_ep_reward = 0
        current_ep_cost = 0
        for t in range(1, max_ep_len + 1):
            multi_task_env.env.render()
            # RELEVANT
            prev_states = last_states.copy()
            action, safety_scores, no_safe_action = ppo_agent.select_action(last_states, valid_actions, time_step)
            #  print("the selected action is", action)
            state, reward, done, info = multi_task_env.step(action)
            last_states.append(state)
            if len(last_states) > k_last_states:
                last_states = last_states[1:]
            trajectory.appendleft((action, state))
            ##  adding only one reward to the buffer
        #    ppo_agent.buffer.rewards.append(reward)
        #    ppo_agent.buffer.costs.append(info['cost'])
        #    ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward
            #    print(" += info['cost'] is",  info['cost'])
            current_ep_cost += info['cost']
            if info["cost"] > 0:
                print("COLLISION !!!" + str(no_safe_action))
            # printing average reward
            if time_step % print_freq == 0:
                recent_reward = np.array(rewards[max(0, len(rewards) - 10):]).mean()
                recent_cost = np.array(costs[max(0, len(costs) - 10):]).mean()
                recent_shield_loss = np.array(shield_losses[max(0, len(shield_losses) - 10):]).mean()
                print(f"Episode : {i_episode:4d} Reward {recent_reward:6.2f} Cost {recent_cost:6.2f} Shield Loss {recent_shield_loss:6.2f} Time {(time.time() - start_time) / 60:.2f} min")

            # THE EPOCH FINISHES IF DONE == TRUE (REACHED TO FINAL STATE) OR REACHED TO MAX_EP_LEN
            if done:
                print("REACHED TO FINISH STATE BEFORE MAX_STEPS")
                break
        # IN THE END OF EACH EPOCH
        rewards.append(current_ep_reward)
        costs.append(current_ep_cost)
        tasks.append(task_name)
        time_steps.append(time_step)

        i_episode += 1
    end_time = time.time()
    total_training_time = end_time - start_time
    print("ENDED: REWARDS (PER EPOCH):", rewards)
    print("ENDED: COSTS (PER EPOCH):", costs)
    multi_task_env.close()


if __name__ == '__main__':
    simulate(ac_path ="models/22_11_test_problem/model.pth", shield_path ="models/22_11_test_problem/shield.pth", max_simulation_timesteps = 1000)


