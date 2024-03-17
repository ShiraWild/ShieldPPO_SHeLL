# imports
# extenral
import argparse
import collections
import json
import gym
import os
import numpy as np
import time
from datetime import datetime
import torch
from gym.utils import seeding
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym import spaces, register

# internal
from updated_ppo_GAN import ShieldPPO

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


def train(arguments=None):
    # max_training_timestamps - the maximal number of interactions of the agent with the environment
    # max_ep_len - maximum number of steps in each episode
    """
    time step = interaction with the environment (outer loop) = one action
    episodes = sequence of interactions
    each iteration in the inner loop is a new episodes (sequence of actions) and the number of actions is limited due to max_timesteps
    trajectory - sequence of (state,action) pairs that the agent encounters during its interaction with the environment within a single episode.
    """
    global env, rewards
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                        help="algorithm to use:  PPO | ShieldPPO | RuleBasedShieldPPO (REQUIRED)")
    parser.add_argument("--envs", default=["CartPoleWithCost-v0"], nargs="+",
                        help="names of the environment to train on")
    parser.add_argument("--print_freq", default=1000,
                        help="print avg reward in the interval (in num timesteps)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log_freq", type=int, default=400,
                        help="log avg reward in the interval (in num timesteps)")
    parser.add_argument("--save_model_freq", type=int, default=int(5e4),
                        help="save model frequency (in num timesteps)")
    parser.add_argument("--k_epochs_ppo", type=int, default=30,
                        help="update policy for K epochs")
    parser.add_argument("--k_epochs_shield", type=int, default=30,
                        help="update Shield for K epochs")
    parser.add_argument("--k_epochs_gen", type=int, default=30,
                        help="update Gen for K epochs")
    parser.add_argument("--eps_clip", type=float, default=0.3,
                        help="clip parameter for PPO")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--shield_gamma", type=float, default=0.6,
                        help="discount factor for shield, while calculating loss function")
    parser.add_argument("--lr_actor", type=float, default=5e-5,
                        help="learning rate for actor network")
    parser.add_argument("--lr_critic", type=float, default=5e-5,
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
    # Shira - New Arguments
    parser.add_argument("--masking_threshold", type=int, default=0, help="Time step at which to start using the shield")
    parser.add_argument("--lr_shield", type=float, default=5e-5, help="Shield learning rate")
    parser.add_argument("--lr_gen", type=float, default=5e-5, help="Generator learning rate")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering during simulation")
    parser.add_argument("--unsafe_tresh", type=float, default=0.5, help="Unsafe treshold for the Shield network")
    parser.add_argument("--gen_masking_tresh", type=float, default=0,
                        help="Episode Number at which to start using the Generator, for GAN")
    parser.add_argument("--update_gen_timestep", type=float, default=500,
                        help="Update the generator network each update_gen_timestep time steps")
    parser.add_argument("--update_shield_timestep", type=float, default=500,
                        help="Update the shield network each update_shield_timestep time steps")
    parser.add_argument("--gen_batch_size", type=float, default=1024,
                        help="Batch size to sample from buffer while updating generator")
    parser.add_argument("--shield_episodes_batch_size", type=float, default=3,
                        help="The number of episdoes from shield buffer while updating Shield")
    parser.add_argument("--generator_latent_dim", type=float, default=32,
                        help="The dimension of latent space (Generator)")

    args = parser.parse_args(arguments)
    # nv_name = "highway-three-v0"
    # env_name = "highway-v0"
    # env_name = "highway-v0"
    # env_name = "highway-fast-v0"
    # env_name = "intersection-v0"
    # env_name = "two-way-v0"
    # env_name = "HighwayEnvFastNoNormalization-v0"
    has_continuous_action_space = False
    unsafe_tresh = args.unsafe_tresh
    gen_masking_tresh = args.gen_masking_tresh
    max_ep_len = args.max_ep_len  # max timesteps in one episode
    max_training_timesteps = args.max_training_timesteps  # break training loop if timeteps > max_training_timesteps
    # print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
    # log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
    print_freq = args.print_freq  # print avg reward in the interval (in num timesteps)
    log_freq = args.log_freq  # log avg reward in the interval (in num timesteps)
    save_model_freq = args.save_model_freq  # save model frequency (in num timesteps)
    masking_threshold = args.masking_threshold
    update_gen_timestep = args.update_gen_timestep
    shield_episodes_batch_size = args.shield_episodes_batch_size
    action_std = None
    batch_size = args.gen_batch_size
    no_render = args.no_render
    action_std_decay_freq = None
    action_std_decay_rate = None
    min_action_std = None
    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 8  # update policy every n timesteps
    update_shield_timestep = args.update_shield_timestep
    k_epochs_ppo = args.k_epochs_ppo  # update policy for K epochs [10,50,100]
    k_epochs_shield = args.k_epochs_shield  # update shield for K epochs [10,50,100]
    k_epochs_gen = args.k_epochs_gen  # update shield for K epochs [10,50,100]

    eps_clip = args.eps_clip  # clip parameter for PPO  [0.1,0.2]
    gamma = args.gamma  # discount factor [0.9,0.95  0.99]
    shield_gamma = args.shield_gamma
    lr_actor = args.lr_actor  # learning rate for actor network [1e-4, 5e-4, 1e-3]
    lr_critic = args.lr_critic  # learning rate for critic network [1e-4, 5e-4, 1e-3]
    lr_shield = args.lr_shield
    lr_gen = args.lr_gen
    latent_dim = args.generator_latent_dim
    ############################################gy#########
    agent = args.algo
    register_envs(args.envs)
    env_list = [gym.make(x) for x in args.envs]
    random_seed = args.seed
    # env_list = [gym.make("MergeEnvNoNormalization-v0"),
    #             gym.make("HighwayEnvFastNoNormalization-v0"),
    #             gym.make("IntersectionEnvNoNormalization-v0"),
    #             gym.make("RoundaboutEnvNoNormalization-v0"),
    #             gym.make("TwoWayEnvNoNormalization-v0"),
    #             gym.make("UTurnEnvNoNormalization-v0"),
    #             gym.make("MergeEnvNoNormalization-v0")
    #             ]
    multi_task_env = EnvsWrapper(env_list, has_continuous_action_space, no_render)
    if args.record_mistakes:
        for env in multi_task_env.envs:
            env.metadata["video.frames_per_second"] = 30
            env.metadata["video.output_frames_per_second"] = 30
    action_dim = multi_task_env.action_space_size()
    if args.no_render:
        print("Rendering is disabled")
    else:
        print("Rendering is enabled")
    #### create new log file for each run
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_goal = "hi"
    base_path = f"./models/17.03.24/maximum_rewards/maximum_rewards_optuna"
    save_model_path = f"./{base_path}/model.pth"

    # Added another path to save the shield network (updated parameters)
    save_shield_path = f"./{base_path}/shield.pth"
    save_gen_path = f"./{base_path}/gen.pth"
    save_collision_info_path = f"./{base_path}/collision_info_log.log"
    save_stats_path = f"./{base_path}/stats.log"
    save_stats_path_renv = f"./{base_path}/stats_renv.log"

    save_shield_loss_stats_path = f"./{base_path}/shield_loss_stats.log"
    save_gen_loss_stats_path = f"./{base_path}/gen_loss_stats.log"
    save_args_path = f"./{base_path}/commandline_args.txt"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/Videos", exist_ok=True)
    # safe_rl_baselines = ['ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']
    # Input for gen - defines the parameters that we want to generate for the env configuration, it's range and type.
    """
    # Input for HighwayEnv
    param_ranges = {'speed_limit': {'range': (20, 40), 'type': float},
                    'vehicles_count': {'range': (10, 30), 'type': int},
                    'vehicles_density': {'range': (1, 5), 'type': int}, 'ego_spacing': {'range': (1, 3), 'type': float}}
    """
    #  If we change this dict, we also need to change it in the self.init (nonormalizationenv.py file) so it will be the same
    param_ranges = {
        'gravity': {'range': (9.0, 10.0), 'type': float},
        'masscart': {'range': (0.5, 2), 'type': float},
        'masspole': {'range': (0.05, 0.5), 'type': float},
        'length': {'range': (0.4, 0.6), 'type': float}
    }
    # Create agent object
    with open(save_args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if agent == "PPO":
        ppo_agent = PPO(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip,
                        has_continuous_action_space, action_std)
    elif agent == "ShieldPPO":
        ppo_agent = ShieldPPO(state_dim = multi_task_env.state_dim, action_dim = action_dim, lr_actor = lr_actor, lr_critic = lr_critic, gamma = gamma,  eps_clip = eps_clip, k_epochs_ppo = k_epochs_ppo, k_epochs_shield = k_epochs_shield, k_epochs_gen = k_epochs_gen,
                              has_continuous_action_space = has_continuous_action_space, lr_shield = lr_shield, lr_gen = lr_gen, latent_dim = latent_dim, shield_gamma = shield_gamma, action_std_init = action_std, masking_threshold=masking_threshold,
                              unsafe_tresh = unsafe_tresh, param_ranges=param_ranges)

    elif agent == 'PPOCAR':
        ppo_agent = PPOCostAsReward(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs,
                                    eps_clip,
                                    has_continuous_action_space, action_std)
    else:
        raise NotImplementedError
    time_step = 0
    time_step_renv = 0
    i_episode = 0
    time_steps = []
    rewards = []
    tasks = []
    episodes_len = []
    episodes_len_renv = []
    collision_info = {}
    costs = []
    shield_losses = []
    gen_losses = []
    stats = []
    start_time = time.time()
    amount_of_done = 0
    # Save times steps, rewards and costs for random environment , for Training Evaluation.
    amount_of_done_renv = 0
    time_steps_renv = []
    rewards_renv = []
    costs_renv = []
    # training loop
    shield_loss_update_stats = {}
    gen_loss_update_stats = {}
    steps_before_collision = None
    while time_step <= max_training_timesteps:
        if i_episode >= gen_masking_tresh:
            # using generator to get a generated configuration for env, and the first chosen action
            steps_before_collision = 0
            param_dict, unsafe_scores = ppo_agent.get_generated_env_config()
            state, state_vf = multi_task_env.reset(param_dict)
            gen_chosen_state = state
            # The first action to apply, chosen by the generator (the action with the maximal score)
            gen_chosen_action = unsafe_scores.index(max(unsafe_scores))
        else:
            # not using gen
            state, state_vf = multi_task_env.reset()
        task_name = multi_task_env.get_current_env_name()
        valid_actions = multi_task_env.get_valid_actions()
        trajectory = collections.deque([state])
        recorder_closed = False
        # INITIALIZE THE REWARDS & COSTS (CUMULATIVE)
        current_ep_reward = 0
        current_ep_cost = 0
        current_ep_len = 0
        # Save episode rewards and costs for random env episodes (renv), for Training Evaluation.
        current_ep_reward_renv = 0
        current_ep_cost_renv = 0
        current_ep_len_renv = 0
        # shield_epoch_trajectory is for Shield Buffer
        is_mistake = False
        if args.record_mistakes:
            base_video_path = f"./{base_path}/Videos"
            os.makedirs(base_path, exist_ok=True)
            video_path = f"{base_video_path}/episode_" + str(i_episode)
            trajectory_path = f"{base_video_path}/episode_" + str(i_episode) + "_trajectory.txt"
            video_recorder = VideoRecorder(multi_task_env.env, base_path=video_path, enabled=video_path is not None)
        shield_epoch_trajectory = []
        # Starting a new episode - length max_ep_len
        for t in range(1, max_ep_len + 1):
            if i_episode >= gen_masking_tresh:
                # An episode generated by GAN (in case of i_episode >= gen_masking_tresh)
                steps_before_collision += 1
            if args.render and not args.no_render:
                multi_task_env.env.render()
            prev_state = state.copy()
            if type(ppo_agent) == ShieldPPO:
                action, unsafe_scores = ppo_agent.select_action(state, valid_actions, time_step)
            else:
                action = ppo_agent.select_action(state, valid_actions, time_step)
            if (i_episode >= gen_masking_tresh) and (t == 1):
                # For the first time, the agent will choose the action chosen by generator in any case.
                action = gen_chosen_action
            state, reward, done, info = multi_task_env.step(action)
            trajectory.appendleft((action, state))
            if len(trajectory) > args.record_trajectory_length:
                trajectory.pop()
            if args.record_mistakes:
                video_recorder.capture_frame()
            # Adding trajectories to Buffers - ppo_agent buffer and shield buffer
            cost = info['cost']
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.costs.append(cost)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            current_ep_reward += reward
            current_ep_cost += cost
            shield_epoch_trajectory.append((torch.tensor(prev_state), torch.tensor([action]), cost, done))
            # Update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
            # Update Shield (more freuqnet)
            if time_step % update_shield_timestep == 0 and type(ppo_agent) == ShieldPPO:
                shield_loss = ppo_agent.update_shield(shield_episodes_batch_size)
                shield_losses.append(shield_loss)
                shield_loss_update_stats[time_step] = (i_episode, t, shield_loss)
            if time_step % update_gen_timestep == 0 and agent == "ShieldPPO" and i_episode >= gen_masking_tresh:
                gen_loss = ppo_agent.update_gen(batch_size)
                gen_loss_update_stats[time_step] = (i_episode, t, gen_loss)
                gen_losses.append(gen_loss)
            # cost - the real label of the action
            if info["cost"] > 0:
                is_mistake = True
            if agent == "ShieldPPO" or agent == "RuleBasedShieldPPO":
                if info["cost"] > 0:
                    # COLLISION
                    if args.record_mistakes:
                        video_recorder.close()
                        recorder_closed = True
                        with open(trajectory_path, 'w') as f:
                            for item in reversed(trajectory):
                                f.write(f"{item}\n")

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # Log in logging file
            if time_step % log_freq == 0:
                torch.save((time_steps, rewards, costs, tasks, time.time() - start_time, episodes_len, amount_of_done, i_episode),
                           save_stats_path)
                torch.save(shield_loss_update_stats, save_shield_loss_stats_path)
                torch.save(gen_loss_update_stats, save_gen_loss_stats_path)

            # Print average stats
            if time_step % print_freq == 0:
                recent_reward = np.array(rewards[max(0, len(rewards) - 10):]).mean()
                recent_cost = np.array(costs[max(0, len(costs) - 10):]).mean()
                if agent == "ShieldPPO":
                    recent_shield_loss = np.array(shield_losses[max(0, len(shield_losses) - 10):]).mean()
                    recent_gen_loss = np.array(gen_losses[max(0, len(gen_losses) - 10):]).mean()
                    print("Time Step is", time_step)
                    print(
                        f"Episode : {i_episode:4d} Reward {recent_reward:6.2f} Cost {recent_cost:6.2f} Shield Loss {recent_shield_loss:6.2f} Time {(time.time() - start_time) / 60:.2f} min")
                else:
                    print(
                        f"Episode : {i_episode:4d} Reward {recent_reward:6.2f} Cost {recent_cost:6.2f} Time {(time.time() - start_time) / 60:.2f} min")

            # Save model weights
            if time_step % save_model_freq == 0:
                # ppo_agent.save(save_model_path, save_shield_path, save_gen_path)
                if type(ppo_agent) == ShieldPPO:
                    ppo_agent.save(save_model_path, save_shield_path, save_gen_path)
                else:
                    ppo_agent.save(save_model_path)
                # THE EPOCH FINISHES IF DONE == TRUE (REACHED TO FINAL STATE) OR REACHED TO MAX_EP_LEN
            if done:
                amount_of_done += 1
                break
        # IN THE END OF EACH EPOCH
        if i_episode >= gen_masking_tresh:
            # In case of using generator - saving gen_chosen_state in list (same structure as k_last_states, as it is sent to the shield in the Gen.loss())
            if steps_before_collision == 1:
                steps_before_collision = 0
            ppo_agent.add_to_gen(gen_chosen_state, gen_chosen_action, steps_before_collision)
        # Save to shield buffer in the end of each episode a list of [(s1,a1,cost1,done1),.............] for all steps in episode
        ppo_agent.add_to_shield(shield_epoch_trajectory)
        # Save episode stats (rewards, costs, task, time_steps and episodes_length)
        rewards.append(current_ep_reward)
        costs.append(current_ep_cost)
        tasks.append(task_name)
        time_steps.append(time_step)
        episodes_len.append(current_ep_len)
        if args.record_mistakes and not recorder_closed:
            video_recorder.close()
            if not is_mistake:
                os.remove(video_recorder.path)
                os.remove(video_recorder.metadata_path)
            else:
                with open(trajectory_path, 'w') as f:
                    for item in reversed(trajectory):
                        f.write("%s\n" % item)
        # Initialize a random environment - for Training Evaluation.
        # Another episode - with a random env (not generated by Gan) - for Training Evaluation.
        state_renv, state_vf_renv = multi_task_env.reset()
        for t_renv in range(1, max_ep_len+1):
            if args.render and not args.no_render:
                multi_task_env.env.render()
            prev_state_renv = state_renv.copy()
            if type(ppo_agent) == ShieldPPO:
                action_renv, unsafe_scores_renv = ppo_agent.select_action(state_renv, valid_actions, time_step_renv, evaluation = True)
            else:
                # Sending Evaluation = True so it will not save the rewards/ etc in the ppo_agent
                action_renv, unsafe_scores_renv = ppo_agent.select_action(state_renv, valid_actions, time_step_renv, evaluation = True )
            state_renv, reward_renv, done_renv, info_renv = multi_task_env.step(action_renv)
            time_step_renv  += 1
            current_ep_reward_renv += reward_renv
            current_ep_cost_renv += info_renv['cost']
            # Log in logging file
            if time_step_renv % log_freq == 0:
                torch.save((time_steps_renv, rewards_renv, costs_renv, time.time() - start_time, episodes_len_renv, amount_of_done_renv, i_episode),
                           save_stats_path_renv)
            if done_renv:
                amount_of_done_renv +=1
                break
        rewards_renv.append(current_ep_reward_renv)
        costs_renv.append(current_ep_cost_renv)
        time_steps_renv.append(time_step_renv)
        episodes_len_renv.append(current_ep_len_renv)

        # i_episodes = counts the amount of episodes
        i_episode += 1
    # The time  now includes the training loop so it's not relevant.
    end_time = time.time()
    total_training_time = end_time - start_time
    torch.save((time_steps, rewards, costs, tasks, total_training_time, episodes_len, amount_of_done, i_episode), save_stats_path)
    torch.save((time_steps_renv, rewards_renv, costs_renv, total_training_time, episodes_len_renv, amount_of_done_renv, i_episode), save_stats_path_renv)
    # list of tuples - tuple for each update time step, the first element of each tuple is the time_step, than episode, step in episode, etc.
    torch.save(gen_loss_update_stats, save_gen_loss_stats_path)
    torch.save(shield_loss_update_stats, save_shield_loss_stats_path)
    multi_task_env.close()


if __name__ == '__main__':
    train()
