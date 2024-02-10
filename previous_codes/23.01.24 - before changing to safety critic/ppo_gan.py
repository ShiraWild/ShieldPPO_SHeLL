"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

################################## set device ##################################

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        # MeetingComment - changed states so now it stored a tuple - (k_last_states, last_informative_layer )
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.costs = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.costs[:]



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init).to(device)
        # actor
        # for a given states returns the safety score per each action
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, action_dim),
                nn.Softmax(dim=-1)
            )
        # critic
        # for a given state - "how good is"
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy




class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gen_output_dim,
                 has_continuous_action_space, action_std_init=0.6, k_last_states = 1):

        self.has_continuous_action_space = has_continuous_action_space

        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, valid_actions, time_step = None):
        # initalize tensor size action_dim (amount of actions to choose) with zeros
        valid_mask = torch.zeros(self.action_dim).to(device)
        # fill it with 1 in the indices of the valid-actions -> filter unwanted actions. 1 = valid action, 0 = not valid
        for a in valid_actions:
            valid_mask[a] = 1.0
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                # predict the action_probs based on the last state using the actor network
                action_probs = self.policy_old.actor(state)
                # copy the actions probs with clone()
                action_probs_ = action_probs.clone()
                # in the indices that are not valid - put -inf so that it will not choose it
                action_probs_[valid_mask == 0] = -1e10
                # apply softmax (probabilities)
                action_probs_ = F.softmax(action_probs_)
                # categorial distribution - based on the original distribution (before softmax) and the soft-max probabilities
                dist = Categorical(action_probs_)
                dist2 = Categorical(action_probs)
                # sample based on score (an action)
                action = dist.sample()
                # compute it's log prob
                action_logprob = dist2.log_prob(action)
            # apply the buffer - (state, action)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            # return the chosen action - flattened
            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action_probs = self.policy_old.actor(state)
                action_probs_ = action_probs.clone()
                action_probs_[valid_mask == 0] = -1e10
                action_probs_ = F.softmax(action_probs_)
                dist = Categorical(action_probs_)
                dist2 = Categorical(action_probs)
                action = dist.sample()
                action_logprob = dist2.log_prob(action)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            return action.item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))"""


import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
# set device to cpu or cuda
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.costs = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.costs[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, action_dim),
                nn.Softmax(dim=-1)
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip, gen_output_dim, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs_ppo = k_epochs_ppo

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, valid_actions, time_step):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item(),

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        # old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        # advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs_ppo):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class GeneratorBuffer:
    def __init__(self, n_max=1000000):
        self.n_max = n_max
        # PFL - LATER: Maybe here we will save the original prediction of the shield in the buffers / use mutual buffer for shield and generator
        # collision_rb - a buffer that saves tuples of (s, last_informative_layer, a) -> in case there was a collision (the label is positive)
        self.collision_rb = []
        # no_collision_rb - a buffer that saves tuples of (s, last_informative_layer, a) -> in case there was not a collision (the label is negative)
        self.no_collision_rb = []

    def __len__(self):
        # returns the length of both collision and no collision buffers
        return len(self.collision_rb) + len(self.no_collision_rb)

    def add(self, s,  a, label):
        """
        The generator buffer saves (s,a, label) each epoch:
        s is the initial state generated by the generator
        a is the initial action generated by the generator
        label determined wether there was a collision throughout the epoch
        """
        # label = if there was a collision or not.
        if label > 0.5:  # there wasn't a collision, add to no_collision_rb
            self.no_collision_rb.append((s, a))
        else: # COLLISION -> add to collision_rb
            self.collision_rb.append((s,  a))

    def sample(self, n):
       # n_neg = min(n // 2, len(self.neg_rb))
        n_collision = min(max(n // 2,n-len(self.no_collision_rb)), len(self.collision_rb))
        # original - n_pos = n - n_neg
        n_no_collision = min(n - n_collision, len(self.no_collision_rb))
        no_collision_batch = random.sample(self.no_collision_rb, n_no_collision)
        # s_pos, a_pos = map(np.stack, zip(*pos_batch))
        s_no_collision = np.array([item[0] for item in no_collision_batch])
        a_no_collision = np.array([item[1] for item in no_collision_batch])
        collision_batch = random.sample(self.collision_rb, n_collision)
        # s_neg, a_neg = map(np.stack, zip(*neg_batch))
        s_collision = np.array([item[0] for item in collision_batch])
        a_collision = np.array([item[1] for item in collision_batch])
        return torch.FloatTensor(s_no_collision).to(device), torch.LongTensor(a_no_collision).to(device), \
               torch.FloatTensor(s_collision).to(device), torch.LongTensor(a_collision).to(device)


class Generator(nn.Module):
    def __init__(self, action_dim, param_ranges = None,  latent_dim = 32):
        """
        output_dim - set according to the number of parameters values to be predicted
        latent_dim - to sample from
        param_ranges - limitation for the parameters values
        """
        super(Generator, self).__init__()
        # PFL - latent_dim would be controlled by HYPERPARAMETER - make it a paramater
        self.latent_dim = latent_dim
        self.output_dim = len(param_ranges)
        self.params = list(param_ranges.keys())
        self.param_ranges = param_ranges
        self.loss_fn = nn.BCELoss()
        self.action_dim = action_dim
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # output shape - output_dim (same as the amount of parameters to generate) + action_dim (safety score per each action)
            nn.Linear(64, self.output_dim + self.action_dim)
        )

    def forward(self):
        noise = torch.randn(1, self.latent_dim).to(device)
        gen_output = self.model(noise)
        # Separate the output into parameters and safety scores
        param_output = gen_output[:, :self.output_dim]
        # Each head of safety_score_output represents a safety score value for a specific action
        safety_score_output = gen_output[:, self.output_dim:]
        normalized_params = []
        # Normalize the output according to given ranges in self.speed_limit
        # Each 'head' of the output dim represents a generated parameter value
        for i in range(param_output.size(1)):
            param_range = self.param_ranges[self.params[i]]['range']
            param_type =  self.param_ranges[self.params[i]]['type']
            param_output_normalized = F.sigmoid(gen_output[:, i]) * (param_range[1] - param_range[0]) + param_range[0]
            if param_type == int:
                param_output_normalized =  param_output_normalized.int()
            normalized_params.append(param_output_normalized.item())
        safety_scores = F.softmax(safety_score_output, dim = 1)
        param_dict = dict(zip(self.params, normalized_params))
        return param_dict, safety_scores.squeeze().tolist() # {'p1': 30, .. }, [s1, s2, .., s5]

    # pos - with collision
    # neg - without collision
    def loss(self, shield, s_collison, a_collision, s_no_collision,  a_no_collision):
        # TODD -  last_informative layer is len(s_collision) or len(s_no_collision) because there is no padding - only 1 state
        # last_informative_layer_repeated = torch.Tensor([len(states)-1] * self.action_dim).to(torch.int)
        batch_size_collision, batch_size_no_collision = len(s_collison), len(s_no_collision)
        y_shield_collision = shield.forward(s_collison,  a_collision)
        y_shield_no_collision = shield.forward(s_no_collision, a_no_collision)
        # Loss explaination - if there was a collision the label is 1 (becuase we want it to predict it as safe), and vice versa
        loss_collision = self.loss_fn(y_shield_collision, torch.ones_like(y_shield_collision))
        loss_no_collision = self.loss_fn(y_shield_no_collision, torch.zeros_like(y_shield_no_collision))
        loss = loss_collision + loss_no_collision
        return loss, y_shield_collision, y_shield_no_collision

    """
    # Old function without using BCE
    def loss(self, shield, s_collison, last_informative_layers_collision, a_collision, s_no_collision,
             last_informative_layers_no_collision, a_no_collision):
        y_desc_collision = shield.forward(s_collison,last_informative_layers_collision, a_collision)
        y_desc_no_collision = shield.forward(s_no_collision,last_informative_layers_no_collision, a_no_collision)
        loss_collision = -1 * torch.log(y_desc_collision).mean()
        loss_no_collision = -1 * torch.log(1 - y_desc_no_collision).mean()
        total_loss = loss_collision + loss_no_collision
        return total_loss

    """


class ShieldBuffer:
    def __init__(self, n_max=1000000):
        self.n_max = n_max
        # pos_rb - a buffer that saves tupple, each tuple is (s, last_informative_layer,a) - last_informative_layer before padding. for positive actions.
        # neg_rb - a buffer that saves tupple, each tuple is (s, last_informative_layer,a) - last_informative_layer before padding. for negative actions.
        # positive_action - no collision, negative_actions - result collision
        self.pos_rb = []
        self.neg_rb = []

    def move_last_pos_to_neg(self):
        # Error Diffusion - "no safe action according to shield"
        if len(self.pos_rb) > 1:
            self.neg_rb.append(self.pos_rb[-1])
            self.pos_rb = self.pos_rb[:-1]

    def __len__(self):
        return len(self.pos_rb) + len(self.neg_rb)

    def add(self,s,a, label):
        if label > 0.5:  # NO COLLISION - SAFE
            self.pos_rb.append((s, a))
        else: # COLLISION
            self.neg_rb.append((s,  a))

    def sample(self, n):
       # n_neg = min(n // 2, len(self.neg_rb))
        n_neg = min(max(n // 2,n-len(self.pos_rb)), len(self.neg_rb))
        # original - n_pos = n - n_neg
        n_pos = min(n - n_neg, len(self.pos_rb))
        pos_batch = random.sample(self.pos_rb, n_pos)
        s_pos, a_pos = map(np.stack, zip(*pos_batch))
        neg_batch = random.sample(self.neg_rb, n_neg)
        s_neg, a_neg = map(np.stack, zip(*neg_batch))
        return torch.FloatTensor(s_pos).to(device), torch.LongTensor(a_pos).to(device), \
               torch.FloatTensor(s_neg).to(device), torch.LongTensor(a_neg).to(device)



class Shield(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space):
        super().__init__()
        hidden_dim = 256
        self.has_continuous_action_space = has_continuous_action_space
        if not self.has_continuous_action_space:
            self.action_embedding = nn.Embedding(action_dim, action_dim)
            self.action_embedding.weight.data = torch.eye(action_dim)
        #self.lstm = nn.LSTM(state_dim , hidden_dim, num_layers=1, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        self.loss_fn = nn.BCELoss()


    def encode_action(self, a):
        if self.has_continuous_action_space:
            return a
        else:
            return self.action_embedding(a)

    def forward(self, s, a):
        a = self.encode_action(a)
        x = torch.cat([s, a], -1)
        return self.net(x)


    def loss(self, s_pos, a_pos, s_neg, a_neg):
        a_pos = self.encode_action(a_pos)
        a_neg = self.encode_action(a_neg)
        x_pos = torch.cat([s_pos, a_pos], -1)
        x_neg = torch.cat([s_neg, a_neg], -1)
        y_pos = self.net(x_pos)
        y_neg = self.net(x_neg)
        loss = self.loss_fn(y_pos, torch.ones_like(y_pos)) + self.loss_fn(y_neg, torch.zeros_like(y_neg))
        return loss, y_pos, y_neg



class ShieldPPO(PPO):  # currently only discrete action
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, eps_clip,k_epochs_ppo, k_epochs_shield, k_epochs_gen,
                 gen_output_dim, has_continuous_action_space,shield_lr, gen_lr,  action_std_init=0.6, masking_threshold=0, safety_treshold = 0.5,  param_ranges = None,  latent_dim = 32):
        super().__init__(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip, gen_output_dim, has_continuous_action_space, action_std_init)
        self.action_dim = action_dim
        self.shield = Shield(state_dim, action_dim, has_continuous_action_space).to(device)
        self.gen = Generator(self.action_dim, param_ranges, latent_dim).to(device)
        self.shield_opt = torch.optim.Adam(self.shield.parameters(), lr= shield_lr)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr= gen_lr)
        self.shield_buffer = ShieldBuffer()
        self.gen_buffer = GeneratorBuffer()
        self.masking_threshold = masking_threshold
        self.state_dim = state_dim
        self.safety_treshold = safety_treshold
        self.k_epochs_shield = k_epochs_shield
        self.k_epochs_gen = k_epochs_gen
        self.k_epochs_ppo = k_epochs_ppo

    def move_last_pos_to_neg(self):
        # Error Diffusion - "no safe error according to shield"
        # in this case we assume that it happens because there are no safe actions from the given state
        # so we want to teach the agent that the prev state and the action that led to the current state is a bad example for shield buffer.
        self.shield_buffer.move_last_pos_to_neg()

    def add_to_shield(self, s, a, label):
        self.shield_buffer.add(s, a, label)

    def add_to_gen(self, s, a, label):
        self.gen_buffer.add(s, a, label)


    def update_shield(self, batch_size):
        if len(self.shield_buffer.neg_rb) == 0:
            return 0, 0, 0,0,0
        if len(self.shield_buffer) <= batch_size:
            batch_size = len(self.shield_buffer)
        loss_, y_pos_, y_neg_ = 0, 0, 0
        len_y_pos, len_y_neg = len(self.shield_buffer.pos_rb), len(self.shield_buffer.neg_rb)
        for i in range(self.k_epochs_shield):
            s_pos, a_pos, s_neg, a_neg = self.shield_buffer.sample(batch_size)
            self.shield_opt.zero_grad()
            loss, y_pos, y_neg = self.shield.loss(s_pos,  a_pos, s_neg, a_neg)
            y_pos_ += y_pos.mean().item()
            y_neg_ += y_neg.mean().item()
            loss.backward()
            self.shield_opt.step()
            loss_ += loss.item()
        avg_y_pos =  y_pos_ / self.k_epochs_shield
        avg_y_neg = y_neg_ / self.k_epochs_shield
        return loss_ / self.k_epochs_shield, len_y_pos,  avg_y_pos, len_y_neg, avg_y_neg



    def update_gen(self, batch_size):
        if (len(self.gen_buffer.collision_rb) == 0) or (len(self.gen_buffer.no_collision_rb) == 0):
            return 0, 0,0,0,0
        if len(self.gen_buffer) <= batch_size:
            batch_size = len(self.gen_buffer)
        loss_ = 0.
        y_collision_ = 0
        y_no_collision_ = 0
        len_y_collision, len_y_no_collision = len(self.gen_buffer.collision_rb), len(self.gen_buffer.no_collision_rb)
        # k steps to update the shield network - each epoch (step) is one forward and backward pass
        for i in range(self.k_epochs_gen):
            # in each iteration - it samples batch_size positive and negative samples
            #  samples list of k_last_states states. for each sample.
            s_no_collision, a_no_collision, s_collision, a_collision = self.gen_buffer.sample(batch_size)
            self.gen_opt.zero_grad()
            # compute loss - binary cross entropy
            loss, y_shield_collision, y_shield_no_collision = self.gen.loss(self.shield, s_collision, a_collision, s_no_collision, a_no_collision)
            y_collision_ += y_shield_collision.mean().item()
            y_no_collision_ += y_shield_no_collision.mean().item()
            # back propagation
            loss.backward()
            # updating the shield parameters using Adam optimizer
            self.gen_opt.step()
            loss_ += loss.item()
        avg_y_collision = y_collision_ / self.k_epochs_gen
        avg_y_no_collision = y_no_collision_ / self.k_epochs_gen
        print("Finished updating Generator.")
        return loss_ / self.k_epochs_gen, len_y_collision, avg_y_collision, len_y_no_collision, avg_y_no_collision

    def get_generated_env_config(self):
        return self.gen()


    def select_action(self, state, valid_actions, timestep, evaluation = False):
        valid_mask = torch.zeros(self.action_dim).to(device)
        no_safe_action = False
        for a in valid_actions:
            valid_mask[a] = 1.0
        with torch.no_grad():
            # the last state is the states[-1]  - because there is no padding
            state = torch.FloatTensor(state).to(device)
            action_probs = self.policy_old.actor(state)
            # it will have self.k_last_states rows, each with the same action indices - for the batch prediction
            actions = torch.arange(self.action_dim).to(device)  # (n_action,)
            #actions_ = [tensor.unsqueeze(0) for tensor in actions]
            # states_ - a batch of self.action_dim repeated states_ tensor, if self.action_dim is 5 so the batch_size is equal to 5.
            state_ = state.view(1, -1).repeat(self.action_dim, 1)  # (n_action, state_dim)
            if timestep >= self.masking_threshold:
                #  print("USING SAFETY MASKING")
                # BATCH_SIZE = self.action_dim,  EACH SEQUENCE LENGTH IS - K_LAST_STATES
                # Send the Shield network a batch of size self.action_dim - here there is no padding
                safety_scores = self.shield(state_, actions)
                mask = safety_scores.gt(self.safety_treshold).float().view(-1).to(device)
                mask = valid_mask * mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                    # AT LEAST ONE ACTION IS CONSIDERED SAFE
                    action_probs_[mask == 0] = -1e10
                else:
                    # No safe action according to shield
                    # If it happened - it means that the 'problem' is in the action selected by previous state. The state is not safe so none of the actions is safe according to shield.
                    print("No safe action according to shield")
                    no_safe_action = True
                    # Solution - Error Diffusion - teach  the network that the prev state prev state is not safe (from highway.py)
                    action_probs_[valid_mask == 0] = -1e10
            else:
                #   print("NOT USING SAFETY MASKING")
                mask = valid_mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                    # at least one of the actions is safe according to shield or valid
                    action_probs_[mask == 0] = -1e10
                else:
                    print("NO VALID ACTIONS AT ALL - SHOULDN'T HAPPEN ")
                    action_probs_[valid_mask == 0] = -1e10
            action_probs_ = F.softmax(action_probs_)
            dist = Categorical(action_probs_)
            dist2 = Categorical(action_probs)
        # action - [k_last_states, action_dim] -
        action = dist.sample()
        action_logprob = dist2.log_prob(action)
        if not evaluation:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
        if timestep >= self.masking_threshold:
            return action.item(), safety_scores, no_safe_action
        else:
            return action.item(), "no shield yet", False

    def save(self, checkpoint_path_ac, checkpoint_path_shield, checkpoint_path_gen):
        # save actor critic networks
        torch.save(self.policy_old.state_dict(), checkpoint_path_ac)
        # save shield network
        torch.save(self.shield.state_dict(), checkpoint_path_shield)
        # save gen network
        torch.save(self.gen.state_dict(), checkpoint_path_gen)


    def load(self, checkpoint_path_ac, checkpoint_path_shield, checkpoint_path_gen):
        # Load the models - Shield, policy, old_policy, Gen
        self.policy_old.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.shield.load_state_dict(torch.load(checkpoint_path_shield, map_location=lambda storage, loc: storage))
        self.gen.load_state_dict(torch.load(checkpoint_path_gen, map_location=lambda storage, loc: storage))


class RuleBasedShieldPPO:
    pass


class PPOCostAsReward:
    pass