import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

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
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
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
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

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

    def select_action(self, state, valid_actions):
        valid_mask = torch.zeros(self.action_dim).to(device)
        for a in valid_actions:
            valid_mask[a] = 1.0
        if self.has_continuous_action_space:
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
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class PPOCostAsReward(PPO):

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal, cost in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals),
                                             reversed(self.buffer.costs)):
            if is_terminal:
                discounted_reward = 0
            if cost > 0:
                reward = -1000000
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


class ShieldBuffer:
    def __init__(self, n_max=1000000):
        self.n_max = n_max
        self.pos_rb = []
        self.neg_rb = []

    def __len__(self):
        return len(self.pos_rb) + len(self.neg_rb)

    def add(self, s, a, label):
        if label > 0.5:  # safe
            self.pos_rb.append((s, a))
        else:
            self.neg_rb.append((s, a))

    def sample(self, n):
        n_neg = min(n // 2, len(self.neg_rb))
        n_pos = n - n_neg
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
        return loss


class ShieldPPO(PPO):  # currently only discrete action
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):
        super().__init__(state_dim, action_dim,
                         lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                         has_continuous_action_space, action_std_init)
        self.action_dim = action_dim
        self.shield = Shield(state_dim, action_dim, has_continuous_action_space).to(device)
        self.shield_opt = torch.optim.Adam(self.shield.parameters(), lr=5e-4)
        self.shield_buffer = ShieldBuffer()

    def add_to_shield(self, s, a, label):
        self.shield_buffer.add(s, a, label)

    def update_shield(self, batch_size):
        if len(self.shield_buffer.neg_rb) == 0:
            return 0.
        if len(self.shield_buffer) <= batch_size:
            batch_size = len(self.shield_buffer)
        loss_ = 0.
        for i in range(self.K_epochs):
            s_pos, a_pos, s_neg, a_neg = self.shield_buffer.sample(batch_size)
            self.shield_opt.zero_grad()
            loss = self.shield.loss(s_pos, a_pos, s_neg, a_neg)
            loss.backward()
            self.shield_opt.step()
            loss_ += loss.item()
        return loss_ / self.K_epochs

    def select_action(self, state, valid_actions):
        valid_mask = torch.zeros(self.action_dim).to(device)
        for a in valid_actions:
            valid_mask[a] = 1.0

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action_probs = self.policy_old.actor(state)

            actions = torch.arange(self.action_dim).to(device)  # (n_action,)
            state_ = state.view(1, -1).repeat(self.action_dim, 1)  # (n_action, state_dim)
            safety = self.shield(state_, actions)  # (n_action)
            mask = safety.gt(0.5).float().view(-1)
            mask = valid_mask * mask

            action_probs_ = action_probs.clone()
            if mask.sum().item() > 0:
                action_probs_[mask == 0] = -1e10
            else:
                action_probs_[valid_mask == 0] = -1e10

            action_probs_ = F.softmax(action_probs_)
            dist = Categorical(action_probs_)
            dist2 = Categorical(action_probs)

        # found_valid_action = False
        # while not found_valid_action:
        #    action = dist2.sample()
        #    if action in valid_actions:
        #        found_valid_action = True

        action = dist.sample()
        action_logprob = dist2.log_prob(action)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item()


class RuleBasedShieldPPO(PPO):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, safety_rules, action_std_init=0.6):
        super().__init__(state_dim, action_dim,
                         lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                         has_continuous_action_space, action_std_init)
        self.action_dim = action_dim
        self.safety_rules = safety_rules

    def add_to_shield(self, s, a, label):
        if not label:
            # print(s,a)
            # print('A mistake was detected')
            input("A mistake was detected, update rules and press Enter to continue...")
            self.safety_rules.parse_rules()

    def select_action(self, state, valid_actions):
        valid_mask = torch.zeros(self.action_dim).to(device)
        for a in valid_actions:
            valid_mask[a] = 1.0

        with torch.no_grad():
            original_state_representation = state
            state = torch.FloatTensor(state).to(device)
            action_probs = self.policy_old.actor(state)
            mask = [
                (1 if self.safety_rules.check_state_action_if_safe(original_state_representation, action_as_int) else 0)
                for action_as_int in range(self.action_dim)]
            mask = torch.FloatTensor(mask).to(device)
            mask = valid_mask * mask

            action_probs_ = action_probs.clone()
            if mask.sum().item() > 0:
                action_probs_[mask == 0] = -1e10
            else:
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

        # action_safe = False
        # action_valid = False
        # while (not action_safe) or (not action_valid):
        #     action, action_logprob = self.policy_old.act(state)
        #     action_as_int = action.detach().cpu().numpy().flatten() if self.has_continuous_action_space else action.item()
        #     action_safe = self.safety_rules.check_state_action_if_safe(original_state_representation, action_as_int)
        #     action_valid = action_as_int in valid_actions
        # self.buffer.states.append(state)
        # self.buffer.actions.append(action)
        # self.buffer.logprobs.append(action_logprob)
        # return action_as_int

        # state = torch.FloatTensor(state).to(device)
        # action_probs = self.policy_old.actor(state)
        #
        # actions = torch.arange(self.action_dim).to(device) # (n_action,)
        # state_ = state.view(1, -1).repeat(self.action_dim, 1) # (n_action, state_dim)
        # safety = self.shield(state_, actions) # (n_action)
        # mask = safety.gt(0.5).float().view(-1)
        # mask = valid_mask * mask
        #
        # action_probs_ = action_probs.clone()
        # if mask.sum().item() > 0:
        #     action_probs_[mask == 0] = -1e10
        # else:
        #     action_probs_[valid_mask == 0] = -1e10
        #
        # action_probs_ = F.softmax(action_probs_)
        # dist = Categorical(action_probs_)
        # dist2 = Categorical(action_probs)
        # up to here
        # with torch.no_grad():
        #     state = torch.FloatTensor(state).to(device)
        #     if self.has_continuous_action_space:
        #         action_safe = False
        #         while not action_safe:
        #             action, action_logprob = self.policy_old.act(state)
        #             action_safe = self.shield(state, action)
        #         self.buffer.states.append(state)
        #         self.buffer.actions.append(action)
        #         self.buffer.logprobs.append(action_logprob)
        #
        #         return action.detach().cpu().numpy().flatten()
        #
        #     else:ש
        #         action_probs = self.policy_old.actor(state)
        #
        #         actions = torch.arange(self.action_dim).to(device)  # (n_action,)
        #         state_ = state.view(1, -1).repeat(self.action_dim, 1)  # (n_action, state_dim)
        #         safety = self.shield(state_, actions)  # (n_action)
        #         mask = safety.gt(0.5).float()
        #
        #         if mask.sum().item() < 1:  # no safe action
        #             action_probs_ = action_probs
        #         else:
        #             action_probs_ = action_probs * mask.view(*action_probs.shape)
        #         dist1 = Categorical(action_probs)
        #         dist2 = Categorical(action_probs_)
        #
        #         action = dist2.sample()
        #         action_logprob = dist1.log_prob(action)
        #
        #         self.buffer.states.append(state)
        #         self.buffer.actions.append(action)
        #         self.buffer.logprobs.append(action_logprob)
        #         return action.item()