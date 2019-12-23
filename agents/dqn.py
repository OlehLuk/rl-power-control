# implementation based on tutorials:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda

import gym
import math
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Observation = namedtuple('Observation', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save an observation."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Observation(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MlpDqn(nn.Module):
    def __init__(self, n_state_input, n_hidden_1, n_hidden_2, n_actions):
        super(MlpDqn, self).__init__()
        self.fc1 = nn.Linear(n_state_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.out = nn.Linear(n_hidden_2, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DqnAgent:
    def __init__(self, actions, n_state_variables, n_hidden_1, n_hidden_2,
                 buffer_size, batch_size, exploration_rate, expl_rate_decay, expl_rate_final,
                 discount_factor):
        n_actions = len(actions)
        self.actions = actions
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.expl_rate_decay = expl_rate_decay
        self.expl_rate_final = expl_rate_final
        self.batch_size = batch_size
        self.step_counter = 0
        self.actions = actions
        self.model = MlpDqn(n_state_variables, n_hidden_1, n_hidden_2, n_actions)
        self.buffer = ReplayBuffer(buffer_size)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def learn(self, state, reward, next_state, done):
        self.buffer.push(state, reward, next_state, done)
        if len(self.buffer) < self.batch_size:
            return
        if self.step_counter % self.batch_size == 0:
            batch = self.buffer.sample(self.batch_size)
            self.train_dqn(batch)

    def use(self, state):
        optimal_action_index = self.model.forward(state).argmax()
        return self.actions[optimal_action_index]

    def train_dqn(self, batch):
        loss = self.calc_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calc_loss(self, batch):
        batch = Observation(*zip(*batch))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss


class DqnAgentTargetNet:
    def __init__(self, actions, n_state_variables, n_hidden_1, n_hidden_2,
                 buffer_size, batch_size, exploration_rate, expl_rate_decay, expl_rate_final,
                 discount_factor, target_update):
        n_actions = len(actions)
        self.actions = actions
        self.target_update = target_update
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.expl_rate_decay = expl_rate_decay
        self.expl_rate_final = expl_rate_final
        self.step_counter = 0
        self.batch_size = batch_size

        self.model = MlpDqn(n_state_variables, n_hidden_1, n_hidden_2, n_actions)
        self.model_target = MlpDqn(n_state_variables, n_hidden_1, n_hidden_2, n_actions)
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval()
        self.buffer = ReplayBuffer(buffer_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def learn(self, state, reward, next_state, done):
        self.buffer.push(state, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        self.train_dqn(batch)

        self.step_counter += 1

        if self.step_counter % self.target_update == 0:
            self.model_target.load_state_dict(self.model.state_dict())

    def use(self, state):
        if random.random() > self.get_current_expl_rate():
            optimal_action_index = self.model.forward(state).argmax()
            return self.actions[optimal_action_index]
        else:
            return random.sample(self.actions)

    def train_dqn(self, batch):
        loss = self.calc_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calc_loss(self, batch):
        batch = Observation(*zip(*batch))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.model_target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss

    def get_current_expl_rate(self):
        to_return = self.exploration_rate
        self.exploration_rate *= self.expl_rate_decay
        if self.exploration_rate < self.expl_rate_final:
            self.exploration_rate = self.expl_rate_final
        return to_return


