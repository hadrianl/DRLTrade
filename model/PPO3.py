#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 0025 11:03
# @Author  : Hadrianl 
# @File    : PPO3


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
from torch.distributions import Categorical
import time
import numpy as np
from collections import defaultdict
import datetime as dt
from memory_profiler import profile
from utils import Writer
import os
from pathlib import Path

# Hyperparameters
learning_rate = 0.0001
gamma = 0.9999
lmbda = 0.95
ent = 0.001
eps_clip = 0.1
K_epoch = 3
T_horizon = 20


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.lstm = nn.LSTM(256, 128)
        self.fc_actor = nn.Linear(128, action_dim)

    def forward(self, x, hidden):
        # x -> o h l c v p
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        x = x.view(-1, 1, 256)
        x, hidden = self.lstm(x, hidden)
        x = self.fc_actor(x)
        prob = F.softmax(x, dim=2)
        return prob, hidden

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.lstm = nn.LSTM(256, 128)
        self.fc_critic = nn.Linear(128, 1)

    def forward(self, x, hidden):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        x = x.view(-1, 1, 256)
        x, _ = self.lstm(x, hidden)
        x = self.fc_critic(x)
        return x

class PPO:
    ACTOR_LREANING_RATE = 1e-3
    CRITIC_LREANING_RATE = 1e-4
    PPO_EPOCH = 3
    def __init__(self, state_dim):
        self.data = defaultdict(list)
        self.actor_net = Actor(state_dim, 3)
        self.critic_net = Critic(state_dim)
        self.lstm_hidden = (torch.randn([1, 1, 128], dtype=torch.float, requires_grad=False), torch.randn([1, 1, 128], dtype=torch.float, requires_grad=False))
        self.acotr_optimizer = optim.Adam(self.actor_net.parameters(), self.ACTOR_LREANING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), self.CRITIC_LREANING_RATE)

        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')

    def load_params(self, actor_name='', critic_name=''):
        print('load_params')
        # model_name = sys.argv[1] if len(sys.argv) > 1 else model_name
        net_path = Path('./param/net_param')
        if not actor_name and not critic_name:
            l = sorted((f for f in os.listdir(net_path)), key=lambda f: os.path.getmtime(os.path.join(net_path, f)))
            if len(l) >= 2:
                actor_name, critic_name = l[-2:]
        print(actor_name, critic_name)
        try:
            actor_path = Path.joinpath(net_path, actor_name)
            critic_path = Path.joinpath(net_path, critic_name)
            self.actor_net.load_state_dict(torch.load(actor_path))
            self.critic_net.load_state_dict(torch.load(critic_path))
            self.actor_net.eval()
            self.critic_net.eval()
        except Exception as e:
            print(f'no model is found, {e}')
        finally:
            print(f'load actor_net state: {self.actor_net.state_dict()}')
            print(f'load critic_net state: {self.critic_net.state_dict()}')

    def save_param(self):
        torch.save(self.actor_net.state_dict(), './param/net_param/actor_net'+str(time.time())[:10] +'.pkl')
        torch.save(self.critic_net.state_dict(), './param/net_param/critic_net'+str(time.time())[:10] +'.pkl')

    def select_action(self, state):
        hidden_in = self.lstm_hidden
        prob, hidden_out = self.actor_net(torch.from_numpy(state).float(), hidden_in)
        prob = prob.view(-1)
        action = Categorical(prob).sample().item()
        self.lstm_hidden = (hidden_out[0].detach(), hidden_out[1].detach())
        # step once, get the next state and reward
        return action, prob, hidden_in, hidden_out

    def store_transition(self, transition):
        _iter = iter(transition)
        self.data['state'].append(next(_iter))
        self.data['action'].append(next(_iter))
        self.data['reward'].append(next(_iter))
        self.data['next_state'].append(next(_iter))
        self.data['action_prob'].append(next(_iter))
        self.data['hidden_in'].append(next(_iter))
        self.data['hidden_out'].append(next(_iter))
        self.data['done_mask'].append(not next(_iter))

    def clear_data(self):
        self.data = defaultdict(list)
        self.lstm_hidden = (torch.randn([1, 1, 128], dtype=torch.float, requires_grad=False), torch.randn([1, 1, 128], dtype=torch.float, requires_grad=False))

    def get_batch(self):
        state = torch.tensor(self.data['state'], dtype=torch.float)
        action = torch.tensor([self.data['action']]).T
        reward = torch.tensor([self.data['reward']]).T
        next_state = torch.tensor(self.data['next_state'], dtype=torch.float)
        action_prob = torch.tensor([self.data['action_prob']]).T
        done_mask = torch.tensor([self.data['done_mask']]).T

        return state, action, reward, next_state, action_prob, \
               self.data['hidden_in'][0], self.data['hidden_out'][0], done_mask

    def update_net(self):
        state, action, reward, next_state, action_prob, (h1_in, h2_in), (h1_out, h2_out), done_mask = self.get_batch()
        self.clear_data()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(self.PPO_EPOCH):
            v_next_state = self.critic_net(next_state, second_hidden).squeeze(1)
            td_target = reward + gamma * v_next_state * done_mask
            v_state = self.critic_net(state, first_hidden).squeeze(1)
            delta = td_target - v_state * done_mask  # td-error
            delta = delta.detach().numpy()

            advantages = []
            advantage = 0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantages.append(advantage)
            advantages = reversed(torch.tensor([advantages], dtype=torch.float).T)

            pi, _ = self.actor_net(state, first_hidden)
            new_action_prob = pi.squeeze(1).gather(1, action)

            # ratio between pi and old
            ratio = torch.exp(torch.log(new_action_prob) - torch.log(action_prob)) # importance weight?


            # SURROGATE
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages

            # LOSS
            a_loss = -torch.min(surrogate1, surrogate2).mean() # make sure strategy not too far away from the old one
            v_loss = F.smooth_l1_loss(v_state, td_target.detach()).mean()
            # e_loss = -Categorical(pi).entropy()


            # OPTIMIZER
            self.acotr_optimizer.zero_grad()
            a_loss.backward(retain_graph=True)
            self.acotr_optimizer.step()

            self.critic_optimizer.zero_grad()
            v_loss.backward(retain_graph=True)
            self.critic_optimizer.step()
