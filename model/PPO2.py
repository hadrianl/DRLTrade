#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 0009 10:11
# @Author  : Hadrianl 
# @File    : PPO


import gym
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

# Hyperparameters
learning_rate = 0.0001
gamma = 0.98
lmbda = 0.95
ent = 0.001
eps_clip = 0.1
K_epoch = 3
T_horizon = 20


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """

        :param state_dim: not include ohlcv and pos
        :param action_dim:
        """
        super().__init__()
        self.data = defaultdict(list)

        self.ohlc_con = nn.Linear(4, 96)
        self.vol_con = nn.Linear(1, 16)
        self.pos_con = nn.Linear(1, 16)
        self.ohlc_expand_con = nn.Linear(96, 96)
        self.vol_expand_con = nn.Linear(16, 16)
        self.pos_expand_con = nn.Linear(16, 16)
        self.all_con = nn.Linear(128, 256)
        self.lstm = nn.LSTM(256, 128)
        self.fc_actor = nn.Linear(128, action_dim)
        self.fc_critic = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, hidden):
        return self.act(x, hidden)

    def act(self, x, hidden):
        # x -> o h l c v p
        ohlc = self.ohlc_con(x[:4])
        vol = self.vol_con(x[4:5])
        pos = self.pos_con(x[5:6])

        ohlc_e = self.ohlc_expand_con(ohlc)
        vol_e = self.vol_expand_con(vol)
        pos_e = self.pos_expand_con(pos)
        e = torch.cat([ohlc_e, vol_e, pos_e])
        x = F.relu(self.all_con(e))

        x = x.view(-1, 1, 256)
        x, hidden = self.lstm(x, hidden)
        x = self.fc_actor(x)
        prob = F.softmax(x, dim=2)
        return prob, hidden

    def pi(self, x, hidden):
        ohlc = self.ohlc_con(x[:, :4])
        vol = self.vol_con(x[:, 4:5])
        pos = self.pos_con(x[:, 5:6])

        ohlc_e = self.ohlc_expand_con(ohlc)
        vol_e = self.vol_expand_con(vol)
        pos_e = self.pos_expand_con(pos)
        e = torch.cat([ohlc_e, vol_e, pos_e], dim=1)
        x = F.relu(self.all_con(e))

        x = x.view(-1, 1, 256)
        x, hidden = self.lstm(x, hidden)
        x = self.fc_actor(x)
        prob = F.softmax(x, dim=2)
        return prob, hidden

    def criticize(self, x, hidden):
        ohlc = self.ohlc_con(x[:, :4])
        vol = self.vol_con(x[:, 4:5])
        pos = self.pos_con(x[:, 5:6])

        ohlc_e = self.ohlc_expand_con(ohlc)
        vol_e = self.vol_expand_con(vol)
        pos_e = self.pos_expand_con(pos)
        e = torch.cat([ohlc_e, vol_e, pos_e], dim=1)
        x = F.relu(self.all_con(e))

        x = x.view(-1, 1, 256)
        x, _ = self.lstm(x, hidden)
        v = self.fc_critic(x)
        return v

    def put_data(self, transition):
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

    def get_batch(self):
        state = torch.tensor(self.data['state'], dtype=torch.float)
        action = torch.tensor([self.data['action']]).T
        reward = torch.tensor([self.data['reward']]).T
        next_state = torch.tensor(self.data['next_state'], dtype=torch.float)
        action_prob = torch.tensor([self.data['action_prob']]).T
        done_mask = torch.tensor([self.data['done_mask']]).T

        return state, action, reward, next_state, action_prob, \
               self.data['hidden_in'][0], self.data['hidden_out'][0], done_mask

    # @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def update_net(self):
        state, action, reward, next_state, action_prob, (h1_in, h2_in), (h1_out, h2_out), done_mask = self.get_batch()
        self.clear_data()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_next_state = self.criticize(next_state, second_hidden).squeeze(1)
            td_target = reward + gamma * v_next_state * done_mask
            v_state = self.criticize(state, first_hidden).squeeze(1)
            delta = td_target - v_state * done_mask
            delta = delta.detach().numpy()

            advantages = []
            advantage = 0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantages.append(advantage)
            advantages = reversed(torch.tensor([advantages], dtype=torch.float).T)

            pi, _ = self.pi(state, first_hidden)
            pi_action = pi.squeeze(1).gather(1, action)

            # ratio between pi and old
            ratio = torch.exp(torch.log(pi_action) - torch.log(action_prob))


            # SURROGATE
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages

            # LOSS
            pi_loss = -torch.min(surrogate1, surrogate2) # make sure strategy not too far away from the old one
            v_loss = F.smooth_l1_loss(v_state, td_target.detach())
            # e_loss = -Categorical(pi).entropy()
            loss = pi_loss + v_loss
            print(pi_loss.mean(), v_loss.mean())

            # OPTIMIZER
            self.optimizer.zero_grad()
            loss_mean = loss.mean()
            Writer.add_scalar('Loss', loss_mean)
            loss_mean.backward(retain_graph=True) # backward K_epoch, so retain the graph
            self.optimizer.step()

