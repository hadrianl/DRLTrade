#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 0009 11:32
# @Author  : Hadrianl 
# @File    : train_loop

from model.PPO import ActorCritic
from env.ohlcv import OHLCVEnv
import torch
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import datetime as dt
from pathlib import Path

Env = OHLCVEnv()
Model = ActorCritic(5, 3)
model_name = 'PPOModel'
writer = SummaryWriter('run/ppo')

def load_params(path=''):
    print('load_params')
    try:
        path = Path.joinpath(Path(path), model_name)
        Model.load_state_dict(torch.load(path))
        Model.eval()
    except Exception as e:
        print(f'no model is found, {e}')
    finally:
        print(f'load model state: {Model.state_dict()}')

def save_params(path=''):
    print('save_params')
    path = Path.joinpath(Path(path), model_name)
    torch.save(Model.state_dict(), path)


def main():
    load_params()
    total_profit = 0
    for n_epi in range(1000):
        hidden_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        state = Env.reset()
        isDone = False

        print(f'{dt.datetime.now()}   episode: {n_epi} begin')
        while not isDone:
            hidden_in = hidden_out
            prob, hidden_out = Model.act(torch.from_numpy(state).float(), hidden_in)
            prob = prob.view(-1)
            action = Categorical(prob).sample().item()
            # step once, get the next state and reward
            next_state, reward, isDone, profit = Env.step(action)

            Model.put_data((state, action, reward, next_state, prob[action].item(), hidden_in, hidden_out, isDone))

            state = next_state
        print(f'{dt.datetime.now()}   episode: {n_epi} end')
        Model.update_net()
        total_profit += profit
        print(f'{dt.datetime.now()}   episode: {n_epi}, profit: {profit} total profit: {total_profit}')

        if (n_epi + 1) % 100 == 0:
            save_params()


if __name__ == '__main__':
    main()