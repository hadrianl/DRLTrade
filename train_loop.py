#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 0009 11:32
# @Author  : Hadrianl 
# @File    : train_loop

from model.PPO3 import PPO
from env.ohlcvp import OHLCVPEnv
import torch
from torch.distributions import Categorical
import datetime as dt
from pathlib import Path
from utils import Writer, Writer_test
import sys
import random
import calendar
import numpy as np

Env = OHLCVPEnv()
Model = PPO(6, 3)
model_name = 'PPOModel'
Writer.add_graph(Model.actor_net, [torch.zeros(6, dtype=torch.float), (torch.zeros([1, 1, 128], dtype=torch.float), torch.zeros([1, 1, 128], dtype=torch.float))])

def set_test_env(env,  daily=False):
    env.balance = env.initial_capital
    env.position = 0
    env.current_nbar = 0
    env.market_value = 0
    env.pnl = 0
    # if env.data is None:
    #     env.data = env.get_sample() # get_sample once
    env.data, env.raw_data = get_env_test_sample(env, daily)
    v = env.data[env.current_nbar]
    ohlcv = torch.cat([v, torch.tensor([env.position], dtype=torch.float)])
    return ohlcv

def get_env_test_sample(env, daily=False):
    year = 2020
    month = 7
    if not daily:
        d = dt.datetime(year=year, month=month, day=1)
        total_sample = env._data_source.objects(code='HSI2007', datetime__gte=d)
        count = total_sample.count()
        start = random.randint(0, count - 1000)
        return torch.tensor(total_sample[start:start + 500].values_list('open', 'high', 'low', 'close', 'volume'),
                            dtype=torch.float)
    else:
        cal = calendar.Calendar()
        ds = []
        for day, weekday in cal.itermonthdays2(year, month):
            if day != 0 and weekday not in [5, 6]:
                ds.append(day)

        day = random.choice(ds[:-2])
        d = dt.datetime(year=year, month=month, day=day, hour=9, minute=0)
        total_sample = env._data_source.objects(code='HSI2007', datetime__gte=d)
        sample = total_sample[:500]
        print(sample.first().datetime)
        data = torch.tensor(sample.values_list('open', 'high', 'low', 'close', 'volume'), dtype=torch.float)
        return ((data - data.mean(0)) / data.std(0), data) if total_sample.count() >= 500 else get_env_test_sample(env, daily)
        # return torch.tensor(total_sample[: 500].values_list('open', 'high', 'low', 'close', 'volume'), dtype=torch.float)

def save_params(path=''):
    print('save_params')
    path = Path.joinpath(Path(path), model_name)
    torch.save(Model.state_dict(), path)


def main():
    Model.load_params()
    total_profit = 0
    for n_epi in range(100000):
        state = Env.reset(daily=True)
        isDone = False

        print(f'{dt.datetime.now()}   episode: {n_epi} begin')
        total_pos = 0
        while not isDone:
            action, prob, hidden_in, hidden_out = Model.select_action(state)
            # step once, get the next state and reward
            next_state, reward, isDone, profit, pos = Env.step(action)
            total_pos += pos
            Model.store_transition((state, action, reward, next_state, prob[action].item(), hidden_in, hidden_out, isDone))

            state = next_state
        print(f'{dt.datetime.now()}   episode: {n_epi} end')
        print(f'{dt.datetime.now()}   episode: {n_epi} update net')
        Model.update_net()
        total_profit += profit
        Writer.add_scalar('profit', profit)
        Writer.add_scalar('total_profit', total_profit)
        print(f'{dt.datetime.now()}   episode: {n_epi}, profit: {profit} total profit: {total_profit} total pos: {total_pos}')

        if (n_epi + 1) % 100 == 0:
            Model.save_params()

def test():
    print('test')
    Model.load_params()
    total_profit = 0
    for n_epi in range(1000):
        state = set_test_env(Env, True)
        isDone = False

        print(f'{dt.datetime.now()}   episode: {n_epi} begin')
        total_pos = 0
        n = 0
        while not isDone:
            n += 1

            action, prob, hidden_in, hidden_out = Model.select_action(state)
            # step once, get the next state and reward
            next_state, reward, isDone, profit, pos = Env.step(action)
            # if pos != 0:
            #     print(f'Action:{action}  at {n} bar')
            total_pos += pos
            state = next_state
        Writer_test.add_scalar('profit', profit)
        Writer_test.add_scalar('total_pos', total_pos)
        Writer_test.add_scalar('total_profit', total_profit)
        print(f'{dt.datetime.now()}   episode: {n_epi} end')
        print(f'{dt.datetime.now()}   episode: {n_epi} update net')
        total_profit += profit

        print(f'{dt.datetime.now()}   episode: {n_epi}, profit: {profit} total profit: {total_profit} total pos: {total_pos}')




if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 1:
        main()
    else:
        mode = sys.argv[1]
        if mode == 'train':
            main()
        elif mode == 'test':
            test()