#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 0009 11:00
# @Author  : Hadrianl 
# @File    : ohlc2order


from utils import HKMarketDataBaseDocument, load_json_settings
import random
import datetime as dt
import numpy as np
from mongoengine import register_connection
import torch

codes = [f'HSI{y}{m:02}' for y in range(11, 21) for m in range(1, 13)][:-7]


class OHLCVEnv:
    def __init__(self, fee=19.05, initial_capital=200000, margin=150000, period='1min'):
        self.initial_capital = initial_capital
        self.fee = fee
        self.done = False
        self.current_nbar = 0
        self.position = 0
        self.margin = margin
        self.pnl = 0
        self.multiplier = 50
        self._data_source = type(f'HKMarketData_{period}',
                                (HKMarketDataBaseDocument,),
                                {'meta': {'collection': f'future_{period}_'}})
        db_config = load_json_settings('mongodb_settings.json')
        register_connection('HKFuture', db='HKFuture',
                            host=db_config['host'], port=db_config['port'],
                            username=db_config['user'], password=db_config['password'],
                            authentication_source='admin')

    def get_sample(self):
        c = random.choice(codes)
        d = dt.datetime(year=int(f'20{c[3:5]}'), month=int(c[5:]), day=1)
        total_sample = self._data_source.objects(code=c, datetime__gte=d)
        count = total_sample.count()
        start = random.randint(0, count - 1000)
        return total_sample[start:start + 500].values_list('open', 'high', 'low', 'close', 'volume')

    def reset(self):
        self.balance = self.initial_capital
        self.position = 0
        self.current_nbar = 0
        self.market_value = 0
        self.pnl = 0
        self.data = self.get_sample()
        v = self.data[self.current_nbar]
        self.init_state = np.array([v[0]] * 4 + [0])
        ohlcv = np.array(v) - self.init_state
        return ohlcv

    # def step(self, action):
    #     isDone = False
    #     fee = 0
    #     if action == 1:
    #         if self.position >= 0:
    #             if self.balance > self.margin:
    #                 self.position += 1
    #                 self.balance -= self.margin
    #                 fee = self.fee
    #         else:
    #             self.position += 1
    #             self.balance += self.margin
    #             fee = self.fee
    #     elif action == 2:
    #         if self.position <= 0:
    #             if self.balance > self.margin:
    #                 self.position -= 1
    #                 self.balance -= self.margin
    #                 fee = self.fee
    #         else:
    #             self.position -= 1
    #             self.balance += self.margin
    #             fee = self.fee
    #
    #     current_close = self.data[self.current_nbar][3]
    #     self.current_nbar += 1
    #     if self.current_nbar == 499:
    #         isDone = True
    #     next_close = self.data[self.current_nbar][3]
    #
    #     reward = self.position * (next_close - current_close) * self.multiplier - fee
    #     self.pnl += reward
    #     if self.pnl <= -self.initial_capital * 0.2:
    #         isDone = True
    #
    #     next_state = np.array(self.data[self.current_nbar]) - self.init_state
    #
    #     return next_state, reward, isDone, self.pnl

    # adjust pos instead of order
    def step(self, action):
        isDone = False
        fee = 0
        if action == 0:
            if self.position != 0:
                fee = self.fee * abs(self.position)
            self.position = 0
        elif action == 1:
            if self.position <= 0:
                pos = 1 - self.position
                fee = pos * self.fee
            self.position = 1
        elif action == 2:
            if self.position >= 0:
                pos = self.position + 1
                fee = pos * self.fee
            self.position = -1

        current_close = self.data[self.current_nbar][3]
        self.current_nbar += 1
        if self.current_nbar == 499:
            isDone = True
        next_close = self.data[self.current_nbar][3]

        reward = self.position * (next_close - current_close) * self.multiplier - fee
        self.pnl += reward
        if self.pnl <= -self.initial_capital * 0.2:
            isDone = True

        next_state = np.array(self.data[self.current_nbar]) - self.init_state

        return next_state, reward, isDone, self.pnl