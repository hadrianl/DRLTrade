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
import calendar

codes = [f'HSI{y}{m:02}' for y in range(18, 21) for m in range(1, 13)][:-7]


class OHLCVPEnv:
    def __init__(self, fee=19.05, initial_capital=200000, margin=150000, period='1min'):
        self.data = None
        self.raw_data = None
        self.initial_capital = initial_capital
        self.fee = fee
        self.done = False
        self.current_nbar = 0
        self.position = 0
        self.total_trades = 0
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
        start = random.randint(500, count - 1000)
        data = torch.tensor(total_sample[start:start + 500].values_list('open', 'high', 'low', 'close', 'volume'), dtype=torch.float)
        pre_data = torch.tensor(total_sample[start-500:start].values_list('open', 'high', 'low', 'close', 'volume'),
                                dtype=torch.float)
        return (data - pre_data.mean(0))/pre_data.std(0), data

    def get_daily_sample(self):
        c = random.choice(codes)
        cal = calendar.Calendar()
        year = int(f'20{c[3:5]}')
        month = int(c[5:])
        ds = []
        for day, weekday in cal.itermonthdays2(year, month):
            if day != 0 and weekday not in [5, 6]:
                ds.append(day)

        day = random.choice(ds[:-2])
        d = dt.datetime(year=year, month=month, day=day, hour=9, minute=0)
        total_sample = self._data_source.objects(code=c, datetime__gte=d)
        pre_sample = self._data_source.objects(code=c, datetime__lt=d)
        pre_count = pre_sample.count()
        if pre_count < 500:
            return self.get_daily_sample()
        pre_data = torch.tensor(pre_sample[pre_count-500:pre_count].values_list('open', 'high', 'low', 'close', 'volume'), dtype=torch.float)
        sample = total_sample[:500]
        print(sample.first().datetime)
        data = torch.tensor(sample.values_list('open', 'high', 'low', 'close', 'volume'), dtype=torch.float)
        return ((data - pre_data.mean(0))/pre_data.std(0), data) if sample.count() >= 500 else self.get_daily_sample()

    def reset(self, daily=False):
        self.balance = self.initial_capital
        self.position = 0
        self.total_trades = 0
        self.current_nbar = 0
        self.market_value = 0
        self.pnl = 0
        self.data, self.raw_data = self.get_daily_sample() if daily else self.get_sample()
        v = self.data[self.current_nbar]
        # self.init_state = np.array([v[0]] * 4 + [0, 0])
        # ohlcv = np.concatenate([v, [self.position]]) - self.init_state
        ohlcv = torch.cat([v, torch.tensor([self.position],dtype=torch.float)])
        return ohlcv

    def step(self, action):
        isDone = False
        fee = 0
        pos = 0
        extra_rewards = 0
        if action == 0:
            if self.position != 0:
                pos = abs(self.position)
                fee = self.fee * pos
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

        self.total_trades += pos

        current_close = self.raw_data[self.current_nbar, 3]
        self.current_nbar += 1
        if self.current_nbar == 499:
            # self.pnl -= self.initial_capital * 0.08 / 250 / 50
            isDone = True

        next_close = self.raw_data[self.current_nbar, 3]

        reward = self.position * (next_close - current_close) / self.multiplier + extra_rewards
        # print(f'pos: {self.position} fee: {fee} reward: {reward} cur_close: {current_close} next_close: {next_close}')
        self.pnl += reward
        if self.pnl <= -self.initial_capital * 0.2:
            # self.pnl -= self.initial_capital * 0.08 / 250 / 50
            isDone = True

        next_state = torch.cat([self.data[self.current_nbar], torch.tensor([self.position], dtype=torch.float)])

        return next_state, reward, isDone, self.pnl, pos